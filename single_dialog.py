from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, vectorize_candidates_sparse, tokenize
from sklearn import metrics
from memn2n import MemN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import tensorflow as tf
import numpy as np
import os
import json
from matplotlib import pyplot as plt

# Define all flags with their description
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 6, "bAbI task id, 1 <= id <= 6")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/dialog-bAbI-tasks/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "model/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_boolean('train', False, 'if True, begin to train')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
tf.flags.DEFINE_boolean('source', False, 'if True, use Source Awareness')
tf.flags.DEFINE_boolean('resFlag', False, 'if True, use Result Source Awareness')
tf.flags.DEFINE_boolean('wrong_conversations', False, 'if True, show predicted answers')
tf.flags.DEFINE_boolean('error', False, 'if True, do error inspecting')
tf.flags.DEFINE_boolean('acc_each_epoch', False, 'if True, do accuracy each epoch')
tf.flags.DEFINE_boolean('acc_ten_epoch', False, 'if True, do accuracy every tenth epoch')
tf.flags.DEFINE_boolean('conv_wrong_right', False, 'if True, give a list of conversations with right answer and wrong answer')

FLAGS = tf.flags.FLAGS

class chatBot(object):
    def __init__(self, data_dir, model_dir, task_id, source, resFlag, wrong_conversations, error, acc_each_epoch, acc_ten_epoch, 
        conv_wrong_right, epochs, OOV=False,  memory_size=50, random_state=None, batch_size=32, learning_rate=0.001, 
        epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, hops=3, embedding_size=20):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        self.OOV = OOV
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.source = source
        self.resFlag = resFlag
        self.wrong_conversations = wrong_conversations
        self.error = error
        self.acc_each_epoch = acc_each_epoch
        self.acc_ten_epoch = acc_ten_epoch
        candidates, self.candid2indx = load_candidates(self.data_dir, self.task_id)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key], key) for key in self.candid2indx)

        # create train, test and validation data
        self.trainData, self.testData, self.valData = load_dialog_task(self.data_dir, self.task_id, self.candid2indx, self.OOV)
        data = self.trainData + self.testData + self.valData
        self.build_vocab(data, candidates)

        self.test_acc_list = []
        self.candidates_vec = vectorize_candidates(candidates, self.word_idx, self.candidate_sentence_size)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
        self.sess = tf.Session()
        self.model = MemN2NDialog(  self.batch_size, 
                                    self.vocab_size,    
                                    self.n_cand, 
                                    self.sentence_size, 
                                    self.embedding_size, 
                                    self.candidates_vec, 
                                    session=self.sess,
                                    hops=self.hops, 
                                    max_grad_norm=self.max_grad_norm, 
                                    optimizer=optimizer, 
                                    task_id=task_id,
                                    source=self.source,
                                    resFlag=self.resFlag,
                                    oov = self.OOV) 
        self.saver = tf.train.Saver(max_to_keep=50)
        self.summary_writer = tf.summary.FileWriter(self.model.root_dir, self.model.graph_output.graph)

    def build_vocab(self, data, candidates):
        vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(st)) + us) for st, us, sy, w_u, w_s, an, re in data))
        vocab |= reduce(lambda x, y: x | y, (set(candidate)
                                             for candidate in candidates))
        vocab = sorted(vocab)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (st for st, us, sy, w_u, w_s, an, re in data)))
        mean_story_size = int(np.mean([len(st) for st, us, sy, w_u, w_s, an, re in data]))
        self.sentence_size = max(map(len, chain.from_iterable(st for st, us, sy, w_u, w_s, an, re in data)))
        self.candidate_sentence_size = max(map(len, candidates))
        query_size = max(map(len, (us for st, us, sy, w_u, w_s, an, re in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        self.sentence_size = max(query_size, self.sentence_size)  # for the position
        
        # params
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length",
              self.candidate_sentence_size)
        print("Longest story length", max_story_size)
        print("Average story length", mean_story_size)


    # Create paths to save the diagrams to
    def create_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    # Function created to plot the loss. All the figures are saved to the corresponding subdirectory.
    def plot_loss(self, train, val, train_acc):  
        x = []
        x = list(range(1, len(train)+1))
        fig = plt.figure()
        plt.plot(x, train)
        plt.plot(x, val)
        plt.xlabel('Epochs')
        plt.ylabel('Loss value')
        plt.legend(["train", "val"])
        if FLAGS.source == True:
            if FLAGS.OOV == True:
                self.create_path('diagrams/loss/source/oov/')
                fig.savefig('diagrams/loss/source/oov/task_{}_source_OOV_{}.png'.format(self.task_id, round(train_acc, 6)))
            else:
                self.create_path('diagrams/loss/source/')
                fig.savefig('diagrams/loss/source/task_{}_source_{}.png'.format(self.task_id, round(train_acc, 6)))
        else:
            if FLAGS.OOV == True:
                self.create_path('diagrams/loss/regular/oov/')
                fig.savefig('diagrams/loss/regular/oov/task_{}_regular_OOV_{}.png'.format(self.task_id, round(train_acc, 6)))
            else:
                self.create_path('diagrams/loss/regular/')
                fig.savefig('diagrams/loss/regular/task_{}_regular_{}.png'.format(self.task_id, round(train_acc, 6)))

    # Function created to plot the accuracy. All the figures are saved to the corresponding subdirectory.
    def plot_acc(self,test_acc): 
        length = len(test_acc)
        x = []
        x = list(range(1, length+1))
        fig = plt.figure()
        plt.plot(x, test_acc)
        plt.xlabel('Epochs')
        plt.ylabel('Test accuracy')
        plt.legend(["Test"])
        if FLAGS.source == True:
            if FLAGS.OOV == True:
                self.create_path('diagrams/accuracy/source/oov/')
                fig.savefig('diagrams/accuracy/source/oov/task_{}_source_OOV_{}.png'.format(self.task_id, round(test_acc[length-1], 6)))
            else:                
                self.create_path('diagrams/accuracy/source/')
                fig.savefig('diagrams/accuracy/source/task_{}_source_{}.png'.format(self.task_id, round(test_acc[length-1], 6)))                
        else: 
            if FLAGS.OOV == True:
                self.create_path('diagrams/accuracy/regular/oov/')
                fig.savefig('diagrams/accuracy/regular/oov/task_{}_regular_OOV_{}.png'.format(self.task_id, round(test_acc[length-1], 6)))
            else:
                self.create_path('diagrams/accuracy/regular/')
                fig.savefig('diagrams/accuracy/regular/task_{}_regular_{}.png'.format(self.task_id, round(test_acc[length-1], 6)))

    # Count amount of mistakes 
    def amount_mistakes(self, real, predicted):
        real = real.split(' ')
        predicted = predicted.split(' ')
        same = 0
        mistake = 0
        for i  in range(len(real)):
            try:
                if real[i] != predicted[i]:
                    mistake += 1
            except IndexError:
                return mistake
        return mistake

    # Print information about the dialogue with the story, predicted answer and real answer
    def print_conv_wrong(self, real, key2, story_words):
        print('real_answer: {}'.format(real))
        print('predicted answer: {}'.format(key2))
        print('story: {}'.format(story_words))
        print('\n-----------------------')  

    # Print information about the type of mistakes made by the system
    def print_mistake_info(self, fu, oam,  tam, tmam, osm, tsm, tmsm, cw, cr):
        print('-----------------------')
        print('Wrong follow up: {}'.format(fu))
        print('One API mistake: {}'.format(oam))
        print('Two API mistake: {}'.format(tam))
        print('Three or more API mistake: {}'.format(tmam))
        print('One wrong suggeston: {}'.format(osm))
        print('Two wrong suggestons: {}'.format(tsm))
        print('Three or more wrong suggestons: {}'.format(tmsm))
        print('-----------------------')
        if FLAGS.conv_wrong_right == True:
            print('Conversations wrong: {}'.format(cw))
            print('Conversations right: {}'.format(cr))

    def print_epoch_info(self, total_cost, total_cost_val, train_acc, val_acc, test_acc):
        print('Total Cost Training:', total_cost)
        print('Total Cost Validation:', total_cost_val)
        print('Training Accuracy:', train_acc)
        print('Validation Accuracy:', val_acc)
        print('\n-----------------------')  
        print("Testing Size", n_test)
        print('Test accuracy list:', test_acc)
        print('\n-----------------------')

    # Inspect what type of errors the system makes and how much each mistake occurs
    def error_inspect(self, test_preds, testAnswer, story_words):
        real_answer = []
        predicted_answer = []
        all_info = []
        differences_all = []
        follow_up_wrong = 0
        bigger_api_mistakes = 0
        should_be_api = 0
        three_or_more_api__call_mistakes = 0
        two_api_mistakes = 0
        one_api_mistake = 0
        one_answer_mistake = 0
        two_answer_mistakes = 0
        three_or_more_answer_mistakes = 0
        conversation_number_wrong = []
        conversation_number_right = []
        for i, answers in enumerate(test_preds):
                if test_preds[i] != testAnswer[i]:
                    for real, value in self.candid2indx.items():   
                        # check which written answer corresponds to the given number by testAnswer. 
                        if value == testAnswer[i]:
                            real_answer.append(real)
                            for key2, value in self.candid2indx.items():
                                if value == test_preds[i]:
                                    if FLAGS.wrong_conversations == True:
                                        self.print_conv_wrong(real, key2, story_words[i])                               
                                    predicted_answer.append(key2)
                                    if 'where' or  'preference' or 'people' or 'range' in real: 
                                        if 'here' not in real:
                                            if'what' not in real: 
                                                if 'api_call' not in real: 
                                                    follow_up_wrong += 1
                                                else:
                                                    mistake = self.amount_mistakes(real, key2)
                                                    if mistake == 1:
                                                        one_api_mistake += 1
                                                    elif mistake == 2:
                                                        two_api_mistakes += 1
                                                    else:
                                                        three_or_more_api__call_mistakes += 1
                                            else:
                                                mistake = self.amount_mistakes(real, key2)
                                                if mistake == 1:
                                                    one_answer_mistake += 1
                                                elif mistake == 2:
                                                    two_answer_mistakes += 1
                                                else:
                                                    three_or_more_answer_mistakes += 1
                                        else:
                                            mistake = self.amount_mistakes(real, key2)
                                            if mistake == 1:
                                                    one_answer_mistake += 1
                                            elif mistake == 2:
                                                two_answer_mistakes += 1
                                            else:
                                                three_or_more_answer_mistakes += 1  
                            conversation_number_wrong.append(story_words[i][1])   
                else:
                    conversation_number_right.append(story_words[i][1])                                                            
        self.print_mistake_info(follow_up_wrong, one_api_mistake, two_api_mistakes, three_or_more_api__call_mistakes, 
            one_answer_mistake, two_answer_mistakes, three_or_more_answer_mistakes, conversation_number_right, conversation_number_wrong)
        return follow_up_wrong, one_api_mistake, two_api_mistakes, three_or_more_api__call_mistakes, one_answer_mistake, \
        two_answer_mistakes, three_or_more_answer_mistakes, conversation_number_right, conversation_number_wrong

    # This function is adapted so it can capture error inspecting. Also this function is adapted so it is able to 
    # handle the Source Awareness part.
    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        testStory, testQuery, testSystem, testAnswer, testWholeU, testWholeS, testResults, story_words = vectorize_data(self.testData, 
            self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        n_test = len(testStory)
        test_preds = self.batch_predict(testStory, testWholeU, testWholeS, testQuery, testResults, n_test)

        if FLAGS.error == True:
            follow_up_wrong, one_api_mistake, two_api_mistakes, three_or_more_api__call_mistakes, one_answer_mistake, two_answer_mistakes, three_or_more_answer_mistakes, conversation_number_right, conversation_number_wrong = self.error_inspect(test_preds, testAnswer, story_words)
        
        test_acc = metrics.accuracy_score(test_preds, testAnswer)
        self.test_acc_list.append(test_acc)
        print("Testing Accuracy:", test_acc)
        print('----------------------\n')  
        if FLAGS.error == True:
            return self.test_acc_list, follow_up_wrong, one_api_mistake, two_api_mistakes, three_or_more_api__call_mistakes, \
            one_answer_mistake, two_answer_mistakes, three_or_more_answer_mistakes, conversation_number_right, conversation_number_wrong  

    # This function is adapted so it can capture the training and validation loss. Also this function is adapted so it is able to 
    # handle the Source Awareness part.
    def train(self):
        trainStory, trainQuery, trainSystem, trainAnswer, trainWholeU, trainWholeS, trainResults, _ = vectorize_data(self.trainData, 
            self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        valStory, valQuery, valSystem, valAnswer, valWholeU, valWholeS, valResults, _ = vectorize_data(self.valData, 
            self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        n_train = len(trainStory)
        n_val = len(valStory)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches_train = zip(range(0, n_train - self.batch_size, self.batch_size), range(self.batch_size, n_train, self.batch_size))
        batches_train = [(start_t, end_t) for start_t, end_t in batches_train]
        batches_val = zip(range(0, n_val - self.batch_size, self.batch_size), range(self.batch_size, n_val, self.batch_size))
        batches_val = [(start_v, end_v) for start_v, end_v in batches_val]
        best_validation_accuracy = 0
        cost_array = []
        cost_val_array = []
        for t in range(1, self.epochs + 1):
            print('Epoch', t)
            np.random.shuffle(batches_train)
            np.random.shuffle(batches_val)
            total_cost = 0.0
            total_cost_val = 0.0
            # Get the right batches to feed the network to calculate the trainingloss
            for start_t, end_t in batches_train:
                cost_t = self.model.batch_fit(trainStory[start_t:end_t], trainWholeU[start_t:end_t], trainWholeS[start_t:end_t], 
                    trainQuery[start_t:end_t], trainAnswer[start_t:end_t], trainResults[start_t:end_t], FLAGS.source, FLAGS.resFlag)
                total_cost += cost_t
            # Get the right batches to feed the network to calculate the validation loss
            for start_v, end_v in batches_val:
                cost_t_val = self.model.batch_fit(valStory[start_v:end_v], valWholeU[start_v:end_v], valWholeS[start_v:end_v], 
                    valQuery[start_v:end_v], valAnswer[start_v:end_v], valResults[start_v:end_v], FLAGS.source, FLAGS.resFlag)
                total_cost_val += cost_t_val
            cost_val_array.append(total_cost_val)
            cost_array.append(total_cost)
            train_preds = self.batch_predict(trainStory, trainWholeU, trainWholeS, trainQuery, trainResults, n_train)
            val_preds = self.batch_predict(valStory, valWholeU, valWholeS, valQuery, valResults, n_val)
            train_acc = metrics.accuracy_score(np.array(train_preds), trainAnswer)
            val_acc = metrics.accuracy_score(val_preds, valAnswer)
            train_acc_summary = tf.summary.scalar('task_' + str(self.task_id) + '/' + 'train_acc', tf.constant((train_acc), dtype=tf.float32))
            val_acc_summary = tf.summary.scalar('task_' + str(self.task_id) + '/' + 'val_acc', tf.constant((val_acc), dtype=tf.float32))
            merged_summary = tf.summary.merge([train_acc_summary, val_acc_summary])
            summary_str = self.sess.run(merged_summary)
            self.summary_writer.add_summary(summary_str, t)
            self.summary_writer.flush()

            if val_acc > best_validation_accuracy:
                best_validation_accuracy = val_acc
                self.saver.save(self.sess, self.model_dir +'model.ckpt', global_step=t)
            if FLAGS.acc_each_epoch == True:
                test_acc, follow_up_wrong, one_api_mistake, two_api_mistakes, three_or_more_api__call_mistakes, 
                one_answer_mistake, two_answer_mistakes, three_or_more_answer_mistakes, conv_right, conv_wrong = self.test()
                if t % self.evaluation_interval == 0:
                    self.print_epoch_info(total_cost, total_cost_val, train_acc, val_acc, test_acc)      
            elif FLAGS.acc_ten_epoch:
                if t % self.evaluation_interval == 0:
                    test_acc, follow_up_wrong, one_api_mistake, two_api_mistakes, three_or_more_api__call_mistakes, 
                    one_answer_mistake, two_answer_mistakes, three_or_more_answer_mistakes, conv_right, conv_wrong = self.test()
                    self.print_epoch_info(total_cost, total_cost_val, train_acc, val_acc, test_acc)  
        if FLAGS.acc_each_epoch == True or FLAGS.acc_ten_epoch == True:
            self.plot_acc(test_acc)
        self.plot_loss(cost_array, cost_val_array, train_acc)

    # This function is adapted so it is able to use all the different dialogue parts.
    def batch_predict(self, story, whole_u, whole_s, query, results, n):
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            storY = story[start:end]
            wholeU = whole_u[start:end]
            wholeS = whole_s[start:end]
            result = results[start:end]
            q = query[start:end]
            pred = self.model.predict(storY, wholeU, wholeS, q, result, self.source, self.resFlag)
            preds += list(pred)
        return preds

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    model_dir = "task" + str(FLAGS.task_id) + "_" + FLAGS.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(FLAGS.data_dir, model_dir, FLAGS.task_id, OOV=FLAGS.OOV,
                      batch_size=FLAGS.batch_size, source=FLAGS.source, epochs=FLAGS.epochs, 
                      resFlag=FLAGS.resFlag, wrong_conversations=FLAGS.wrong_conversations, error=FLAGS.error, 
                      acc_each_epoch=FLAGS.acc_each_epoch, acc_ten_epoch=FLAGS.acc_ten_epoch, conv_wrong_right=FLAGS.conv_wrong_right)
    if FLAGS.train == True:
        chatbot.train()
    else:
        chatbot.test()
    chatbot.close_session()
