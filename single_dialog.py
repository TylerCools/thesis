from __future__ import absolute_import
from __future__ import print_function

import inputs_babi_dialog
import initialize_embeddings

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
tf.flags.DEFINE_boolean('interactive', False, 'if True, interactive')
tf.flags.DEFINE_boolean('OOV', False, 'if True, use OOV test set')
tf.flags.DEFINE_boolean('source', False, 'if True, use Source Awareness')
FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)

# METADATA_PER_TASK = {
#     1: 'dialog-babi-task1-API-calls_metadata.json',
#     2: 'dialog-babi-task2-API-refine_metadata.json',
#     3: 'dialog-babi-task3-options_metadata.json',
#     4: 'dialog-babi-task4-phone-address_metadata.json',
#     5: 'dialog-babi-task5-full-dialogs_metadata.json',
#     6: 'dialog-babi-task6-dstc2_metadata.json'
# }

class chatBot(object):
    def __init__(self, data_dir, model_dir, task_id, source, epochs, isInteractive=True, OOV=False, memory_size=50, 
        random_state=None, batch_size=32, learning_rate=0.001, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, 
        hops=3, embedding_size=20):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        self.isInteractive = isInteractive
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
        candidates, self.candid2indx = load_candidates(self.data_dir, self.task_id)
        # print(candidates)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key], key) for key in self.candid2indx)

        # task data
        self.trainData, self.testData, self.valData = load_dialog_task(self.data_dir, self.task_id, self.candid2indx, self.OOV)
        data = self.trainData + self.testData + self.valData
        self.build_vocab(data, candidates)

        # print(trainData)

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
                                    source=self.source) 
        self.saver = tf.train.Saver(max_to_keep=50)
        self.summary_writer = tf.summary.FileWriter(self.model.root_dir, self.model.graph_output.graph)

    def build_vocab(self, data, candidates):
        # for i in data:
        #     # print('line: {}\n'.format(i))
            # print('context: {}\n'.format(i[0]))
        #     # print('system: {}\n'.format(i[1]))
        #     print('whole_u: {}\n'.format(i[2]))
            # print('whole_s: {}\n'.format(i[5]))
        vocab = reduce(lambda x, y: x | y, (set(list(chain.from_iterable(st)) + us) for st, us, sy, w_u, w_s, an in data))
        vocab |= reduce(lambda x, y: x | y, (set(candidate)
                                             for candidate in candidates))
        vocab = sorted(vocab)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (st for st, us, sy, w_u, w_s, an in data)))
        mean_story_size = int(np.mean([len(st) for st, us, sy, w_u, w_s, an in data]))
        self.sentence_size = max(map(len, chain.from_iterable(st for st, us, sy, w_u, w_s, an in data)))
        self.candidate_sentence_size = max(map(len, candidates))
        query_size = max(map(len, (us for st, us, sy, w_u, w_s, an in data)))
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


    def interactive(self):
        context = []
        u = None
        r = None
        nid = 1
        while True:
            line = input('--> ').strip().lower()
            if line == 'exit':
                break
            if line == 'restart':
                context = []
                nid = 1
                print("clear memory")
                continue
            u = tokenize(line)
            data = [(context, u, -1)]
            story, query, system, answer, wholeU, wholeS = vectorize_data(
                data, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            preds = self.model.predict(story, wholeU, wholeS, q, self.source)
            r = self.indx2candid[preds[0]]
            # print(r)
            r = tokenize(r)
            u.append('$u')
            u.append('#' + str(nid))
            r.append('$r')
            r.append('#' + str(nid))
            context.append(u)
            context.append(r)
            nid += 1  
        
    def train(self):
        trainStory, trainQuery, trainSystem, trainAnswer, trainWholeU, trainWholeS = vectorize_data(self.trainData, 
            self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        valStory, valQuery, valSystem, valAnswer, valWholeU, valWholeS = vectorize_data(self.valData, 
            self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
        n_train = len(trainStory)
        n_val = len(valStory)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size), range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy = 0
        
        for t in range(1, self.epochs + 1):
            print('Epoch', t)
            np.random.shuffle(batches)
            total_cost = 0.0
            total_cost_val = 0.0
            for start, end in batches:
                story = trainStory[start:end]
                query = trainQuery[start:end]
                system = trainSystem[start:end]
                answer = trainAnswer[start:end]
                whole_u = trainWholeU[start:end]
                whole_s = trainWholeS[start:end]
                print(valWholeU)
                # print('story length: {}\n'.format(story))
                cost_t = self.model.batch_fit(story, whole_u, whole_s, query, answer, FLAGS.source)
                # cost_t_val = self.model.batch_fit(valStory[start:end], valWholeU[start:end], valWholeS[start:end], valQuery[start:end], valAnswer[start:end], FLAGS.source)
                total_cost += cost_t
                # total_cost_val += cost_t_val
            if t % self.evaluation_interval == 0:
                train_preds = self.batch_predict(trainStory, trainWholeU, trainWholeS, trainQuery, n_train)
                val_preds = self.batch_predict(valStory, valWholeU, valWholeS, valQuery, n_val)
                train_acc = metrics.accuracy_score(np.array(train_preds), trainAnswer)
                val_acc = metrics.accuracy_score(val_preds, valAnswer)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost Training:', total_cost)
                print('Total Cost Validation:', total_cost_val)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')

                # write summary
                train_acc_summary = tf.summary.scalar('task_' + str(self.task_id) + '/' + 'train_acc', tf.constant((train_acc), dtype=tf.float32))
                val_acc_summary = tf.summary.scalar('task_' + str(self.task_id) + '/' + 'val_acc', tf.constant((val_acc), dtype=tf.float32))
                merged_summary = tf.summary.merge([train_acc_summary, val_acc_summary])
                summary_str = self.sess.run(merged_summary)
                self.summary_writer.add_summary(summary_str, t)
                self.summary_writer.flush()

                if val_acc > best_validation_accuracy:
                    best_validation_accuracy = val_acc
                    self.saver.save(self.sess, self.model_dir +
                                    'model.ckpt', global_step=t)

    def test(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        if self.isInteractive:
            self.interactive()
        else:
            testStory, testQuery, testSystem, testAnswer, testWholeU, testWholeS = vectorize_data(self.testData, 
                self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            n_test = len(testStory)
            print("Testing Size", n_test)
            test_preds = self.batch_predict(testStory, testWholeU, testWholeS, testQuery, n_test)
            test_acc = metrics.accuracy_score(test_preds, testAnswer)
            print("Testing Accuracy:", test_acc)

    def batch_predict(self, story, whole_u, whole_s, query, n):
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            storY = story[start:end]
            wholeU = whole_u[start:end]
            wholeS = whole_s[start:end]
            q = query[start:end]
            pred = self.model.predict(storY, wholeU, wholeS, q, self.source)
            preds += list(pred)
        return preds

    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    model_dir = "task" + str(FLAGS.task_id) + "_" + FLAGS.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(FLAGS.data_dir, model_dir, FLAGS.task_id, OOV=FLAGS.OOV,
                      isInteractive=FLAGS.interactive, batch_size=FLAGS.batch_size, source=FLAGS.source, epochs=FLAGS.epochs)
    # chatbot.run()
    if FLAGS.train == True:
        chatbot.train()
    else:
        chatbot.test()
    chatbot.close_session()
