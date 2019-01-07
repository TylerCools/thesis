from __future__ import absolute_import

import os
import re
import numpy as np
import tensorflow as tf

stop_words=set(["a","an","the"])


def load_candidates(data_dir, task_id):
    assert task_id > 0 and task_id < 7
    candidates=[]
    candidates_f=None
    candid_dic={}
    if task_id==6:
        candidates_f='dialog-babi-task6-dstc2-candidates.txt'
    else:
        candidates_f='dialog-babi-candidates.txt'
    with open(os.path.join(data_dir,candidates_f)) as f:
        for i,line in enumerate(f):
            candid_dic[line.strip().split(' ',1)[1]] = i
            line=tokenize(line.strip())[1:]
            candidates.append(line)
    # return candidates,dict((' '.join(cand),i) for i,cand in enumerate(candidates))
    return candidates,candid_dic


def load_dialog_task(data_dir, task_id, candid_dic, isOOV):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 7

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'dialog-babi-task{}-'.format(task_id)
    train_file = [f for f in files if s in f and 'trn' in f][0]
    if isOOV:
        test_file = [f for f in files if s in f and 'tst-OOV' in f][0]
    else: 
        test_file = [f for f in files if s in f and 'tst.' in f][0]
    val_file = [f for f in files if s in f and 'dev' in f][0]
    train_data = get_dialogs(train_file,candid_dic)
    test_data = get_dialogs(test_file,candid_dic)
    val_data = get_dialogs(val_file,candid_dic)
    # print(train_data)
    return train_data, test_data, val_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple']
    '''
    sent=sent.lower()
    if sent=='<silence>':
        return [sent]
    result=[x.strip() for x in re.split('(\W+)?', sent) if x.strip() and x.strip() not in stop_words]
    if not result:
        result=['<silence>']
    if result[-1]=='.' or result[-1]=='?' or result[-1]=='!':
        result=result[:-1]
    return result

def parse_dialogs_per_response(lines,candid_dic):
    '''
        Parse dialogs provided in the babi tasks format
        The user and system responses are split here. 
        Also the length of the dialog is being monitored
        with the nid.
    '''
    data=[]
    context=[]
    user= None
    system= None
    system_final = None
    result = None
    whole_system = []
    whole_user = []
    results = []
    # print(candid_dic)
    for line in lines:
        line=line.strip()
        # if ('R_phone' or 'R_cuisine' or 'R_address' or 'R_location' or 'R_number' or 'R_price' or 'R_rating') in tokenize(line):
        if line:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            # print(nid)
            if '\t' in line:
                user, system = line.split('\t')
                # print('system_final: {}'.format(system))
                answer = candid_dic[system]

                # print('answer: {}'.format(candid_dic[answer]))
                system = tokenize(system)
                user = tokenize(user)
                #     result = tokenize(system)
                # else:
                
                # print('result: {}'.format(result))
                # print('system: {}'.format(system_final))
                # print('result: {}'.format(result))
                # print('user: {}'.format(user))
                # temporal encoding, and utterance/response encoding
                # data.append((context[:],u[:],candid_dic[' '.join(r)]))

                user.append('$user')
                user.append('#'+str(nid))
                system.append('$system')
                system.append('#'+str(nid))
                # if result != None:
                #     result.append('$result')
                #     result.append('#'+str(nid))
                #     results.append(result)
                # print(results)
                whole_system.append(system)
                # whole_system.append('$whole_system')
                whole_user.append(user)
                context.append(user)
                context.append(system)
                # whole_user.append('$whole_user')

                data.append([context[:], user[:], system, whole_user, whole_system, answer, results])
                # print('results {}\n'.format(results))
                # print('user: {}\n'.format(user))
                # print('system whole: {}\n'.format(whole_system))
                # print('user whole2: {}\n'.format(whole_user))
                # print(data)
            else:
                # print( 'line: {}'.format(line))
                result=tokenize(line)
                result.append('$result')
                result.append('#'+str(nid))
                context.append(result)
                results.append(result)
        else:
            whole_system = []
            whole_user = []
            context=[]
            results = []
        # print('datalength: {}\n'.format(data))
    return data

def get_dialogs(f,candid_dic):
    '''Given a file name, read the file, retrieve the dialogs, and then convert the sentences into a single dialog.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_dialogs_per_response(f.readlines(),candid_dic)

def vectorize_candidates_sparse(candidates,word_idx):
    shape=(len(candidates),len(word_idx)+1)
    indices=[]
    values=[]
    for i,candidate in enumerate(candidates):
        for w in candidate:
            indices.append([i,word_idx[w]])
            values.append(1.0)
    return tf.SparseTensor(indices,values,shape)

def vectorize_candidates(candidates,word_idx,sentence_size):
    shape=(len(candidates),sentence_size)
    C=[]
    for i,candidate in enumerate(candidates):
        lc=max(0,sentence_size-len(candidate))
        C.append([word_idx[w] if w in word_idx else 0 for w in candidate] + [0] * lc)
    return tf.constant(C,shape=shape)


def vectorize_data(data, word_idx, sentence_size, batch_size, candidates_size, max_memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    System = []
    Query = []
    Story = []
    Answer = []
    WholeU = []
    WholeS = []
    Results = []
    Story_words = []
    data.sort(key=lambda x:len(x[0]),reverse=True)
    for i, (story, query, system, whole_user, whole_system, answer, results) in enumerate(data):
        # print('answer:{}\n'.format(answer))
        # print('whole_system{}\n'.format(story))
        # print('results:{}\n'.format(results))
        if i%batch_size==0:
            memory_size=max(1,min(max_memory_size,len(story)))
        
        stor = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            stor.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
        # take only the most recent sentences that fit in memory
        stor = stor[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(stor))
        for _ in range(lm):
            stor.append([0] * sentence_size)

        sys = []
        for i, sentence in enumerate(system, 1):
            ls = max(0, sentence_size - len(sentence))
            sys.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
        # take only the most recent sentences that fit in memory
        sys = sys[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(sys))
        for _ in range(lm):
            sys.append([0] * sentence_size)

        wu = []
        for i, sentence in enumerate(whole_user, 1):
            ls = max(0, sentence_size - len(sentence))
            wu.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
        # take only the most recent sentences that fit in memory
        wu = wu[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(wu))
        for _ in range(lm):
            wu.append([0] * sentence_size)

        ws = []
        for i, sentence in enumerate(whole_system, 1):
            ls = max(0, sentence_size - len(sentence))
            ws.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
        # take only the most recent sentences that fit in memory
        ws = ws[::-1][:memory_size][::-1]


        # pad to memory_size
        lm = max(0, memory_size - len(ws))
        for _ in range(lm):
            ws.append([0] * sentence_size)

        re = []
        for i, sentence in enumerate(results, 1):
            ls = max(0, sentence_size - len(sentence))
            re.append([word_idx[w] if w in word_idx else 0 for w in sentence] + [0] * ls)
        # take only the most recent sentences that fit in memory
        re = re[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(re))
        for _ in range(lm):
            re.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w in word_idx else 0 for w in query] + [0] * lq


        Story.append(np.array(stor))
        Query.append(np.array(q))
        System.append(np.array(sys))
        Answer.append(np.array(answer))
        WholeU.append(np.array(wu))
        WholeS.append(np.array(ws))
        Results.append(np.array(re))
        Story_words.append(story)
        # print("whole user in vec data: {}".format(WholeU))
    return Story, Query, System, Answer, WholeU, WholeS, Results, Story_words
