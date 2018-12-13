"""
Module responsible for input data.
"""
import re
import os
import json
import tensorflow as tf
import functools

def serialize_example(example, story_length, sentence_length):
    '''This function serializes an example from a tf.records file.
    (bytes to Tensor)'''

    feature_definition = {'story':tf.FixedLenFeature(
                                    shape=[story_length, sentence_length], 
                                    dtype=tf.int64),
                        'query':tf.FixedLenFeature(
                                    shape=[1, sentence_length], 
                                    dtype=tf.int64),
                        'answer': tf.FixedLenFeature(
                                    shape=[],
                                    dtype=tf.int64),
                        'user_story': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64),
                        'system_story': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64),
                        'result_story': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64),
                        'story_tags': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64),
                        'query_tags':tf.FixedLenFeature(
                                    shape=[1, sentence_length], 
                                    dtype=tf.int64),
                        'user_story_tags': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64),
                        'system_story_tags': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64),
                        'result_story_tags': tf.FixedLenFeature(
                                    shape=[story_length, sentence_length],
                                    dtype=tf.int64)}
    return tf.parse_single_example(example, features = feature_definition)
    
    
def create_dataset_iterator(filename, batch_size,epochs, story_length, sentence_length, shuffle=True,buffer_size=1000):
    #Read the raw data
    dataset = tf.data.TFRecordDataset(filenames=filename, compression_type='GZIP', buffer_size=buffer_size)
    # dataset = tf.data.TFRecordDataset(filename, compression_type='GZIP')
    #Shuffle with a buffer
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    #The serialize_example function requires the story length and sentence length, they are the same for every example
    #Also the map function requires a single argument funtion, thus a partial function is created
    serialize_example_partial = functools.partial(serialize_example,
                                                  story_length=story_length,
                                                  sentence_length=sentence_length)
    
    #Serialize examples
    dataset = dataset.map(map_func=serialize_example_partial,
                          num_parallel_calls=2)
    
    #Make batches and epochs
    dataset = dataset.prefetch(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size=batch_size)

    return dataset.make_initializable_iterator()   
    

def make_placeholders(story_length, sentence_length):
    placeholders = {
                        'story': tf.placeholder(name='input/story', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64), 
                        'query': tf.placeholder(name='input/query', 
                                                shape=[None, 1, sentence_length], 
                                                dtype= tf.int64),
                        'user_story': tf.placeholder(name='input/user_story', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64), 
                        'system_story': tf.placeholder(name='input/system_story', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64),                             

                        'result_story': tf.placeholder(name='input/result_story', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64),          

                        'story_tags': tf.placeholder(name='input/story_tags', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64),

                        'query_tags': tf.placeholder(name='input/query_tags', 
                                                shape=[None, 1, sentence_length], 
                                                dtype= tf.int64),

                        'user_story_tags': tf.placeholder(name='input/user_story_tags', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64), 
                        'system_story_tags': tf.placeholder(name='input/system_story_tags', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64),                             

                        'result_story_tags': tf.placeholder(name='input/result_story_tags', 
                                                shape=[None, story_length, sentence_length], 
                                                dtype= tf.int64),
                        'answer': tf.placeholder(name='answer', 
                                                           shape=[None], 
                                                           dtype= tf.int64)
                    }    
    return placeholders





