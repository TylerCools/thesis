import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json

def reduce_vector_dimension(word_vectors, new_dimension):
    assert word_vectors.shape[1] > new_dimension
    #Reduce the dimension using the marix of word vectros
    pca = decomposition.PCA(n_components=new_dimension)
    pca.fit(word_vectors)
    reduced_vectors = pca.transform(word_vectors)

    return reduced_vectors      

def load_embeddings_and_vocab(location):
    ''' This function loads pre-trained embeddings from a location.
    The function assumes that the location has a numpy array with embeddings called embeddings.npy
    and a vocabulary file named w2idx.json'''

    
    #Load the numpy array from disk
    try:
        embedding_location = os.path.join(location, 'embeddings.npy')
        embeddings = np.load(embedding_location)
    except Exception as ex:
        print('Loading the embeddings failed', ex)
        exit(1)
        
    #Load the vocabulary file from disk
    w2idx_location = os.path.join(location, 'w2idx.json')
    w2idx = json.load(open(w2idx_location))
    
    return embeddings, w2idx

def load_embeddings(location, w2idx, dimension, sigma=0.1):
    '''This function takes a vocabulary, dimension and vector location to load pre-trained embeddings.
        For every word that is not in the vocabulary of the pre-trained embeddings a random vector is initialized
     '''

    embeddings, embedding_w2idx = load_embeddings_and_vocab(location)
    embedding_dimension = embeddings.shape[1]
    if embedding_dimension < dimension:
        print('Embedding dimension are to low.')
        exit(1)

    #initialize empty embeddings
    word_embeddings =np.zeros([len(w2idx), embedding_dimension])
    not_in_vocab = 0
    # load the embeddings for the word in the embedding vocabulary and initialize random vectors 
    #for words not in the vocabulary
    for word, index in w2idx.items():
        #If the word is in the vocabulary of the embedding, load the vector
        if word in embedding_w2idx.keys():
            index_np = embedding_w2idx[word]
            word_embeddings[index,:] = embeddings[index_np,:]
        else:

            #The word is not in the VOCABulary of the wordvectors. A random vector is created
            not_in_vocab +=1
            word_embeddings[index,:] = np.random.normal(0, sigma, embedding_dimension)

    #Reduce the dimension if needed
    if embedding_dimension > dimension:
        word_embeddings = reduce_vector_dimension(word_embeddings, dimension)
    #convert to a tensor and return
    return tf.convert_to_tensor(word_embeddings, dtype=tf.float32)  





