import numpy as np
import tensorflow as tf

def positive_constraint(weights):
    return tf.where(weights <= 0., 0.0001, weights)


def remove_empty_vocab(word_freqs, docs):
    cols = np.argwhere(word_freqs.sum(axis=0)!=0).reshape(-1)
    word_freqs_reduced = word_freqs[:,cols]
    word_lookup_map = dict(zip(cols+1, np.arange(len(cols))+1))
    word_lookup_map_inv = dict(zip(np.arange(len(cols))+1, cols+1))
    for i in range(docs.shape[0]):
        for j in range(docs.shape[1]):
            if docs[i,j]!=-1:
                docs[i,j] = word_lookup_map[docs[i,j]]
            else:
                docs[i,j] = 1
    return word_freqs_reduced, docs, word_lookup_map_inv

def print_topic(words):
    s = ""
    with open("data/vocab.txt") as f:
        for i,x in enumerate(f):
            if i+1 in words:
                s += x[:-1] + " "
    print(s)