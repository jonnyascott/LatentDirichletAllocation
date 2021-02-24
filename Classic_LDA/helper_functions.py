from scipy.special import gamma as gamma_func
from scipy.special import digamma, polygamma, loggamma
import numpy as np
import matplotlib.pyplot as plt

def bow_to_onehot(X):
	M,V = X.shape
	X_one_hot = []
	for d,x in enumerate(X):
		to_stack = [np.pad(np.ones((freq,1)), [(0,0),(i,V-i-1)], mode='constant') for i,freq in enumerate(x)]
		X_one_hot.append(np.concatenate(to_stack, axis=0))
		if (d+1)%10==0:
			print("{} documents transformed".format(d+1))
	return X_one_hot

def topic_words(topic_word_probs, word_lookup, words_per_topic=10):
	num_topics = topic_word_probs.shape[0]
	top_word_indices = np.argpartition(topic_word_probs, -words_per_topic)[:, -words_per_topic]
	for idxs in top_word_indices:
		words = [word_lookup[i] for i in idxs]
		print(words)


