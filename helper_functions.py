from scipy.special import gamma as gamma_func
from scipy.special import digamma, polygamma, loggamma
import numpy as np
import matplotlib.pyplot as plt

def bow_to_onehot(X):
	M,V = X.shape
	X_one_hot = []
	for d in range(M):
		to_stack = [np.pad(np.ones((freq,1)), [(0,0),(i,V-i-1)], mode='constant') for i,freq in enumerate(X[d])]
		X_one_hot.append(np.concatenate(to_stack, axis=0))
		if d%10==0:
			print("{} documents transformed".format(d))
	return X_one_hot

