import math
import numpy as np

from scipy.special import gamma as gamma_func
from scipy.special import loggamma, digamma, polygamma

from sklearn.decomposition import LatentDirichletAllocation as LDA_correct
from helper_functions import bow_to_onehot

class LatentDirichletAllocation:

	def __init__(self, n_components=10, smoothing=False):
		self.k = n_components
		self.smoothing = smoothing

	#Checks if all paramater values are legal
	def legal_params(self,X):
		legal = True
		if (self.params['alpha'] <= 0).any():
			print("Alpha illegal: ", self.params['alpha'])
			legal = False

		if (self.params['beta'] < 0).any() or (not np.allclose(self.params['beta'].sum(axis=1),1)):
			legal = False
			print("Beta illegal: ", self.params['beta'])
			if (self.params['beta'] < 0).any():
				print("Beta negative")
			if not np.allclose(self.params['beta'].sum(axis=1),1):
				print("Beta not summing to 1: ", self.params['beta'].sum(axis=1))


		for d in range(len(X)):
			if (self.params['gamma'][d] <= 0).any():
				legal = False
				print("Gamma {} illegal: ".format(d), self.params['gamma'][d])

			if (self.params['phi'][d] < 0).any() or (not np.allclose(self.params['phi'][d].sum(axis=1),1)):
				legal = False
				print("Phi {} illegal: ".format(d))
				if (self.params['phi'][d] < 0).any():
					print("Phi {} negative".format(d))
				if not np.allclose(self.params['phi'][d].sum(axis=1),1):
					print("Phi {} not summing to 1".format(d))
		if legal:
			print("All parameters legal.")
		

	#Initialize parameters of lda and of approximate posterior distributions
	def initialize_params(self, X):
		M = len(X)
		V = X[0].shape[1]
		params = {}

		#add model parameters
		params['alpha'] = 0.1*np.ones(self.k) + np.random.normal(loc=0, scale=0.001, size=self.k) #TO BE COMPLETED
		if self.smoothing:
			params['eta'] = 0.01 #TO BE COMPLETED
		else:
			params['beta'] = 1/V * np.ones((self.k, V)) + np.random.normal(loc=0, scale=0.1/V, size=(self.k, V)) #TO BE COMPLETED
			params['beta'] /= params['beta'].sum(axis=1, keepdims=True)

		#add variational parameters
		params['phi'] = [(1/self.k) * np.ones((doc.shape[0], self.k)) for doc in X]
		params['gamma'] = [params['alpha'] + doc.shape[0]/self.k for doc in X]
		if self.smoothing:
			params['lambda'] = np.ones((self.k,V)) + np.random.normal(loc=0, scale=0.01, size=(self.k,V)) #TO BE COMPLETED

		self.params = params

	#Computes lower bound that EM is optimizing
	#If smoothing then this is function of alpha, eta, gamma, phi, lambda
	#If not smoothing then this is function of alpha, beta, gamma, phi
	def L(self, X):
		alpha = self.params['alpha']
		if self.smoothing:
			eta = self.params['eta']
		else:
			beta = self.params['beta']
		
		l = 0
		
		#add dirichlet terms for parameters eta and lambda
		if self.smoothing:
			V = X[0].shape[1]
			lambda_ = self.params['lambda']
			l += self.k * (loggamma(V*eta) - V*loggamma(eta))
			l += (eta-1) * (np.sum(digamma(lambda_)) - V * np.sum(digamma(np.sum(lambda_, axis=1))))
			l -= np.sum(loggamma(np.sum(lambda_, axis=1)) - np.sum(loggamma(lambda_), axis=1))
			l -= np.sum((lambda_-1) * (digamma(lambda_) - digamma(lambda_.sum(axis=1, keepdims=True))))


		for x, phi, gamma in zip(X, self.params['phi'], self.params['gamma']):
			digamma_diff = digamma(gamma) - digamma(gamma.sum())

			#in smoothed case use expectation of beta under vartiational distribution in non-smoothed use beta
			if self.smoothing:
				expected_beta = digamma(lambda_) - digamma(lambda_.sum(axis=1,keepdims=True))
				l += (phi * (x.dot(expected_beta.T))).sum()
			else:
				l += (phi * (x.dot(np.log(beta.T)))).sum()

			
			#remaining terms are independent of smoothing
			l += (loggamma(alpha.sum()) - loggamma(alpha).sum() + (alpha-1).dot(digamma_diff))
			l += phi.dot(np.diag(digamma_diff)).sum()
			l += (-loggamma(gamma.sum()) + loggamma(gamma).sum() - (gamma-1).dot(digamma_diff))
			l += -(phi * np.log(phi)).sum()

		return l

	#Fit LDA to corpus X using variational EM
	#X is a list of length M, of N_d x V arrays, M is number of documents in corpus, V is size of vocabulary,
	#N_d is number of words in doc d. One hot encoding of each word.   
	def fit(self, X):
		self.initialize_params(X)

		converged = False
		eps = 0.01
		print(self.L(X))

		while not converged:
			l_old = self.L(X)
			self.e_step(X)
			print("E-Step Complete, l: {}".format(self.L(X)))
			#self.legal_params(X)
			self.m_step(X)
			print("M-Step Complete, l: {}".format(self.L(X)))
			#self.legal_params(X)
			l_new = self.L(X)
			converged = (np.abs(l_new - l_old) < eps)
			print("l:", l_new)

			if self.L(X)!=self.L(X):
				print("alpha: ", self.params['alpha'])
				print("beta: ", self.params['beta'])


			#check if L is nan
			assert self.L(X)==self.L(X)

		print("alpha: ", self.params['alpha'])
		print("beta: ", self.params['beta'])





	#Update approximate posterior params
	def e_step(self, X):
		eps = 0.001
		
		converged = False
		while not converged:
			old_phis = []
			old_gammas = []
			for d, (x, phi, gamma) in enumerate(zip(X, self.params['phi'], self.params['gamma'])):
				old_phis.append(phi.copy())
				old_gammas.append(gamma.copy())

				#update phi
				if self.smoothing:
					digamma_diff_gamma = digamma(gamma) - digamma(gamma.sum())
					digamma_diff_beta = digamma(x.dot(self.params['lambda'].T)) - digamma(self.params['lambda'].sum(axis=1))
					self.params['phi'][d] = np.exp(digamma_diff_beta + digamma_diff_gamma)
				else:
					beta = self.params['beta']
					digamma_diff = digamma(gamma) - digamma(gamma.sum())
					diagonal = np.diag(np.exp(digamma_diff))
					self.params['phi'][d] = x.dot(beta.T.dot(diagonal))
				#Normalize multinomial rows
				self.params['phi'][d] /= self.params['phi'][d].sum(axis=1, keepdims=True)

				#update gamma
				self.params['gamma'][d] = self.params['alpha'] + np.sum(self.params['phi'][d], axis=0)

					
			#update lambda
			if self.smoothing:
				old_lambda = self.params['lambda'].copy()
				self.params['lambda'] = self.params['eta']
				for x, phi in zip(X,self.params['phi'][d]):
					self.params['lambda'] += phi.T.dot(x)

			#check convergence of variational parameters
			change_phi_norms = [np.sqrt(np.sum((phi-old_phi)**2)) for phi,old_phi in zip(self.params['phi'],old_phis)]
			change_gamma_norms = [np.sqrt(np.sum((gamma-old_gamma)**2)) for gamma,old_gamma in zip(self.params['gamma'],old_gammas)]
			if self.smoothing:
				change_lambda_norm = np.sqrt(np.sum((self.params['lambda'] - old_lambda)**2))
			else:
				change_lambda_norm = 0

			converged = (np.max([*change_phi_norms,*change_gamma_norms,change_lambda_norm]) < eps)


	#Update lda params alpha and beta, in smoothed case alpha and eta
	def m_step(self, X):
		M = len(X)
		#update alpha
		alpha = self.params['alpha']
		converged = False
		eps = 0.001
		while not converged:
			alpha_old = alpha.copy()
			h = -M * polygamma(1, alpha)
			z = M * polygamma(1, np.sum(alpha))
			g = M * (digamma(np.sum(alpha)) - digamma(alpha))
			for gamma in self.params['gamma']:
				g += (digamma(gamma) - digamma(np.sum(gamma)))

	
			c = np.sum(g/h) / (z**(-1.0) + np.sum(h**(-1.0)))

			alpha = alpha - ((g - c) / h)
			#check if any alpha component is non-positive after the update
			mask = (alpha <= 0)
			alpha[mask] = 0.00001

			assert (alpha > 0).all()

			# if t%10 == 0:
			# 	step_size *= 0.5
			# M = len(X)
			# grad = M * (digamma(alpha.sum()) - digamma(alpha))
			# for d in range(M):
			# 	gamma = self.params['gamma'][d]
			# 	grad += (digamma(gamma) - digamma(gamma.sum()))

			
			# alpha += step_size * grad
			# print(alpha)

			change_alpha = alpha - alpha_old
			converged = (np.sqrt(np.sum(change_alpha*change_alpha)) < eps)

		self.params['alpha'] = alpha
		

		if self.smoothing:
			#update eta using Newtons method
			eta = self.params['eta']
			converged = False
			eps = 0.0001
			while not converged:
				l_old = self.L(X)
				eta_old = eta
				lambda_ = self.params['lambda']
				V = X[0].shape[1]
				L_1 = self.k * V * (digamma(V*eta) - digamma(eta))
				L_1 += np.sum(digamma(lambda_)) - V * np.sum(digamma(np.sum(lambda_, axis=1)))
				L_2 = self.k * V * (V * polygamma(1, V*eta) - polygamma(1, eta))
				eta -= L_1/L_2
				self.params['eta'] = eta
				l_new = self.L(X)
				change_eta = np.abs(eta - eta_old)
				converged = (change_eta < eps)
			self.params['eta'] = eta

		else:
			#update beta
			beta = np.zeros_like(self.params['beta'])
			for phi, x in zip(self.params['phi'], X):
				beta += phi.T.dot(x)
			#smooth beta so non-zero
			beta += 0.0000001
			beta /= beta.sum(axis=1, keepdims=True)
			self.params['beta'] = beta

# test_LDA = LatentDirichletAllocation(n_components=2, smoothing=False)

# doc1 = np.array([
# 	[1,0,0,0,0,0],
# 	[0,1,0,0,0,0],
# 	[0,0,1,0,0,0],
# 	[1,0,0,0,0,0],
# 	[0,1,0,0,0,0],
# 	[0,0,1,0,0,0],
# 	[1,0,0,0,0,0],
# 	[0,1,0,0,0,0],
# 	[0,0,1,0,0,0],
# 		])

# doc2 = np.array([
# 	[0,0,0,1,0,0],
# 	[0,0,0,0,1,0],
# 	[0,0,0,0,0,1],
# 	[0,0,0,1,0,0],
# 	[0,0,0,0,1,0],
# 	[0,0,0,0,0,1],
# 	[0,0,0,1,0,0],
# 	[0,0,0,0,1,0],
# 	[0,0,0,0,0,1],
# 		])

# X = [doc1, doc2]
# test_LDA.fit(X)

