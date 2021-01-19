import numpy as np
import math

from scipy.special import gamma as gamma_func
from scipy.special import loggamma, digamma, polygamma

from sklearn.decomposition import LatentDirichletAllocation as LDA_correct

class LatentDirichletAllocation:

	def __init__(self, n_components=10, smoothing=False):
		self.k = n_components
		self.smoothing = smoothing
		

	#Initialize parameters of lda and of approximate posterior distributions
	def initialize_params(self, X):
		M, V = X.shape
		N = np.sum(X, axis=1)
		params = {}

		#add model parameters
		params['alpha'] = np.ones(self.k) + np.random.normal(loc=0, scale=0.01, size=self.k) #TO BE COMPLETED
		if self.smoothing:
			params['eta'] = 0.01 #TO BE COMPLETED
		else:
			params['beta'] = 1/V * np.ones((self.k, V)) + np.random.normal(loc=0, scale=0.1/V, size=(self.k, V)) #TO BE COMPLETED
			params['beta'] /= np.sum(params['beta'], axis=1)[:, np.newaxis]

		#add variational parameters
		params['phi'] = [(1/self.k) * np.ones((N[d], self.k)) for d in range(M)]
		params['gamma'] = [params['alpha'] + N[d]/self.k for d in range(M)]
		if self.smoothing:
			params['lambda'] = np.ones((self.k,V)) + np.random.normal(loc=0, scale=0.01, size=(self.k,V)) #TO BE COMPLETED

		self.params = params

	#Computes lower bound that EM is optimizing
	#If smoothing then this is function of alpha, eta, gamma, phi, lambda
	#If not smoothing then this is function of alpha, beta, gamma, phi
	def L(self, X):
		V,M = X.shape
		alpha = self.params['alpha']
		if self.smoothing:
			eta = self.params['eta']
		else:
			beta = self.params['beta']
		
		l = 0
		
		#add dirichlet terms for parameters eta and lambda
		if self.smoothing:
			
			lambda_ = self.params['lambda']
			l += self.k * (loggamma(V*eta) - V*loggamma(eta))
			l += (eta-1) * (np.sum(digamma(lambda_)) - V * np.sum(digamma(np.sum(lambda_, axis=1))))
			l -= np.sum(loggamma(np.sum(lambda_, axis=1)) - np.sum(loggamma(lambda_), axis=1))
			l -= np.sum((lambda_-1) * (digamma(lambda_) - digamma(np.sum(lambda_, axis=1))[:,np.newaxis]))


		for d in range(M):
			phi = self.params['phi'][d]
			gamma = self.params['gamma'][d]
			
			digamma_diff = digamma(gamma) - digamma(gamma.sum())

			#in smoothed case use expectation of beta under vartiational distribution in non-smoothed use beta
			if self.smoothing:
				expected_beta = digamma(lambda_) - digamma(np.sum(lambda_, axis=1))[:,np.newaxis]
				l += (phi * (X[d].dot(expected_beta.T))).sum()
			else:
				l += (phi * (X[d].dot(np.log(beta.T)))).sum()

			
			#remaining terms are independent of smoothing
			l += (loggamma(alpha.sum()) - loggamma(alpha).sum() + (alpha-1).dot(digamma_diff))
			l += phi.dot(np.diag(digamma_diff)).sum()
			l += (-loggamma(gamma.sum()) + loggamma(gamma).sum() - (gamma-1).dot(digamma_diff))
			l += -(phi * np.log(phi)).sum()

		return l

	#Fit LDA to corpus X using variational EM
	#X is a bag of words representation of shape MxV where M is number of documents, V is vocabulary size
	def fit(self, X):
		self.initialize_params(X)

		converged = False
		eps = 0.00001

		while not converged:
			l_old = self.L(X)
			self.e_step(X)
			for d in range(len(X)):
				print("gamma " + str(d) + ": ", self.params['gamma'][d])
				print("phi " + str(d) + ": ", self.params['phi'][d])
			print("lambda:", self.params["lambda"])
			self.m_step(X)
			l_new = self.L(X)
			converged = (np.abs(l_new - l_old) < eps)
			print("l_old:", l_old)
			print("l_new:", l_new)

			assert not self.L(X)!=self.L(X)




	#Update approximate posterior params
	def e_step(self, X):
		eps = 0.0001
		
		converged = False
		while not converged:
			old_phi = {}
			old_gamma = {}
			for d in range(len(X)):
				old_phi[d] = self.params['phi'][d].copy()
				old_gamma[d] = self.params['gamma'][d].copy()

				#update phi
				if self.smoothing:
					digamma_diff_gamma = digamma(self.params['gamma'][d]) - digamma(self.params['gamma'][d].sum())
					digamma_diff_beta = digamma(X[d].dot(self.params['lambda'].T)) - digamma(self.params['lambda'].sum(axis=1))
					self.params['phi'][d] = np.exp(digamma_diff_beta + digamma_diff_gamma)
				else:
					beta = self.params['beta']
					digamma_diff = digamma(self.params['gamma'][d]) - digamma(self.params['gamma'][d].sum())
					diagonal = np.diag(np.exp(digamma_diff))
					self.params['phi'][d] = X[d].dot(beta.T.dot(diagonal))
				#Normalize multinomial rows
				row_sums = np.sum(self.params['phi'][d], axis = 1)[:, np.newaxis]
				self.params['phi'][d] /= row_sums

				#update gamma
				self.params['gamma'][d] = self.params['alpha'] + np.sum(self.params['phi'][d], axis=0)

					
			#update lambda
			if self.smoothing:
				old_lambda = self.params['lambda'].copy()
				self.params['lambda'] = self.params['eta']
				for d in range(len(X)):
					self.params['lambda'] += self.params['phi'][d].T.dot(X[d])

			#check convergence of variational parameters
			change_phi_norms = [np.sqrt(np.sum((self.params['phi'][d] - old_phi[d])**2)) for d in range(len(X))]
			change_gamma_norms = [np.sqrt(np.sum((self.params['gamma'][d] - old_gamma[d])**2)) for d in range(len(X))]
			if self.smoothing:
				change_lambda_norm = np.sqrt(np.sum((self.params['lambda'] - old_lambda)**2))
			else:
				change_lambda_norm = 0

			converged = (np.max([*change_phi_norms,*change_gamma_norms,change_lambda_norm]) < eps)


	#Update lda params alpha and beta
	def m_step(self, X):
		#update alpha
		alpha = self.params['alpha']
		converged = False
		eps = 0.0001
		while not converged:
			alpha_old = alpha.copy()
			h = -len(X) * polygamma(1, alpha)
			z = len(X) * polygamma(1, np.sum(alpha))
			g = len(X) * (digamma(np.sum(alpha)) - digamma(alpha))
			for d in range(len(X)):
				gamma = self.params['gamma'][d]
				g += (digamma(gamma) - digamma(np.sum(gamma)))

	
			c = np.sum(g/h) / (1/z + np.sum(1/h))

			alpha -= (g - c) / h

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
				print(l_old)
				print(l_new)
				print("eta:", eta)
				change_eta = np.abs(eta - eta_old)
				converged = (change_eta < eps)
			self.params['eta'] = eta

		else:
			#update beta
			beta = np.zeros_like(self.params['beta'])
			for d in range(len(X)):
				phi = self.params['phi'][d]
				beta += phi.T.dot(X[d])
			
			beta /= np.sum(beta, axis=1)[:, np.newaxis]
			self.params['beta'] = beta

		print("alpha:", self.params['alpha'])
		if self.smoothing:
			print("eta:", self.params['eta'])
		else:
			print("beta:", self.params['beta'])



test = LatentDirichletAllocation(n_components=2, smoothing=True)

# doc1 = np.array([
# 	[1,0,0,0,0,0,0,0,0,0],
# 	[0,1,0,0,0,0,0,0,0,0],
# 	[0,0,1,0,0,0,0,0,0,0],
# 		])

# doc2 = np.array([
# 	[0,0,0,1,0,0,0,0,0,0],
# 	[0,0,0,0,1,0,0,0,0,0],
# 	[0,0,0,0,0,1,0,0,0,0],
# 	[0,0,0,0,0,0,1,0,0,0],
# 	[0,0,0,0,0,0,0,1,0,0],
# 	[0,0,0,0,0,0,0,0,1,0],
# 	[0,0,0,0,0,0,0,0,0,1],
# 		])

doc1 = np.array([
	[1,0,0,0,0,0],
	[0,1,0,0,0,0],
	[0,0,1,0,0,0],
	[1,0,0,0,0,0],
	[0,1,0,0,0,0],
	[0,0,1,0,0,0],
	[1,0,0,0,0,0],
	[0,1,0,0,0,0],
	[0,0,1,0,0,0],
		])

doc2 = np.array([
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1],
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1],
	[0,0,0,1,0,0],
	[0,0,0,0,1,0],
	[0,0,0,0,0,1],
		])


# doc3 = np.array([
# 	[0,0,0,1,0,0,0,0,0,0],
# 	[0,0,0,0,1,0,0,0,0,0],
# 	[0,0,0,0,0,0,1,0,0,0],
# 	[0,0,0,0,0,1,0,0,0,0],
# 	[0,0,0,0,0,0,0,1,0,0],
# 	[0,0,0,0,0,0,0,0,1,0],
# 	[0,0,0,0,0,0,0,0,0,1],
# 		])
# doc4 = np.array([
# 	[0,0,0,0,0,0,1,0,0,0],
# 	[0,0,0,0,0,1,0,0,0,0],
# 	[0,0,0,0,0,0,0,1,0,0],
# 	[0,0,0,0,0,0,0,0,1,0],
# 	[0,0,0,0,0,0,0,0,0,1],
# 		])
# doc5 = np.array([
# 	[1,0,0,0,0,0,0,0,0,0],
# 	[0,1,0,0,0,0,0,0,0,0],
# 	[0,0,1,0,0,0,0,0,0,0],
# 		])


X = [doc1, doc2]
# Y = np.array([
# 	[1,1,1,0,0,0],
# 	[0,0,0,1,1,1]
# 	])

# test_correct = LDA_correct(n_components = 2, topic_word_prior=5)

#print(test_correct.topic_word_prior_)
# test_correct.fit(Y)
# print(test_correct.topic_word_prior_)
test.fit(X)
#test.initialize_params(X)
#print(test.L(X))