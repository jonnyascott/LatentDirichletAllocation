import numpy as np
import math

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

		if (self.params['beta'] < 0).any() or (np.abs(self.params['beta'].sum(axis=1) - 1 ) > 0.001).any():
			legal = False
			print("Beta illegal: ", self.params['beta'])
			if (self.params['beta'] < 0).any():
				print("Beta negative")
			if (np.abs(self.params['beta'].sum(axis=1) - 1 ) > 0.001).any():
				print("Beta not summing to 1: ", self.params['beta'].sum(axis=1))


		for d in range(len(X)):
			if (self.params['gamma'][d] <= 0).any():
				legal = False
				print("Gamma {} illegal: ".format(d), self.params['gamma'][d])

			if (self.params['phi'][d] < 0).any() or (np.abs(self.params['phi'][d].sum(axis=1) - 1) > 0.001).any():
				legal = False
				print("Phi {} illegal: ".format(d))
				if (self.params['phi'][d] < 0).any():
					print("Phi {} negative".format(d))
				if (np.abs(self.params['phi'][d].sum(axis=1) - 1) > 0.001).any():
					print("Phi {} not summing to 1".format(d))
		if legal:
			print("All parameters legal.")
		

	#Initialize parameters of lda and of approximate posterior distributions
	def initialize_params(self, X):
		V = X[0].shape[1]
		params = {}

		#add model parameters
		params['alpha'] = 0.1*np.ones(self.k) + np.random.normal(loc=0, scale=0.001, size=self.k) #TO BE COMPLETED
		if self.smoothing:
			params['eta'] = 0.01 #TO BE COMPLETED
		else:
			params['beta'] = 1/V * np.ones((self.k, V)) + np.random.normal(loc=0, scale=0.1/V, size=(self.k, V)) #TO BE COMPLETED
			params['beta'] /= np.sum(params['beta'], axis=1)[:, np.newaxis]

		#add variational parameters
		params['phi'] = [(1/self.k) * np.ones((X[d].shape[0], self.k)) for d in range(len(X))]
		params['gamma'] = [params['alpha'] + X[d].shape[0]/self.k for d in range(len(X))]
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
			l -= np.sum((lambda_-1) * (digamma(lambda_) - digamma(np.sum(lambda_, axis=1))[:,np.newaxis]))


		for d in range(len(X)):
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
	#X is a list of length M, of N_d x V arrays, M is number of documents in corpus, V is size of vocabulary,
	#N_d is number of words in doc d. One hot encoding of each word.   
	def fit(self, X, X_form="bow"):
		self.initialize_params(X)

		converged = False
		eps = 0.001
		print(self.L(X))

		while not converged:
			l_old = self.L(X)
			self.e_step(X)
			print("E-Step Complete, l: {}".format(self.L(X)))
			self.legal_params(X)
			self.m_step(X)
			print("E-Step Complete, l: {}".format(self.L(X)))
			self.legal_params(X)
			l_new = self.L(X)
			converged = (np.abs(l_new - l_old) < eps)
			print("l:", l_new)

			if self.L(X)!=self.L(X):
				print("alpha: ", self.params['alpha'])
				print("beta: ", self.params['beta'])


			#check if L is nan
			assert self.L(X)==self.L(X)




	#Update approximate posterior params
	def e_step(self, X):
		eps = 0.001
		
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


	#Update lda params alpha and beta, in smoothed case alpha and eta
	def m_step(self, X):
		#update alpha
		alpha = self.params['alpha']
		print("alpha: ", alpha)
		converged = False
		eps = 0.001
		while not converged:
			alpha_old = alpha.copy()
			h = -len(X) * polygamma(1, alpha)
			z = len(X) * polygamma(1, np.sum(alpha))
			g = len(X) * (digamma(np.sum(alpha)) - digamma(alpha))
			for d in range(len(X)):
				gamma = self.params['gamma'][d]
				print("gamma {}".format(d), gamma)
				g += (digamma(gamma) - digamma(np.sum(gamma)))

	
			c = np.sum(g/h) / (z**(-1.0) + np.sum(h**(-1.0)))

			alpha = alpha - ((g - c) / h)
			print("alpha: ", alpha)
			#check if any alpha component is non-positive after the update
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
			for d in range(len(X)):
				phi = self.params['phi'][d]
				beta += phi.T.dot(X[d])
			#smooth beta so non-zero
			beta += 0.0000001
			beta /= np.sum(beta, axis=1)[:, np.newaxis]
			self.params['beta'] = beta

test_LDA = LatentDirichletAllocation(n_components=2, smoothing=False)

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

X = [doc1, doc2]
test_LDA.fit(X)

