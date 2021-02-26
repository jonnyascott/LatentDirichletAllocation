import matplotlib.pyplot as plt
import numpy as np
import os 
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from helper_functions import positive_constraint, remove_empty_vocab, print_topic

#Variational posterior q_(phi)(z|x), approximates categorical true posterior p(z|x) with factorized Gumbel-Softmax.
#Encoder computes the parameters phi, then samples from the variational distribution, then computes likelihood of samples.
class Encoder(keras.Model):
    def __init__(self, output_dim, num_hidden, num_hidden_units, temp, **kwargs):
        super().__init__(**kwargs)
        self.hidden = []
        for _ in range(num_hidden):
            self.hidden.append(keras.layers.Dense(num_hidden_units, activation='relu'))
        self.out = keras.layers.Dense(output_dim, activation='softmax')
        self.temp = temp
        self.history = []

    #Takes 2d input of x represented as a word frequency vector, shape: MxV
    def call(self, inputs):
        #Compute parameters phi
        x = tf.cast(inputs, tf.float32)
        N = tf.reduce_max(tf.reduce_sum(x, axis=1)) #max length of any doc in minibatch
        doc_lengths = tf.reduce_sum(x, axis=1, keepdims=True)
        for layer in self.hidden:
            x = layer(x)
        phi = self.out(x) #shape M,k
        #Take samples using phi
        gumbel_softmax = tfd.RelaxedOneHotCategorical(self.temp, probs=phi) 
        z = tfd.Sample(gumbel_softmax, sample_shape=int(N)).sample() #phi is (M, k) so z is (M,N,k)
        #Compute mask to zero out extra samples for shorter documents
        y = np.arange(int(N))
        mask = tf.cast(doc_lengths>y, dtype=tf.float32) #shape M,N
        #Compute log probability of samples
        log_var_prob = tf.reduce_sum(tf.transpose(gumbel_softmax.log_prob(tf.transpose(z, [1,0,2]))) * mask) #has shape M,N
        return [phi, z * tf.expand_dims(mask, axis=-1), log_var_prob]

#Computes the true joint probability p(x,z)
class Decoder(keras.Model):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.model_parameters = {}
        self.history = ([],[])
        self.vocab_size = vocab_size

    def build(self, batch_input_shape):
        _, z_shape = batch_input_shape
        num_topics = z_shape[-1]
        self.alpha = self.add_weight(
            name="alpha", shape=[num_topics], 
            initializer=keras.initializers.RandomNormal(0.1, 0.001),
            constraint=positive_constraint)
        self.beta = self.add_weight(
            name="beta", shape=[num_topics, self.vocab_size],
            initializer='ones')
        self.model_parameters['alpha'] = self.alpha
        self.model_parameters['beta'] = self.beta
        super().build(batch_input_shape)

    #Takes input [x,z] where:
    #x is ordered word representation of corpus, shape MxN
    #z is 3d gumbel-softmax samples, shape MxNxK. (N is max length of document in current batch)
    def call(self, inputs):
        x, z = inputs
        M, N, K = z.shape
        x = tf.cast(x, tf.int32)
        log_beta = tf.math.lbeta
        #Compute log p(z|alpha)
        count_each_topic = tf.reduce_sum(z, axis=1)
        log_latent_prob = tf.reduce_sum(log_beta(self.alpha + count_each_topic)) - M * log_beta(self.alpha)
        self.count_topic_history = np.array([count_each_topic])
        #compute log p(w|z,beta)
        multinomial_probs = tf.gather_nd(z@tf.nn.softmax(self.beta), x[:,:,tf.newaxis]-1, batch_dims=2) #word indexes start from 1
        log_likelihood = tf.reduce_sum(tf.math.log(tf.where(multinomial_probs==0., 1., multinomial_probs)))
        return log_latent_prob, log_likelihood

#Autoencoding VB algorithm, optimizes the elbo jointly wrt both encoder and decoder parameters. 
class AEVB(keras.Model):
    def __init__(self, output_dim, num_hidden, num_hidden_units, temp, vocab_size, corpus_size, kl_scale, **kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(output_dim, num_hidden, num_hidden_units, temp)
        self.decoder = Decoder(vocab_size)
        self.model_parameters = self.decoder.model_parameters
        self.corpus_size = corpus_size
        self.kl_scale = kl_scale

    #Takes as inputs two representations of x:
    #1. Word frequencies, shape MxV, used by encoder
    #2. Ordered documents, shape MxN_max, used by decoder, N_max is max length of any document
    #Ordered documents has a row per document, one int per word in each document, padded with zeros to reach N_max for short docs
    def call(self, inputs):
        word_freqs, docs = inputs
        _, z, log_var_prob = self.encoder(word_freqs)
        M, N, _ = z.shape
        log_latent_prob, log_likelihood = self.decoder([docs[:,:N],z])
        elbo_estimate = (self.corpus_size / M) * (log_likelihood + self.kl_scale * (log_latent_prob - log_var_prob))
        self.encoder.history.append(log_var_prob)
        self.decoder.history[0].append(log_latent_prob)
        self.decoder.history[1].append(log_likelihood)
        return -elbo_estimate #to maximize L we minimize -L

    def fit_aevb(self, x, optimizer, n_epochs, batch_size):
        #temp settings for Gumbel-Softmax
        scale = 0.001
        min_temp = 0.5
        initial_temp = self.encoder.temp
        n_steps = self.corpus_size // batch_size
        idxs = np.arange(self.corpus_size)
        for epoch in range(1, n_epochs+1):
            np.random.shuffle(idxs)
            for step in range(n_steps):
                total_step_number = -scale * ((epoch - 1) * n_steps + step)
                self.encoder.temp = np.max(np.array([min_temp, initial_temp*np.exp(total_step_number)], dtype=np.float32))
                batch_idxs = idxs[step * batch_size : (step + 1) * batch_size]
                x_batch = (x[0][batch_idxs], x[1][batch_idxs])
                with tf.GradientTape() as tape:
                    loss=self.call(x_batch)
                gradients = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                for variable in self.trainable_variables:
                    if variable.constraint:
                        variable.assign(variable.constraint(variable))
                print(f"Step: {step} / {n_steps}. Loss: {np.array(loss):.4f}.")
            print(f"epoch: {epoch}. CurrentTemp: {self.encoder.temp:.2f}. Loss: {np.array(loss):.4f}.")
        return self.encoder.history, self.decoder.history


if __name__ == "__main__":
    np.random.seed(4)

    word_freqs = np.load("data/corpus_train_word_freqs_small.npy")
    docs = np.load("data/corpus_train_docs_small.npy")
    #selected = np.hstack((np.arange(25), np.arange(1000, 1025)))
    word_freqs_short, docs_short, word_lookup_map_inv = remove_empty_vocab(word_freqs, docs)
    x = (word_freqs_short, docs_short)
    # np.save("data/corpus_train_word_freqs_small", word_freqs[selected])
    # np.save("data/corpus_train_docs_small", docs[selected])
    corpus_size, vocab_size = x[0].shape
    K = 10
    
    train_model = True
    if train_model:
        output_dim = K #num_topics
        num_hidden = 2 #hidden layers
        num_hidden_units = 100 #unit per hidden layer
        initial_temp = 3
        kl_scale = 0.01

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.9)

        test_aevb = AEVB(output_dim, num_hidden, num_hidden_units, initial_temp, vocab_size, corpus_size, kl_scale)
        encoder_history, decoder_history = test_aevb.fit_aevb(x, optimizer, n_epochs=500, batch_size=50)

    show_plots = True
    if train_model and show_plots:
        num_steps = len(encoder_history)
        xs = np.arange(num_steps)
        plt.plot(xs, kl_scale*np.array(encoder_history), label='log_var_probs')
        plt.plot(xs, kl_scale*np.array(decoder_history[0]), label='log_latent_probs')
        plt.plot(xs, np.array(decoder_history[1]), label='log_likelihoods')
        plt.plot(xs, kl_scale*np.array(decoder_history[0]) + np.array(decoder_history[1]) - kl_scale*np.array(encoder_history), label='ELBO')
        plt.legend()
        plt.show()

    alpha_file = "alpha_shortest"
    beta_file = "beta_shortest"
    model_weights_file = "model_weights_shortest"

    save_parameters = True
    if train_model and save_parameters:
        alpha = np.array(test_aevb.model_parameters['alpha'])
        beta = np.array(test_aevb.model_parameters['beta'])
        np.save(alpha_file, alpha)
        np.save(beta_file, beta)
        test_aevb.save_weights(model_weights_file)

    display_topics = True
    if display_topics:
        beta = np.load(beta_file + ".npy")
        num_topics = K
        num_words_per_topic = 10
        word_idxs = np.argpartition(beta, -num_words_per_topic)[:,-num_words_per_topic:]
        for k in range(num_topics):
            for j in range(num_words_per_topic):
                word_idxs[k,j] = word_lookup_map_inv[word_idxs[k,j]+1]
        
        for k in range(num_topics):
            print_topic(word_idxs[k])