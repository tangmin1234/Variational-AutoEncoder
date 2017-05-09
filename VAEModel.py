# -*- coding:utf-8 -*-
"""
reproduction of Auto-Encoding Variational Bayes
paper:
author:
inspired by https://github.com/saemundsson/semisupervised_vae
"""
import numpy as np
import tensorflow as tf
#import prettytensor as pt

from neural_network import Encoder, Decoder

class VAEModel(object):
	"""
	params:
	dim_z:
	dim_x:
	num_lay_zx:
	num_lay_xz:
	"""
	def __init__(self, FLAGS, sess, dim_x,
					num_lay_zx = [100, 100], num_lay_xz = [100, 100]):

		self.FLAGS = FLAGS
		self.sess = sess
		
		self.dim_x = dim_x
		self.num_lay_zx, self.num_lay_xz = num_lay_zx, num_lay_xz

		self.x = tf.placeholder(tf.float32, [self.FLAGS.batch_size, self.dim_x])

		#self.encoder = Encoder(self.x, self.FLAGS.hidden_dim, self.num_lay_zx)
		
		#inference model
		self.encoder = Encoder(self.FLAGS.batch_size, self.x, self.FLAGS.hidden_dim, self.num_lay_zx)
		self.objective()
		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.saver()
		self	
	def draw_sample(self, mu, sigma):
		epsilon = tf.random_normal([self.FLAGS.batch_size, self.FLAGS.hidden_dim//2])
		#epsilon = tf.random_normal(shape = (tf.shape(mu)), mean = mu, stddev = sigma)
		sample = mu + epsilon * sigma 
		return sample

	def generate_z_x(self):

		print tf.shape(self.encoder.z)
		mu = self.encoder.z[:, :self.FLAGS.hidden_dim//2]
		sigma = tf.sqrt(tf.exp(self.encoder.z[:, self.FLAGS.hidden_dim//2:]))
		#mu = self.encoder.z[0:50]
		#sigma = self.encoder.z[50:100]
		z_sample = self.draw_sample(mu, sigma)
		return mu, sigma, z_sample
	
	def generate_x_z(self, z_sample):

		self.decoder = Decoder(self.FLAGS.batch_size, z_sample, self.FLAGS.hidden_dim, self.dim_x, self.num_lay_xz)
		x_hat = self.decoder.x_hat
		return x_hat

	def get_vae_loss(self, mu, sigma, epsilon = 1e-8):
		#return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) - 2.0 * tf.log(stddev + epsilon) - 1.0))
		vae_loss = tf.reduce_sum(0.5 * (tf.square(mu) + tf.square(sigma) - 2.0 * tf.log(sigma + epsilon) - 1.0))
		return vae_loss
	
	def get_reconstrct_loss(self, x, x_hat, epsilon = 1e-8):

		x_hat = tf.reshape(x_hat, (tf.shape(x)))
		reconstruct_loss = - tf.reduce_sum(x * tf.log(x_hat + epsilon) -
                         (1.0 - x) * tf.log(1.0 - x_hat + epsilon))
		return reconstruct_loss

	def objective(self):
		mu, sigma, z_sample = self.generate_z_x()

		self.mu = mu

		x_hat = self.generate_x_z(z_sample)
		self.x_hat = x_hat

		#cost
		self.loss = 0.0

		vae_loss = self.get_vae_loss(mu, sigma)
		
		reconstruct_loss = self.get_reconstrct_loss(self.x, x_hat)
		
		self.vae_loss = vae_loss
		self.reconstruct_loss = reconstruct_loss
		self.loss = reconstruct_loss + vae_loss
		#self.loss = - self.loss
		#self.logits = logits

		#evaluation
		#_, _, z_sample_eval = self.generate_z_x(self.x)
		self.x_hat_eval = self.generate_x_z(None)
		#log_likelihood_eval = Decoder(FLAGS.batch_size, z_sample, FLAGS.hidden_dim, self.dim_x, self.num_lay_xz).logits

		#self.log_likelihood_eval = log_likelihood_eval

	def train(self, dataset, max_epoch = 100):

		#num_sample = 
		#epoch = num_sample // self.FLAGS.batch_size
		train_epoch = max_epoch
		#valid_epoch = int(epoch * 0.2)

		seed = max_epoch
		np.random.seed(seed)
		tf.set_random_seed(seed)


		self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate, epsilon = 1.0)
		self.train_op = self.optimizer.minimize(self.loss)
		
		#log_lik_hood = - np.inf

		#with Session.as_default() as sess:
		self.sess.run(self.init_op)

		for i in xrange(train_epoch):
			train_loss = 0.0
			for k in xrange(self.FLAGS.updates_per_epoch):
				x_batch, _ = dataset.train.next_batch(self.FLAGS.batch_size)
				feed = {self.x: x_batch}
				_, loss_value, mu, reconstruct_loss = self.sess.run([self.train_op, self.loss, self.mu, self.reconstruct_loss],feed_dict=feed)
				#print x_hat
				#print np.shape(x_hat), x_hat[0]
				#print reconstruct_loss,loss_value
				train_loss += loss_value
				#print mu
			train_loss /= self.FLAGS.updates_per_epoch
			print "%dth Iteration" % i
			print "train_loss:%f" % train_loss
			#print mu
			#print "logits:"
			#print logits
		#save model
		self.saver.save(self.sess, './my_model')

	def test(self):
		#read model parameter
		self.sess.run(self.init_op)
		self.saver.restore(self.sess, './my_model')

		for i in xrange(train_epoch):
			imgs = sess.run(x_hat_eval)
			for k in range(self.FLAGS.batch_size):
				imgs_folder = os.path.join(self.LAGS.working_directory, 'imgs')
				if not os.path.exists(imgs_folder):
					os.makedirs(imgs_folder)

				imsave(os.path.join(imgs_folder, '%d.png') % k, imgs[k].reshape(28, 28))
