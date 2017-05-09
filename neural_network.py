# -*- coding:utf-8 -*-#

#from prettytensor.pretty_tensor_class import PAD_VALID, PAD_SAME
#from prettytensor.pretty_tensor_methods import reshape
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from deconv import deconv2d
from prettytensor import wrap
import prettytensor as pt
class Encoder(object):
	"""
	params；
		input:
		dim_z
		dim_hidden
	"""
	def __init__(self, batch_size, x, hidden_dim, hidden_layer_dim):
		#self.x = tf.reshape([FLAGS.batch_size, 28, 28, 1])
		'''net = slim.flatten(x)
		for dim in hidden_layer_dim:
			net = slim.fully_connected(net, dim)

		#params of gaussian posteror distribution ----->q(z|x)
		#self.mu = slim.fully_connected(net, dim_z)
		#self.sigma = slim.fully_connected(net, dim_z)
		self.z = slim.fully_connected(net,hidden_dim)'''
		self.z = (wrap(x).
            reshape([batch_size, 28, 28, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            dropout(0.9).
            flatten().
            fully_connected(hidden_dim, activation_fn=None)).tensor

class Decoder(object):
	"""
	params:
		input:sampled from q(z|x)
	"""
	def __init__(self, batch_size, z_sample, hidden_dim, dim_x, hidden_layer_dim):

		epsilon = tf.random_normal([batch_size, hidden_dim//2])
		if z_sample is None:
			input_sample = epsilon
		else:
			#mu = z[:50]
			#sigma = z[50:]
			#mu = z[:, :dim_z//2]
			#sigma = tf.sqrt(tf.exp(z[:, dim_z//2:]))
			#input_sample = mu + sigma * z
			input_sample = z_sample
		'''net = input_sample
		for dim in hidden_layer_dim:
			net = slim.fully_connected(net, dim)
		logits = slim.fully_connected(net, dim_x)
		logits = tf.nn.softmax(logits)
		self.logits = logits
		self.x_hat = logits
		#self.x_hat = tf.reshape(logits, [-1, 28, 28, 1])'''
		#print (wrap(input_sample)).type

		'''self.x_hat = reshape((wrap(input_sample), [batch_size, 1, 1, hidden_dim//2]).
			deconv2d(5, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()).tensor
            '''
		print input_sample.shape 
		self.x_hat = (pt.wrap(tf.transpose(input_sample)).
            reshape([batch_size, 1, 1, hidden_dim//2]).
            deconv2d(3, 128, edges='VALID').
            deconv2d(5, 64, edges='VALID').
            deconv2d(5, 32, stride=2).
            deconv2d(5, 1, stride=2, activation_fn=None).
            flatten()).tensor
