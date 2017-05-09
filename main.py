# -*- coding:utf-8 -*-
import os
import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from VAEModel import VAEModel

flags = tf.flags
flags.DEFINE_string("working_dir", "", "working directory")
flags.DEFINE_integer("batch_size", 128, "batch_size")
flags.DEFINE_integer("hidden_dim", 20, "hidden_dim")
flags.DEFINE_boolean("is_test", False, "running mode")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_integer("updates_per_epoch", 100, "number of update each epoch")
FLAGS = flags.FLAGS

def main(_):

	data_dir = os.path.join(FLAGS.working_dir, "mnist")
	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	import ipdb
	ipdb.set_trace()
	#get data
	dataset = input_data.read_data_sets(data_dir, one_hot=True)
	x_dim = 28 * 28
	#print x_dim
	#print dataset
	##train

	with tf.Session() as sess:
		model = VAEModel(FLAGS, sess, x_dim)

		if FLAGS.is_test:
			model.test()
		else:
			model.train(dataset)

if __name__ == '__main__':
	tf.app.run()