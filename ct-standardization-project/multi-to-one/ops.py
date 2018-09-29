#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: Atom operations and losses
#	Last modified: July 19, 2018
#####################################################

import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops


def conv_cond_concat(x, y):
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		return conv


def conv2d_n(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
		scale = tf.get_variable(name + "_scale", [1], tf.float32, initializer=tf.truncated_normal_initializer(1.0, stddev=stddev))
		norm = tf.sqrt(tf.reduce_sum(tf.square(w)))
		weights = w/norm * scale
		conv = tf.nn.conv2d(input_, weights, strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
		return conv

def deconv2d(input_, dim,
			 k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		shape = input_.get_shape().as_list()
		w = tf.get_variable('w', [k_h, k_w, dim, shape[-1]],
			initializer=tf.random_normal_initializer(stddev=stddev))

		deconv = tf.nn.conv2d_transpose(input_, w, 
			output_shape = [shape[0], shape[1]*d_h, shape[2]*d_w, dim], 
			strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [dim],
			initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv
	   

def deconv2d_n(input_, dim,
			 k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02,
			 name="deconv2d", with_w=False):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		shape = input_.get_shape().as_list()
		w = tf.get_variable('w', [k_h, k_w, dim, shape[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		scale = tf.get_variable(name + "_scale", [1], tf.float32, initializer=tf.truncated_normal_initializer(1.0, stddev=stddev))
		norm = tf.sqrt(tf.reduce_sum(tf.square(w)))
		weights = w/norm * scale

		deconv = tf.nn.conv2d_transpose(input_, weights, 
			output_shape = [shape[0], shape[1]*d_h, shape[2]*d_w, dim], 
			strides=[1, d_h, d_w, 1], padding='SAME')

		biases = tf.get_variable('biases', [dim],
			initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		if with_w:
			return deconv, w, biases
		else:
			return deconv

def linear_n(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		w = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		scale = tf.get_variable(scope + "_scale", [1], tf.float32, initializer=tf.truncated_normal_initializer(1.0, stddev=stddev))
		norm = tf.sqrt(tf.reduce_sum(tf.square(w)))
		weights = w/norm * scale
		tf.add_to_collection('weights', weights)
		if with_w:
			return tf.matmul(input_, weights) + bias, weights, bias
		else:
			return tf.matmul(input_, weights) + bias


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		tf.add_to_collection('weights', matrix)
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias

def instance_norm(input, name="instance_norm"):
	with tf.variable_scope(name):
		depth = input.get_shape()[3]
		scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
		mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
		epsilon = 1e-5
		inv = tf.rsqrt(variance + epsilon)
		normalized = (input-mean)*inv
		return scale*normalized + offset

def batch_norm(x, epsilon=1e-5, momentum=0.9, name="batch_norm"):
	return tf.contrib.layers.batch_norm(x, decay=momentum, \
		updates_collections=None, epsilon=epsilon, scale=True, scope=name)

def binary_cross_entropy(preds, targets, name=None):
	eps = 1e-12
	with ops.op_scope([preds, targets], name, "bce_loss") as name:
		preds = ops.convert_to_tensor(preds, name="preds")
		targets = ops.convert_to_tensor(targets, name="targets")
		return tf.reduce_mean(-(targets * tf.log(preds + eps) +
							  (1. - targets) * tf.log(1. - preds + eps)))

def gan_loss(A,B):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
					logits=A, labels=B))

def sigmoid(x):
	return tf.nn.sigmoid(x)

def tanh(x):
	return tf.nn.tanh(x)

def dropout(x,r):
	return tf.nn.dropout(x,r)

def mse_loss(A,B):
	return tf.losses.mean_squared_error(A,B)

def abs_loss(A,B):
	return tf.losses.absolute_difference(A,B)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.nn.leaky_relu(x, alpha=leak)

def relu(x):
	return tf.nn.relu(x)

def penalty_mse(x, b=1.0):
	return tf.reduce_mean((x-b)**2)

def penalty_abs(x,b=1.0):
	return tf.reduce_mean(relu(tf.abs(x)-b))
