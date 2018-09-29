#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: Network modules, including generator
#							and discrinimator
#	Last modified: July 19, 2018
#####################################################

import tensorflow as tf
from ops import *

def discriminator(image, dim=64, reuse=False, name="discriminator"):
	batch_size = image.get_shape().as_list()[0]
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		h0 = lrelu(conv2d(image, dim, name='d_h0_conv'))
		h1 = lrelu(conv2d(h0, dim*2, name='d_h1_conv'))
		h2 = lrelu(conv2d(h1, dim*4, name='d_h2_conv'))
		h3 = lrelu(conv2d(h2, dim*8, name='d_h3_conv'))
		h4 = conv2d(h3, dim*8, name='d_h3_pred')
		h = tf.reshape(h4, [batch_size, -1])
		h5 = lrelu(linear(h, dim*8, 'd_h5_lin'))
		h6 = lrelu(linear(h5, dim*16, 'd_h6_lin'))
		h7 = lrelu(linear(h6, dim*16, 'd_h7_lin'))
		return h7

def generator(image, dim=64, name='generator', is_training=True, reuse=True):

	output_c_dim = image.get_shape().as_list()[-1]
	dropout_rate = 0.5 if is_training else 1.0
	with tf.variable_scope(name) as scope:
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		e1 = conv2d(image, dim*2, d_w=1, d_h=1, name='g_e1_conv')
		e2 = instance_norm(conv2d(lrelu(e1), dim*2, 
			d_w=1, d_h=1, name='g_e2_conv'), name='g_bn_e2')
		e3 = instance_norm(conv2d(lrelu(e2), dim*4, name='g_e3_conv'), name='g_bn_e3')
		e4 = instance_norm(conv2d(lrelu(e3), dim*4, name='g_e4_conv'), name='g_bn_e4')
		e5 = instance_norm(conv2d(lrelu(e4), dim*4, name='g_e5_conv'), name='g_bn_e5')
		e6 = instance_norm(conv2d(lrelu(e5), dim*8, name='g_e6_conv'), name='g_bn_e6')
		e7 = instance_norm(conv2d(lrelu(e6), dim*8, 
			d_w=1, d_h=1, name='g_e7_conv'), name='g_bn_e7')
		e8 = instance_norm(conv2d(lrelu(e7), dim*8, 
			d_w=1, d_h=1, name='g_e8_conv'), name='g_bn_e8')

		d1 = deconv2d(lrelu(e8),dim*8, d_w=1, d_h=1, name='g_d1')
		d1 = dropout(instance_norm(d1, name='g_bn_d1'), dropout_rate)
		d1 = tf.concat([d1, e7], 3)

		d2 = deconv2d(lrelu(d1),
			dim*8, d_w=1, d_h=1, name='g_d2')
		d2 = dropout(instance_norm(d2, name='g_bn_d2'), dropout_rate)
		d2 = tf.concat([d2, e6], 3)

		d3 = deconv2d(lrelu(d2),
			dim*4, name='g_d3')
		d3 = dropout(instance_norm(d3, name='g_bn_d3'), dropout_rate)
		d3 = tf.concat([d3, e5], 3)

		d4 = deconv2d(lrelu(d3),
			dim*4, name='g_d4')
		d4 = instance_norm(d4, name='g_bn_d4')
		d4 = tf.concat([d4, e4], 3)

		d5 = deconv2d(lrelu(d4),
			dim*4, name='g_d5')
		d5 = instance_norm(d5, name='g_bn_d5')
		d5 = tf.concat([d5, e3], 3)

		d6 = deconv2d(lrelu(d5),
			dim*2, name='g_d6')
		d6 = instance_norm(d6, name='g_bn_d6')
		d6 = tf.concat([d6, e2], 3)

		d7 = deconv2d(lrelu(d6),
			dim*2, d_w=1, d_h=1, name='g_d7')
		d7 = instance_norm(d7, name='g_bn_d7')
		d7 = tf.concat([d7, e1], 3)

		d8 = deconv2d(lrelu(d7),
			output_c_dim, d_w=1, d_h=1, name='g_d8')

		#w = tf.get_variable("g_w", [1], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		#b = tf.get_variable("g_b", [1], initializer=tf.random_normal_initializer(0.0, 0.02, dtype=tf.float32))

		#return tf.clip_by_value(w*tanh(d8)+b, -1, 1)
		return tanh(d8)
