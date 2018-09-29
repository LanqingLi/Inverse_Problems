#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: Network modules, including generator
#							and discrinimator
#	Last modified: Aug 17, 2018
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
		return h6


def dis_style(features, dim=64, reuse=False, name="dis_style"):
	with tf.variable_scope(name):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		
		d1 = lrelu(linear(features, 512, scope='d_linear1'))
		d2 = lrelu(linear(d1, 512, scope='d_linear2'))
		d3 = tanh(linear(d2, 64, scope='d_linear3'))
		
		return d3

def encoder_content(image, dim=8, name='generator', is_training=True, reuse=True):
	output_c_dim = image.get_shape().as_list()[-1]
	dropout_rate = 0.5 if is_training else 1.0
	with tf.variable_scope(name) as scope:
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		ce1 = instance_norm(conv2d(image, dim, name='g_ce1_conv'), name='g_ce_in1')
		ce2 = instance_norm(conv2d(lrelu(ce1), dim*2, name='g_ce2_conv'), name='g_ce_in2')
		ce3 = instance_norm(conv2d(lrelu(ce2), dim*4, name='g_ce3_conv'), name='g_ce_in3')
		ce4 = instance_norm(conv2d(lrelu(ce3), dim*8, name='g_ce4_conv'), name='g_ce_in4')

		ce5 = instance_norm(conv2d(lrelu(ce4), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_ce5_conv'), name='g_ce_in5')
		ce6 = instance_norm(conv2d(lrelu(ce5), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_ce6_conv'), name='g_ce_in6')
		ce7 = instance_norm(conv2d(lrelu(ce6), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_ce7_conv'), name='g_ce_in7')
		ce7 += ce4

		ce8 = instance_norm(conv2d(lrelu(ce7), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_ce8_conv'), name='g_ce_in8')
		ce9 = instance_norm(conv2d(lrelu(ce8), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_ce9_conv'), name='g_ce_in9')
		ce10 = instance_norm(conv2d(lrelu(ce9), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_ce10_conv'), name='g_ce_in10')
		ce10 += ce7

		ce11 = instance_norm(conv2d(lrelu(ce10), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_ce11_conv'), name='g_ce_in11')
		ce12 = instance_norm(conv2d(lrelu(ce11), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_ce12_conv'), name='g_ce_in12')
		ce13 = instance_norm(conv2d(lrelu(ce12), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_ce13_conv'), name='g_ce_in13')
		ce13 += ce10

		return ce13


def encoder_style(image, dim=8, name='encoder_style', is_training=True, reuse=True):
	output_c_dim = image.get_shape().as_list()[-1]
	dropout_rate = 0.5 if is_training else 1.0
	with tf.variable_scope(name) as scope:
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False

		se1 = conv2d(image, dim, name='g_se1_conv')
		se2 = instance_norm(conv2d(lrelu(se1), dim*2, name='g_se2_conv'), name='g_se_in2')
		se3 = instance_norm(conv2d(lrelu(se2), dim*4, name='g_se3_conv'), name='g_se_in3')
		se4 = instance_norm(conv2d(lrelu(se3), dim*8, name='g_se4_conv'), name='g_se_in4')
		'''
		se5 = instance_norm(conv2d(lrelu(se4), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_se5_conv'), name='g_se_in5')
		se6 = instance_norm(conv2d(lrelu(se5), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_se6_conv'), name='g_se_in6')
		se7 = instance_norm(conv2d(lrelu(se6), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_se7_conv'), name='g_se_in7')
		se7 += se4
		'''
		sc = se4
		return sc


def generator(cc, sc, dim, name='generator', is_training=True, reuse=True):

	dropout_rate = 0.5 if is_training else 1.0
	with tf.variable_scope(name) as scope:
		if reuse:
			tf.get_variable_scope().reuse_variables()
		else:
			assert tf.get_variable_scope().reuse is False
		
		g = tf.concat([cc, sc], 3)
		g0 = instance_norm(conv2d(lrelu(g), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g0_conv'), name='g_in0')
		g1 = instance_norm(conv2d(lrelu(g0), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g1_conv'), name='g_in1')
		g2 = instance_norm(conv2d(lrelu(g1), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_g2_conv'), name='g_in2')
		g3 = instance_norm(conv2d(lrelu(g2), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g3_conv'), name='g_in3')
		g3 += g0

		g4 = instance_norm(conv2d(lrelu(g3), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g4_conv'), name='g_in4')
		g5 = instance_norm(conv2d(lrelu(g4), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_g5_conv'), name='g_in5')
		g6 = instance_norm(conv2d(lrelu(g5), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g6_conv'), name='g_in6')
		g6 += g3

		g7 = instance_norm(conv2d(lrelu(g6), dim*2, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g7_conv'), name='g_in7')
		g8 = instance_norm(conv2d(lrelu(g7), dim*2, k_w=3, k_h=3, d_w=1, d_h=1, name='g_g8_conv'), name='g_in8')
		g9 = instance_norm(conv2d(lrelu(g8), dim*8, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g9_conv'), name='g_in9')
		g9 += g6

		#size = g9.get_shape().as_list()
		#gs = tf.tile(sc, [1,size[1]*size[2]])
		#gs = tf.reshape(gs, [size[0],size[1],size[2],-1])
		#g = tf.concat([g9, sc], 3)
		#g9 = dropout(g9, dropout_rate)
		g11 = instance_norm(deconv2d(lrelu(g9), dim*8, name='g_g11_dconv'), name='g_in11')
		#g11 = dropout(g11, dropout_rate)
		g12 = instance_norm(deconv2d(lrelu(g11), dim*4, name='g_g12_dconv'), name='g_in12')
		#g12 = dropout(g12, dropout_rate)
		g13 = instance_norm(deconv2d(lrelu(g12), dim*2, name='g_g13_dconv'), name='g_in13')
		g14 = instance_norm(deconv2d(lrelu(g13), dim, name='g_g14_dconv'), name='g_in14')
		g15 = deconv2d(lrelu(g14), 1, k_w=1, k_h=1, d_w=1, d_h=1, name='g_g15_dconv')

		#w = tf.get_variable("g_w", [1], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
		#b = tf.get_variable("g_b", [1], initializer=tf.random_normal_initializer(0.0, 0.02, dtype=tf.float32))
		#return tf.clip_by_value(w*tanh(d8)+b, -1, 1)

		return tanh(g15)
