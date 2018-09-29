#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: Disentangled representation model
#	Last modified: Aug 17, 2018
#####################################################

from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
import scipy
import pydicom as dicom

from ops import *
from utils import *
from modules import *

class CycleGAN(object):

	def __init__(self, sess, args, is_train=True):

		# Program and data parameters
		self.sess = sess
		self.is_train = is_train
		self.dataset_path = args.dataset_path
		self.test_size = 1
		self.image_size = 512
		self.c_dim = 1
		self.style_A = args.style_A
		self.style_B = args.style_B
		self.style_C = args.style_C
		self.style_D = ['LDHDStdSS0', 'LDHDStdSS80', 'LDHDLungSS40']
		self.style_E = args.style_E
		self.window_center = -400
		self.window_width = 1500
		self.has_paired = args.paired
		self.t_print_every = 50
		self.v_print_every = 50
		self.save_every = 100
		self.val_seed = 12
		self.checkpoint_dir = args.checkpoint_dir
		self.sample_dir = args.sample_dir
		self.out_dir = args.out_path
		self.from_scratch = args.from_scratch
		self.num_sup = args.num_supervised
		self.num_unsup = args.num_unsupervised

		# Model hyper-parameters
		self.w_gan = args.w_gan
		self.w_cc = args.w_cc
		self.w_c = args.w_c
		self.w_rec = args.w_rec
		self.w_trans = args.w_trans
		self.batch_size = 1
		self.fine_size = 512
		self.epoch_num = 500
		self.epoch_size = 100
		self.gf_dim = 96
		self.df_dim = 64
		self.lambda_gp = 10
		self.lambda_g_loss = 0.01
		self.d_lr = 1e-4
		self.g_lr = 1e-4
		self.iter = args.d_iters # number of disc iterations per gen iterations
		self.d_decay = False
		self.g_decay = False
		self.use_L2_reg = False
		self.lambda_l2 = 10
		self.beta1 = 0.5
		self.discriminator = discriminator
		self.generator = generator
		self.loss = mse_loss # choose from mse_loss and abs_loss
		self.optimizer = 'Adam'
		# choose from 'Adam' and 'RMSProp'
		self.penalty = penalty_abs # choose from penalty_abs and penalty_mse

		self.build_model()

	def run(self):
		if self.is_train:
			self.train()
		else:
			self.test()

	def build_model(self):

		mask = np.zeros((self.image_size, self.image_size))
		_center = self.image_size/2 - 0.5
		for i in range(self.image_size):
			for j in range(self.image_size):
				if ((i-_center)**2 + (j-_center)**2)>_center**2:
					mask[i][j] = 1.0
		mask = mask.reshape((1, self.image_size, self.image_size, 1))
		self.mask_test = tf.convert_to_tensor(np.repeat(mask,self.test_size,axis=0))
		self.mask_train = tf.convert_to_tensor(np.repeat(mask,self.batch_size,axis=0))
		self.edge_test = - tf.ones([self.test_size, self.image_size, self.image_size, 1]) * 1.0
		self.edge_train = - tf.ones([self.batch_size, self.image_size, self.image_size, 1]) * 1.0

		self.test_A = tf.placeholder(tf.float32,
								[self.test_size, self.image_size, self.image_size, 1],
								name='test_A')
		#self.test_A = tf.where(tf.equal(self.mask_test, 1.0), self.edge_test, self.test_A)

		code1 = tf.get_variable('g_style_feature1', [1, 32, 32, self.gf_dim*8], initializer=tf.truncated_normal_initializer(stddev=1))
		code2 = tf.get_variable('g_style_feature2', [1, 32, 32, self.gf_dim*8], initializer=tf.truncated_normal_initializer(stddev=1))
		self.code_check = tf.reduce_mean(code2**2)
		content = encoder_content(self.test_A, dim=self.gf_dim, reuse=False, is_training=False)
		style = encoder_style(self.test_A, dim=self.gf_dim, reuse=False, is_training=False)
		self.test_fake = generator(content, code2, dim=self.gf_dim, is_training=False, reuse=False)
		self.test_fake_cmp = generator(content, style, dim=self.gf_dim, is_training=False, reuse=True)

		#self.test_fake = tf.where(tf.equal(self.mask_test, 1.0), self.edge_test, self.test_fake)
		#self.test_fake_cmp = tf.where(tf.equal(self.mask_test, 1.0), self.edge_test, self.test_fake_cmp)
		self.test_fake = tf.round(self.test_fake*self.window_width/2)+self.window_center
		self.test_fake_cmp = tf.round(self.test_fake_cmp*self.window_width/2)+self.window_center

		if self.has_paired:
			self.test_B = tf.placeholder(tf.float32,
									[self.test_size, self.image_size, self.image_size, 1],
									name='test_B')
			#self.test_B = tf.where(tf.equal(self.mask_test, 1.0), self.edge_test, self.test_B)
			self.test_fake_p = (self.test_fake-self.window_center)/self.window_width*2
			self.test_fake_cmp_p = (self.test_fake_cmp-self.window_center)/self.window_width*2
			self.test_original_mse = self.loss(self.test_B,self.test_A)
			self.test_original_psnr = tf.reduce_mean(tf.image.psnr(
																self.test_B+1,self.test_A+1,max_val=2.0))
			self.test_original_ssim = tf.reduce_mean(tf.image.ssim(
																self.test_B+1,self.test_A+1,max_val=2.0))

			self.test_transfer_mse = self.loss(self.test_fake_p,self.test_B)
			self.test_transfer_psnr = tf.reduce_mean(tf.image.psnr(
																self.test_fake_p+1,self.test_B+1,max_val=2.0))
			self.test_transfer_ssim = tf.reduce_mean(tf.image.ssim(
																self.test_fake_p+1,self.test_B+1,max_val=2.0))

			self.test_cmp_mse = self.loss(self.test_fake_cmp_p,self.test_B)
			self.test_cmp_psnr = tf.reduce_mean(tf.image.psnr(
																self.test_fake_cmp_p+1,self.test_B+1,max_val=2.0))
			self.test_cmp_ssim = tf.reduce_mean(tf.image.ssim(
																self.test_fake_cmp_p+1,self.test_B+1,max_val=2.0))

		if self.is_train:
			self.real_A = tf.placeholder(tf.float32,
											[self.batch_size, self.fine_size, self.fine_size, 1],
											name='real_A')
			self.real_B = tf.placeholder(tf.float32,
											[self.batch_size, self.fine_size, self.fine_size, 1],
											name='real_B')

			self.real_A_ = tf.placeholder(tf.float32,
											[self.batch_size, self.fine_size, self.fine_size, 1],
											name='real_A_')
			self.real_B_ = tf.placeholder(tf.float32,
											[self.batch_size, self.fine_size, self.fine_size, 1],
											name='real_B_')
			#self.real_A = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self.real_A)
			#self.real_B = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self.real_B)
			#self.real_A_ = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self.real_A_)
			#self.real_B_ = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self.real_B_)

			self.real_c_a = encoder_content(self.real_A, dim=self.gf_dim, reuse=True, is_training=True)
			self.real_c_b = encoder_content(self.real_B, dim=self.gf_dim, reuse=True, is_training=True)
			self.real_s_a = encoder_style(self.real_A, dim=self.gf_dim, reuse=True, is_training=True)
			self.real_s_b = encoder_style(self.real_B, dim=self.gf_dim, reuse=True, is_training=True)
			self.fake_A = generator(self.real_c_a, self.real_s_a, dim=self.gf_dim, is_training=True, reuse=True)
			self.fake_B = generator(self.real_c_b, self.real_s_b, dim=self.gf_dim, is_training=True, reuse=True)
			self._fake_B = generator(self.real_c_a, self.real_s_b, dim=self.gf_dim, is_training=True, reuse=True)
			self._fake_A = generator(self.real_c_b, self.real_s_a, dim=self.gf_dim, is_training=True, reuse=True)
			#self.fake_A = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self.fake_A)
			#self.fake_B = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self.fake_B)
			#self._fake_A = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self._fake_A)
			#self._fake_B = tf.where(tf.equal(self.mask_train, 1.0), self.edge_train, self._fake_B)
			
			self.fake_c_a = encoder_content(self._fake_B, dim=self.gf_dim, reuse=True, is_training=True)
			self.fake_c_b = encoder_content(self._fake_A, dim=self.gf_dim, reuse=True, is_training=True)
			self.cc_fake_A = generator(self.fake_c_a, self.real_s_a, dim=self.gf_dim, is_training=True, reuse=True)
			self.cc_fake_B = generator(self.fake_c_b, self.real_s_b, dim=self.gf_dim, is_training=True, reuse=True)
			
			self.code_loss1 = self.loss(self.real_s_b, code1)
			self.var1 = self.loss(self.real_s_a, code1)
			self.code_loss_n1 = self.code_loss1/self.var1
			self.code_loss_n1 *= self.w_c

			self.code_loss2 = self.loss(self.real_s_b, code2)
			self.var2 = self.loss(self.real_s_a, code2)
			self.code_loss_n2 = self.code_loss2/self.var2
			self.code_loss_n2 *= self.w_c

			self.rec_a = self.loss(self.fake_A, self.real_A)
			self.rec_b = self.loss(self.fake_B, self.real_B)
			self.g_loss_rec =  self.rec_a + self.rec_b
			self.g_loss_rec *= self.w_rec
			
			self.cc_a = self.loss(self.cc_fake_A, self.real_A)
			self.cc_b = self.loss(self.cc_fake_B, self.real_B)
			self.g_loss_cc =  self.cc_a + self.cc_b
			self.g_loss_cc *= self.w_cc
			
			self.trans_a = self.loss(self._fake_A, self.real_A)
			self.trans_b = self.loss(self._fake_B, self.real_B)
			self.g_loss_trans =  self.trans_a + self.trans_b
			self.g_loss_trans *= self.w_trans
			'''
			self.content_a = self.loss(self.real_c_a, self.fake_c_a)
			self.content_b = self.loss(self.real_c_b, self.fake_c_b)
			self.content_loss = self.content_a + self.content_b
			self.content_loss *= 0
			'''

			if self.iter > 0:
				self.DB_fake = self.discriminator(tf.concat([self.fake_B,self.real_B_],3), reuse=False)
				self.DB_real = self.discriminator(tf.concat([self.real_B,self.real_B_],3), reuse=True)
				self.DA_fake = self.discriminator(tf.concat([self.fake_A,self.real_A_],3), reuse=True)
				self.DA_real = self.discriminator(tf.concat([self.real_A,self.real_A_],3), reuse=True)
				self.db_loss = tf.reduce_mean(self.DB_fake - self.DB_real)
				self.da_loss = tf.reduce_mean(self.DA_fake - self.DA_real)
				self.d_loss_gan = self.db_loss + self.da_loss
				self.g_loss_gan = -tf.reduce_mean(self.DB_fake+self.DA_fake)
				self.g_loss_gan *= self.lambda_g_loss
				alpha = tf.random_uniform(
								shape=[self.batch_size,1,1,1], minval=0.,maxval=1.)
				interpolates = alpha*self.real_B + (1-alpha)*self._fake_B
				disc_interpolates = self.discriminator(tf.concat([interpolates,self.real_B_],3), reuse=True)
				gradients = tf.gradients(disc_interpolates, [interpolates])[0]
				slopes = tf.sqrt(tf.reduce_sum(
								tf.square(gradients), reduction_indices=[1]))
				self.grad_penal_B = self.penalty(slopes)

				alpha = tf.random_uniform(
								shape=[self.batch_size,1,1,1], minval=0.,maxval=1.)
				interpolates = alpha*self.real_A + (1-alpha)*self._fake_A
				disc_interpolates = self.discriminator(tf.concat([interpolates,self.real_A_],3), reuse=True)
				gradients = tf.gradients(disc_interpolates, [interpolates])[0]
				slopes = tf.sqrt(tf.reduce_sum(
								tf.square(gradients), reduction_indices=[1]))
				self.grad_penal_A = self.penalty(slopes)

				self.gradient_penalty = self.grad_penal_B + self.grad_penal_A
				self.d_loss = self.d_loss_gan + self.lambda_gp * self.gradient_penalty

			self._w_1 = tf.placeholder(tf.float32, name='w_unsupervised')
			self._w_2 = tf.placeholder(tf.float32, name='w_supervised')
			self.g_loss = self.g_loss_rec + \
										(self.g_loss_cc + self.code_loss_n2) * self._w_1 + \
										(self.g_loss_trans + self.code_loss_n1) * self._w_2


		t_vars = tf.trainable_variables()
		self.g_vars = [var for var in t_vars if 'g_' in var.name]
		self.d_vars = [var for var in t_vars if 'd_' in var.name]

		self.saver = tf.train.Saver()


	def train(self):

		_d_lr = tf.placeholder(tf.float32, name='d_learning_rate')
		_g_lr = tf.placeholder(tf.float32, name='g_learning_rate')
		
		if self.optimizer == 'Adam':
			g_optim = tf.train.AdamOptimizer(_g_lr, beta1=self.beta1) \
								.minimize(self.g_loss, var_list=self.g_vars)
			if self.iter > 0:
				d_optim = tf.train.AdamOptimizer(_d_lr, beta1=self.beta1) \
									.minimize(self.d_loss, var_list=self.d_vars)

		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)

		counter = 0
		start_time = time.time()
		if not self.from_scratch:
			if self.load():
				print(" [*] Load succeeded!")
			else:
				print(" [!] Load failed...")

		data_a = os.path.join(self.dataset_path, self.style_A)
		data_b = os.path.join(self.dataset_path, self.style_B)
		data_c = self.style_C
		data_e = self.style_E

		for epoch in range(self.epoch_num):
			if self.g_decay:
				g_lr = self.g_lr * float(self.epoch_num-epoch)/self.epoch_num
			else:
				g_lr = self.g_lr
			
			if self.d_decay:
				d_lr = self.d_lr * float(self.epoch_num-epoch)/self.epoch_num
			else:
				d_lr = self.d_lr
			style_d = np.random.choice(self.style_D)
			data_d = os.path.join(self.dataset_path, style_d)
			for it in range(self.epoch_size+1):
				
				for p in range(self.num_unsup):
					batch_files_a = get_batch(data_c, self.batch_size)
					batch_files_b = get_batch(data_e, self.batch_size)
					batch_files_a_ = get_batch(data_c, self.batch_size)
					batch_files_b_ = get_batch(data_e, self.batch_size)
					batch_images_a = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a]
					batch_images_b = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b]
					batch_images_a_ = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a_]
					batch_images_b_ = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b_]

					if p==0 and np.mod(counter, self.t_print_every) == 0:
						rec_a, rec_b, code_loss, var, cc_a, cc_b = self.sess.run(
							[self.rec_a, self.rec_b, self.code_loss2, self.var2,
							self.cc_a, self.cc_b],
							feed_dict={self.real_A: batch_images_a, self.real_B: batch_images_b, 
							self.real_A_: batch_images_a_, self.real_B_: batch_images_b_}
						)
						print('')
						print('-------------------Unsupervised------------------')
						print("Iteration: %d, Time: %4.1f" \
										% (counter, time.time() - start_time))
						print("[Reconstruction]  A: %.4f, B: %.4f" % (rec_a, rec_b))
						print("[Cycel Consistency]A: %.4f, B: %.4f" % (cc_a, cc_b))
						#print("[Content Loss] %.6f" % (c_loss))
						print("[Style Distance]  A: %.4f, B: %.4f" % (var, code_loss))
						#print("[Discriminator]  Gan loss: %.4f, Gradient Penalty: %.4f" % (d_gan, grad_penal))


					self.sess.run([g_optim], feed_dict={
						self.real_A: batch_images_a, self.real_B: batch_images_b, 
						self.real_A_: batch_images_a_, self.real_B_: batch_images_b_, 
						_g_lr: g_lr, self._w_1: 1.0, self._w_2: 0.0
					})
					for t in range(self.iter):
							self.sess.run([d_optim], feed_dict={
								self.real_A: batch_images_a, self.real_B: batch_images_b, 
								self.real_A_: batch_images_a_, self.real_B_: batch_images_b_,
								_d_lr: d_lr
							})
							
							if t < (self.iter-1):
								seed = np.random.randint(1000000, size=(self.batch_size, 2))
								batch_files_a = get_batch(data_c, self.batch_size)
								batch_files_b = get_batch(data_b, self.batch_size)
								batch_files_a_ = get_batch(data_c, self.batch_size)
								batch_files_b_ = get_batch(data_b, self.batch_size)
								batch_images_a = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a]
								batch_images_b = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b]
								batch_images_a_ = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a_]
								batch_images_b_ = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b_]



				for q in range(self.num_sup):
					seed = np.random.randint(1000000, size=(self.batch_size, 2))
					batch_files_a = get_batch(data_d, self.batch_size, seed=seed)
					batch_files_b = get_batch(data_b, self.batch_size, seed=seed)
					batch_images_a = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a]
					batch_images_b = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b]


					if q==0 and np.mod(counter, self.t_print_every) == 0:
						rec_a, rec_b, code_loss, var, trans_a, trans_b = self.sess.run(
							[self.rec_a, self.rec_b, self.code_loss1, self.var1,
							self.trans_a, self.trans_b],
							feed_dict={self.real_A: batch_images_a, self.real_B: batch_images_b, 
							self.real_A_: batch_images_a, self.real_B_: batch_images_b}
						)
						print('')
						print('--------------------Supervised-------------------')
						print("Iteration: %d, Time: %4.1f" \
										% (counter, time.time() - start_time))
						print("[Reconstruction]  A: %.4f, B: %.4f" % (rec_a, rec_b))
						print("[Transfer]  A: %.4f, B: %.4f" % (trans_a, trans_b))
						#print("[Content Loss] %.6f" % (c_loss))
						print("[Style Distance]  A: %.4f, B: %.4f" % (var, code_loss))


					self.sess.run([g_optim], feed_dict={
						self.real_A: batch_images_a, self.real_B: batch_images_b, 
						self.real_A_: batch_images_a, self.real_B_: batch_images_b, 
						_g_lr: g_lr, self._w_1: 0.0, self._w_2: 1.0
					})


				if self.has_paired:
					if np.mod(counter, self.v_print_every) == 0:
						self.val(epoch, it)

				if np.mod(counter, self.save_every) == 0:
					self.save(counter)
					print('Check point saved in %s' %(self.checkpoint_dir))


				counter += 1		

	def load_random_sample(self):
		path_a = os.path.join(self.dataset_path, self.style_A)
		path_b = os.path.join(self.dataset_path, self.style_B)
		file_a, file_b = get_batch_paired(path_a, path_b, has_seed=True, seed=self.val_seed)
		img_a = load_data(file_a, is_train=False, window_center=self.window_center, window_width=self.window_width)
		img_b = load_data(file_b, is_train=False, window_center=self.window_center, window_width=self.window_width)
		size = img_a.shape[0]
		img_a = np.reshape(img_a, [1,size,size,1])
		img_b = np.reshape(img_b, [1,size,size,1])
		return img_a, img_b

	def val(self, epoch, idx):
		sample_a, sample_b = self.load_random_sample()
		sample_dir = os.path.join(self.dataset_path, self.sample_dir)
		if not os.path.exists(sample_dir):
			os.makedirs(sample_dir)

		out_img, o_mse, o_psnr, o_ssim, t_mse, t_psnr, t_ssim, c_mse, c_psnr, c_ssim = self.sess.run(
			[self.test_fake_p, self.test_original_mse, self.test_original_psnr,self.test_original_ssim,
			self.test_transfer_mse, self.test_transfer_psnr, self.test_transfer_ssim,
			self.test_cmp_mse, self.test_cmp_psnr, self.test_cmp_ssim],
			feed_dict={self.test_A: sample_a, self.test_B: sample_b}
		)
		print('')
		print("[Validation]  Original--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
			% (o_mse, o_psnr, o_ssim))
		print("[Validation]  Transfer--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
			% (t_mse, t_psnr, t_ssim))
		print("[Validation]  Comparison--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
			% (c_mse, c_psnr, c_ssim))
		save_png(out_img, '{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
		if epoch == 0 and idx == 0:
			save_png(sample_a, '{}/A.png'.format(sample_dir))
			save_png(sample_b, '{}/B.png'.format(sample_dir))

	def save(self, step):

		checkpoint_dir = os.path.join(self.dataset_path, self.checkpoint_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir,'gan.model'))

	def load(self):
		print(" [*] Reading checkpoint...")

		checkpoint_dir = os.path.join(self.dataset_path, self.checkpoint_dir)
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False

	def test(self):
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		if self.load():
			print(" [*] Load succeeded!")
		else:
			print(" [!] Load failed...")

		self.out_dir = os.path.join(self.dataset_path, self.out_dir)
		if not os.path.exists(self.out_dir):
			os.makedirs(self.out_dir)

		code_check = self.sess.run(self.code_check)
		print(code_check)
		data_a = os.path.join(self.dataset_path, self.style_A)
		data_b = os.path.join(self.dataset_path, self.style_B)
		if self.has_paired:
			mse_1 = 0
			mse_2 = 0
			psnr_1 = 0
			psnr_2 = 0
			ssim_1 = 0
			ssim_2 = 0
		folders = os.listdir(data_a)
		count = 0
		for folder in folders:
			print('Processing patient: -----%s-----'%(folder))
			folder_out = os.path.join(self.out_dir, folder)
			if not os.path.exists(folder_out):
				os.makedirs(folder_out)
			folder_a = os.path.join(data_a, folder)
			if self.has_paired:
				folder_b = os.path.join(data_b, folder)
			imgs = os.listdir(folder_a)
			for img in imgs:
				count += 1
				img_out = os.path.join(folder_out, img)
				img_a = os.path.join(folder_a, img)
				dicom_a = dicom.read_file(img_a)
				try:
					pixels_a = dicom_a.pixel_array
					pixels_a_origin = pixels_a.astype('float32') * dicom_a.RescaleSlope \
										+ dicom_a.RescaleIntercept
					pixels_a = np.reshape(pixels_a_origin,(1, self.image_size, self.image_size, 1))
					pixels_a = (pixels_a-self.window_center)/self.window_width*2
					pixels_a[pixels_a<-1.0] = -1.0
					pixels_a[pixels_a>1.0] = 1.0
					if not self.has_paired:
						image = self.sess.run(self.test_fake, feed_dict={self.test_A: pixels_a})
					else:
						img_b = os.path.join(folder_b, img)
						dicom_b = dicom.read_file(img_b)
						pixels_b = dicom_b.pixel_array
						pixels_b = pixels_b.astype('float32') * dicom_b.RescaleSlope \
											+ dicom_b.RescaleIntercept
						pixels_b = np.reshape(pixels_b,(1, self.image_size, self.image_size, 1))
						pixels_b = (pixels_b-self.window_center)/self.window_width*2
						pixels_b[pixels_b<-1.0] = -1.0
						pixels_b[pixels_b>1.0] = 1.0
						image, o_mse, o_psnr, o_ssim, t_mse, t_psnr, t_ssim = self.sess.run([
								self.test_fake, self.test_original_mse, self.test_original_psnr,
								self.test_original_ssim, self.test_transfer_mse, 
								self.test_transfer_psnr,self.test_transfer_ssim],
								feed_dict={self.test_A: pixels_a, self.test_B: pixels_b})
						mse_1 += o_mse
						mse_2 += t_mse
						psnr_1 += o_psnr
						psnr_2 += t_psnr
						ssim_1 += o_ssim
						ssim_2 += t_ssim
					image = np.reshape(image,(self.image_size, self.image_size))
					image[image<-1024]=-1024
					image = (image - dicom_a.RescaleIntercept) / dicom_a.RescaleSlope
					image = image.astype('uint16')
					dicom_a.PixelData = image.tostring()
					dicom_a.save_as(img_out)
				except (TypeError,AttributeError), e:
					print(img_a)
					print(e)

		if self.has_paired:
			mse_1 /= count
			mse_2 /= count
			psnr_1 /= count
			psnr_2 /= count
			ssim_1 /= count
			ssim_2 /= count
			print("Original--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
				% (mse_1, psnr_1, ssim_1))
			print("Transfer--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
				% (mse_2, psnr_2, ssim_2))
