#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: CycleGAN model, training and testing
#	Last modified: July 20, 2018
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
		self.window_center = -400
		self.window_width = 1500
		self.has_paired = False
		self.t_print_every = 100
		self.v_print_every = 100
		self.save_every = 1000
		self.val_seed = 0
		self.checkpoint_dir = 'checkpoints'
		self.sample_dir = 'samples'
		self.out_dir = args.out_path
		self.from_scratch = True

		# Model hyper-parameters
		self.batch_size = 4
		self.fine_size = 128
		self.epoch_num = 7
		self.epoch_size = 1000
		self.gf_dim = 64
		self.df_dim = 64
		self.lambda_cc_loss = 20 #When using mse, suggest setting this to 20, and 2 for abs
		self.lambda_gp = 5
		self.lambda_g_loss = 0.2
		self.d_lr = 1e-4
		self.g_lr = 1.5e-4
		self.gan_type = 'WGAN' # choose from 'GAN', 'WGAN' and 'SGAN'
		self.iter = 8 # number of disc iterations per gen iterations
		self.d_decay = False
		self.g_decay = False
		self.use_L2_reg = False
		self.lambda_l2 = 10
		self.pretrain_steps = 200
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
		self.test_A = tf.placeholder(tf.float32,
								[self.test_size, self.image_size, self.image_size, 1],
								name='test_A')
		self.test_fake = self.generator(self.test_A, name='A2B', 
																			reuse=False, is_training=False)
		if self.has_paired:
			edge = - tf.ones(tf.shape(self.test_fake),
											dtype=self.test_fake.dtype) * 1.0
			self.test_fake = tf.where(tf.equal(self.test_A, -1.0), 
																	edge, self.test_fake)
		self.test_fake = tf.round(self.test_fake*self.window_width/2)+self.window_center

		if self.has_paired:
			self.test_B = tf.placeholder(tf.float32,
									[self.test_size, self.image_size, self.image_size, 1],
									name='test_B')
			self.test_fake = (self.test_fake-self.window_center)/self.window_width*2
			self.test_original_mse = self.loss(self.test_B,self.test_A)
			self.test_original_psnr = tf.reduce_mean(tf.image.psnr(
																self.test_B+1,self.test_A+1,max_val=2.0))
			self.test_original_ssim = tf.reduce_mean(tf.image.ssim(
																self.test_B+1,self.test_A+1,max_val=2.0))

			self.test_transfer_mse = self.loss(self.test_fake,self.test_B)
			self.test_transfer_psnr = tf.reduce_mean(tf.image.psnr(
																self.test_fake+1,self.test_B+1,max_val=2.0))
			self.test_transfer_ssim = tf.reduce_mean(tf.image.ssim(
																self.test_fake+1,self.test_B+1,max_val=2.0))

		if self.is_train:
			self.real_A = tf.placeholder(tf.float32,
											[self.batch_size, self.fine_size, self.fine_size, 1],
											name='real_A')
			self.real_B = tf.placeholder(tf.float32,
											[self.batch_size, self.fine_size, self.fine_size, 1],
											name='real_B')
			
			self.fake_B = self.generator(self.real_A, name='A2B', reuse=True)
			self.fake_A_ = self.generator(self.fake_B, name='B2A', reuse=False)
			self.fake_A = self.generator(self.real_B, name='B2A', reuse=True)
			self.fake_B_ = self.generator(self.fake_A, name='A2B', reuse=True)

			self.mse_a = self.loss(self.fake_A_,self.real_A)
			self.mse_b = self.loss(self.fake_B_,self.real_B)

			self.DB_fake = self.discriminator(self.fake_B, reuse=False, name="DB")
 			self.DA_fake = self.discriminator(self.fake_A, reuse=False, name="DA")

			self.DB_real = self.discriminator(self.real_B, reuse=True, name="DB")
			self.DA_real = self.discriminator(self.real_A, reuse=True, name="DA")

			self.g_loss_mse =  self.lambda_cc_loss * (self.mse_a + self.mse_b)
			self.db_loss_real = gan_loss(self.DB_real, 
																				tf.ones_like(self.DB_real))
			self.da_loss_real = gan_loss(self.DA_real, 
																				tf.ones_like(self.DA_real))

			# pre-training losses
			self.g_loss_p = self.loss(self.fake_B, self.real_A) \
											+ self.loss(self.fake_A, self.real_B)
			self.g_loss_p *= self.lambda_cc_loss
			self.DA2B_real = self.discriminator(self.real_A, reuse=True, name="DB")
			self.DB2A_real = self.discriminator(self.real_B, reuse=True, name="DA")
			self.d_loss_p = self.db_loss_real + self.da_loss_real \
				+ gan_loss(self.DA2B_real, tf.zeros_like(self.DA2B_real)) \
				+ gan_loss(self.DB2A_real, tf.zeros_like(self.DB2A_real))


			if self.gan_type == 'WGAN':
				self.db_loss = tf.reduce_mean(self.DB_fake - self.DB_real)
				self.da_loss = tf.reduce_mean(self.DA_fake - self.DA_real)
				self.d_loss_gan = self.da_loss + self.db_loss
				self.g_loss_gan = -tf.reduce_mean(self.DB_fake + self.DA_fake)
				self.g_loss_gan *= self.lambda_g_loss
				self.g_loss_tmp = self.g_loss_gan + self.g_loss_mse
				self.g_loss = self.g_loss_tmp * (1 + tf.sign(-self.d_loss_gan)) / 2

				alpha = tf.random_uniform(
								shape=[self.batch_size,1,1,1], minval=0.,maxval=1.)
				interpolates = alpha*self.real_A + (1-alpha)*self.fake_A
				disc_interpolates = self.discriminator(
								interpolates, reuse=True, name="DA")
				gradients = tf.gradients(disc_interpolates, [interpolates])[0]
				slopes = tf.sqrt(tf.reduce_sum(
								tf.square(gradients), reduction_indices=[1]))
				self.grad_penal_A = self.penalty(slopes)

				alpha = tf.random_uniform(
								shape=[self.batch_size,1,1,1], minval=0.,maxval=1.)
				interpolates = alpha*self.real_B + (1.0-alpha)*self.fake_B
				disc_interpolates = self.discriminator(
								interpolates, reuse=True, name="DB")
				gradients = tf.gradients(disc_interpolates, [interpolates])[0]
				slopes = tf.sqrt(tf.reduce_sum(
								tf.square(gradients), reduction_indices=[1]))
				self.grad_penal_B = self.penalty(slopes)

		 		self.gradient_penalty = self.grad_penal_A + self.grad_penal_B
				self.d_loss = self.d_loss_gan + self.lambda_gp * self.gradient_penalty
			else:
				if self.gan_type == 'GAN':
					self.gan_loss = gan_loss
				if self.gan_type == 'SGAN':
					self.gan_loss = mse_loss

				self.db_loss_fake = self.gan_loss(self.DB_fake, 
																					tf.zeros_like(self.DB_fake))
				self.db_loss = self.db_loss_real + self.db_loss_fake
				self.da_loss_fake = self.gan_loss(self.DA_fake, 
																					tf.zeros_like(self.DA_fake))
				self.da_loss = self.da_loss_real + self.da_loss_fake

				self.d_loss = self.da_loss + self.db_loss
				self.g_loss_gan = self.gan_loss(self.DB_fake, 
																				tf.ones_like(self.DB_fake)) \
													+ self.gan_loss(self.DA_fake, 
																					tf.ones_like(self.DA_fake))
				self.g_loss = self.g_loss_gan + self.g_loss_mse

			if self.use_L2_reg:
				self.l2_loss = tf.reduce_mean(self.lambda_l2 *
													tf.stack([tf.reduce_mean(v ** 2) / 2
													for v in tf.get_collection('weights')]))
				self.d_loss += self.l2_loss

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
			d_optim = tf.train.AdamOptimizer(_d_lr, beta1=self.beta1) \
								.minimize(self.d_loss, var_list=self.d_vars)

			g_optim_p = tf.train.AdamOptimizer(_g_lr, beta1=self.beta1) \
								.minimize(self.g_loss_p, var_list=self.g_vars)
			d_optim_p = tf.train.AdamOptimizer(_d_lr, beta1=self.beta1) \
								.minimize(self.d_loss_p, var_list=self.d_vars)
		if self.optimizer == 'RMSProp':
			g_optim = tf.train.RMSPropOptimizer(_g_lr, decay=0.95) \
								.minimize(self.g_loss, var_list=self.g_vars)
			d_optim = tf.train.RMSPropOptimizer(_d_lr, decay=0.95) \
								.minimize(self.d_loss, var_list=self.d_vars)

			g_optim_p = tf.train.RMSPropOptimizer(_g_lr, decay=0.95) \
								.minimize(self.g_loss_p, var_list=self.g_vars)
			d_optim_p = tf.train.RMSPropOptimizer(_d_lr, decay=0.95) \
								.minimize(self.d_loss_p, var_list=self.d_vars)

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

		for epoch in range(self.epoch_num):
			if self.g_decay:
				g_lr = self.g_lr * float(self.epoch_num-epoch)/self.epoch_num
			else:
				g_lr = self.g_lr
			
			if self.d_decay:
				d_lr = self.d_lr * float(self.epoch_num-epoch)/self.epoch_num
			else:
				d_lr = self.d_lr

			for it in range(self.epoch_size+1):

				batch_files_a = get_batch(data_a, self.batch_size)
				batch_files_b = get_batch(data_b, self.batch_size)
				batch_images_a = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a]
				batch_images_b = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b]

				if np.mod(counter, self.t_print_every) == 0:
					mse_a, mse_b, d_loss, g_loss_gan, g_loss_mse = self.sess.run(
						[self.mse_a, self.mse_b, self.d_loss,
						self.g_loss_gan, self.g_loss_mse],
						feed_dict={self.real_A: batch_images_a,
						self.real_B: batch_images_b}
					)
					print("Iteration: %d"%(counter))
					print("[Training] A-MSE: %.4f, B-MSE: %.4f" % (mse_a, mse_b))
					print("[Training] D loss: %.4f, G loss gan: %.4f, G loss mse: %.4f" \
								% (d_loss, g_loss_gan, g_loss_mse))

				if self.has_paired:
					if np.mod(counter, self.v_print_every) == 0:
						self.val(epoch, it)
						print("Iteration: %d, Time: %4.1f" \
									% (counter, time.time() - start_time))

				if np.mod(counter, self.save_every) == 0:
					self.save(counter)
					print('Check point saved in %s' %(self.checkpoint_dir))

				
				if counter < self.pretrain_steps:
					self.sess.run([g_optim_p], feed_dict={
						self.real_A: batch_images_a, self.real_B: batch_images_b, _g_lr: g_lr
					})
					self.sess.run([d_optim_p], feed_dict={
						self.real_A: batch_images_a, self.real_B: batch_images_b, _d_lr: d_lr
					})
				
				else:
					self.sess.run([g_optim], feed_dict={
						self.real_A: batch_images_a, self.real_B: batch_images_b, _g_lr: g_lr 
					})
					for t in range(self.iter):
						self.sess.run([d_optim], feed_dict={
							self.real_A: batch_images_a, self.real_B: batch_images_b, _d_lr: d_lr 
						})
						
						if t < (self.iter-1):
							batch_files_a = get_batch(data_a, self.batch_size)
							batch_files_b = get_batch(data_b, self.batch_size)
							batch_images_a = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_a]
							batch_images_b = [load_data(batch_file, fine_size=self.fine_size, window_center=self.window_center, window_width=self.window_width) for batch_file in batch_files_b]

				counter += 1		

	def load_random_sample(self):
		path_a = os.path.join(self.dataset_path, self.style_A)
		path_b = os.path.join(self.dataset_path, self.style_B)
		file_a, file_b = get_batch_paired(path_a, path_b, has_seed=True, seed=0)
		print(file_a)
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

		out_img, o_mse, o_psnr, o_ssim, t_mse, t_psnr, t_ssim = self.sess.run(
			[self.test_fake, self.test_original_mse, self.test_original_psnr,
			self.test_original_ssim, self.test_transfer_mse, 
			self.test_transfer_psnr, self.test_transfer_ssim],
			feed_dict={self.test_A: sample_a, self.test_B: sample_b}
		)
		print("[Validation]Original--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
			% (o_mse, o_psnr, o_ssim))
		print("[Validation]Transfer--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
			% (t_mse, t_psnr, t_ssim))
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
				img_out = os.path.join(folder_out, img)
				img_a = os.path.join(folder_a, img)
				dicom_a = dicom.read_file(img_a)
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
				#pixels_a_origin = np.reshape(pixels_a_origin,(self.image_size, self.image_size))
				#image[pixels_a_origin<-1500] = -1500
				#image[pixels_a_origin>400] = 400
				image = (image - dicom_a.RescaleIntercept) / dicom_a.RescaleSlope
				image = image.astype('uint16')
				dicom_a.PixelData = image.tostring()
				dicom_a.save_as(img_out)

		if self.has_paired:
			mse_1 /= len(sample_files)
			mse_2 /= len(sample_files)
			psnr_1 /= len(sample_files)
			psnr_2 /= len(sample_files)
			ssim_1 /= len(sample_files)
			ssim_2 /= len(sample_files)
			print("Original--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
				% (mse_1, psnr_1, ssim_1))
			print("Transfer--MSE: %.4f, PSNR: %.4f, SSIM: %.4f" \
				% (mse_2, psnr_2, ssim_2))
