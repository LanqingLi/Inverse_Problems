#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: Data utils. Read batches and do basic 
#							data processing
#	Last modified: Aug 17, 2018
#####################################################

import os
import scipy.misc
import numpy as np
import pydicom as dicom

def get_batch(path, batch_size, seed=None):
	file_list = []
	folders = os.listdir(path)
	for i in range(batch_size):
		if seed is not None:
			np.random.seed(seed = seed[i,0])
		folder = np.random.choice(folders, 1)
		folder = os.path.join(path, folder[0])
		files = os.listdir(folder)
		if seed is not None:
			np.random.seed(seed = seed[i,1])
		img = np.random.choice(files, 1)
		file_list.append(os.path.join(folder, img[0]))
	return file_list

def get_batch_paired(path_A, path_B, has_seed=False, seed=0):
	folders = os.listdir(path_A)
	if has_seed:
		np.random.seed(seed = seed)
	folder = np.random.choice(folders, 1)
	folder_a = os.path.join(path_A, folder[0])
	folder_b = os.path.join(path_B, folder[0])
	files = os.listdir(folder_a)
	if has_seed:
		np.random.seed(seed = seed)
	img = np.random.choice(files, 1)
	img_a = os.path.join(folder_a, img[0])
	img_b = os.path.join(folder_b, img[0])
	return img_a, img_b
	

#####################################################
# load image sample from style A and style B
#####################################################
def load_data(path, fine_size=128, is_train=True, window_center = -600, window_width = 1500):
	img = load_img(path, center=window_center, width=window_width)
	if is_train:
		img = crop_image(img, fine_size=fine_size)
	size = img.shape[0]
	img = np.reshape(img, (size,size,1))
	return img

#####################################################
# load a dicom image and normalize to [-1,1]
#####################################################
def load_img(path, center = -600, width = 1500):
	img = read_dicom(path)

	scaled_img = ((img - center) / width) * 2
	scaled_img[scaled_img>1] = 1.0
	scaled_img[scaled_img<-1] = -1.0
	#scaled_img[img<=-1000] = -1.0
	'''
	scaled_img = ((img - center) / width + 0.5) * 255
	scaled_img[scaled_img>255] = 255
	scaled_img[scaled_img<0] = 0
	scaled_img = scaled_img.astype(int)
	scaled_img = (scaled_img-127.5)/127.5
	'''
	return scaled_img

#####################################################
#	crop the image to fine_size x fine_size
#####################################################
def crop_image(img, fine_size=128):
	if fine_size < img.shape[0]/4:
		load_size = img.shape[0]
		center = load_size / 2.0 - 0.5
		h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		h2 = h1 + fine_size
		w2 = w1 + fine_size
		d = np.maximum((h1-center)**2,(h2-center)**2) \
				+ np.maximum((w1-center)**2,(w2-center)**2) - center**2
		while d > 0:
			h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
			w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
			h2 = h1 + fine_size
			w2 = w1 + fine_size
			d = np.maximum((h1-center)**2,(h2-center)**2) \
					+ np.maximum((w1-center)**2,(w2-center)**2) - center**2
		img = img[h1:h2, w1:w2]
	elif fine_size < img.shape[0]:
		load_size = img.shape[0]
		h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
		h2 = h1 + fine_size
		w2 = w1 + fine_size
		img = img[h1:h2, w1:w2]
	return img

#####################################################
# crop the image to fine_size x fine_size
#####################################################
def read_dicom(path):
	dicom_img = dicom.read_file(path)
	img = dicom_img.pixel_array
	img = img.astype('float32') * dicom_img.RescaleSlope \
				+ dicom_img.RescaleIntercept
	return img

def save_png(image, path):
	img = image[0,:,:,0]
	img = scipy.misc.toimage(img,cmin=-1,cmax=1)
	img.save(path)
