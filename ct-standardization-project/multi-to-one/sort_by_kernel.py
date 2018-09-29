#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 23, 2018
#	Functions: sort the CT cases by convolution kernel					
#	Last modified: July 23, 2018
#####################################################

import os
import shutil
import pydicom as dicom
import time
import numpy as np

folder = '/home/boya/Data/CT'

train_folder = os.path.join(folder,'train')
val_folder = os.path.join(folder,'val')

train_sets = os.listdir(train_folder)
val_sets = os.listdir(val_folder)

kernels = {}
txt_file = '/home/boya/Data/CT/train/train_transfer/SeqsSets/CT_Training_data_1.txt'
f = open(txt_file, 'w')
path = '/home/boya/Data/CT/train/train_transfer/anno'
cases = os.listdir(path)
for case in cases:
	f.write(case)
	f.write('\n')
f.close()
'''
path = '/home/boya/Data/CT/val/2018_05_25_HeNanXiongKeYiYuan_test/dcm'
cases = os.listdir(path)
for case in cases:
	case_path = os.path.join(path, case)
	imgs = os.listdir(case_path)
	for img in imgs:
		dicom_img = dicom.read_file(os.path.join(case_path,img))
		print(img)
		pixels = dicom_img.pixel_array

path = '/home/boya/Data/CT/train/CT_Training_data/dcm'
cases = os.listdir(path)
for case in cases:
	case_path = os.path.join(path, case)
	imgs = os.listdir(case_path)
	dicom_img = dicom.read_file(os.path.join(case_path,imgs[0]))
	pixels = dicom_img.pixel_array
	if len(pixels) != 512:
		print(case)
'''
'''
for data_set in train_sets:
	path = os.path.join(train_folder, data_set, 'dcm')
	for case in os.listdir(path):
		case_path = os.path.join(path, case)
		img = np.random.choice(os.listdir(case_path),1)
		dicom_img = dicom.read_file(os.path.join(case_path,img[0]))
		ker = dicom_img.ConvolutionKernel
		ker = "".join(ker)
		if ker in kernels.keys():
			kernels[ker] += 1
		else:
			kernels[ker] = 1

kernel_l = {}
kernel_h = {}
for data_set in val_sets:
	path = os.path.join(val_folder, data_set, 'dcm')
	for case in os.listdir(path):
		case_path = os.path.join(path, case)
		img = np.random.choice(os.listdir(case_path),1)
		dicom_img = dicom.read_file(os.path.join(case_path,img[0]))
		ker = dicom_img.ConvolutionKernel
		ker = "".join(ker)
		
		if ker in kernels.keys():
			kernels[ker] += 1
		else:
			kernels[ker] = 1


f = zip(kernels.values(),kernels.keys())
pairs = sorted(f)
for v,k in pairs:
	print("kernel: %s, number: %d"%(k, v))

def copyFiles(sourceDir, targetDir): 
	if os.path.exists(targetDir) is False:
		os.makedirs(targetDir)
	for file in os.listdir(sourceDir): 
		sourceFile = os.path.join(sourceDir,  file) 
		targetFile = os.path.join(targetDir,  file) 
		if not os.path.exists(targetFile):  
			shutil.copyfile(sourceFile, targetFile)

style='STANDARD'
out_folder = os.path.join(folder,'val',style)
for data_set in val_sets:
	path = os.path.join(val_folder, data_set, 'dcm')
	for case in os.listdir(path):
		case_path = os.path.join(path, case)
		img = np.random.choice(os.listdir(case_path),1)
		dicom_img = dicom.read_file(os.path.join(case_path,img[0]))
		ker = dicom_img.ConvolutionKernel
		ker = "".join(ker)
		if ker == style:
			s_dir = case_path
			t_dir = os.path.join(out_folder, 'dcm', case)
			copyFiles(s_dir, t_dir)
			s_dir = os.path.join(val_folder, data_set, 'anno', case)
			t_dir = os.path.join(out_folder, 'anno', case)
			copyFiles(s_dir, t_dir)
'''
#styles = ['Lung','FC51','FC53']

'''
if os.path.exists(output_folder) is False:
	os.makedirs(output_folder)

#####################################################
#	copy files from one folder to another
#####################################################
def copyFiles(sourceDir, targetDir): 
	if os.path.exists(targetDir) is False:
		os.makedirs(targetDir)
	for file in os.listdir(sourceDir): 
		sourceFile = os.path.join(sourceDir,  file) 
		targetFile = os.path.join(targetDir,  file) 
		if not os.path.exists(targetFile):  
			shutil.copyfile(sourceFile, targetFile)

folders = os.listdir(input_folder)
for folder in folders:
	print("Processing patient: %s" % folder)
	start_time = time.time()
	do_move = True
	for style in styles:
		path = os.path.join(input_folder,folder,style)
		if not os.path.exists(path):
			do_move = False
	if do_move:
		for style in styles:
			s_dir = os.path.join(input_folder,folder,style)
			t_dir = os.path.join(output_folder,style,folder)
			copyFiles(s_dir, t_dir)
	end_time = time.time()
	print('Elapsed time: %fs' % (end_time - start_time))
'''
