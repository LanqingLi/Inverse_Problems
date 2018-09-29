#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 19, 2018
#	Functions: Put CT images of different patients into
#							different style folders
#	Last modified: July 19, 2018
#####################################################

import os
import shutil
import time

input_folder = '/home/boya/Data/2018_03_26_WuHanTongJi'
output_folder = '/home/boya/Data/CT-styles'

styles = ['LDHDLungSS40', 'LDHDStdSS0', 'LDHDStdSS80', 
					'LDLungSS0', 'LDStdSS80', 'StdSS60']

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
