#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 20, 2018
#	Functions: Main funtion
#	Last modified: July 20, 2018
#####################################################

import argparse
import os
from model import CycleGAN
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', dest='dataset_path', default='./', help='path of the dataset')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--style_A', dest='style_A', default='LDHDLungSS40', help='style A folder')
parser.add_argument('--style_B', dest='style_B', default='LDStdSS80', help='style B folder')
parser.add_argument('--gpu', dest='gpu', default='1', help='which GPU to use')
parser.add_argument('--out_path', dest='out_path', default='out', help='output_dir')

args = parser.parse_args()

def main(_):
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	if args.phase == 'train':
		is_train = True
	else:
		is_train = False
	with tf.Session() as sess:
		model = CycleGAN(sess, args, is_train=is_train)
		model.run()

if __name__ == '__main__':
	tf.app.run()
