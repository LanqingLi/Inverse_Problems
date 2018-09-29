#####################################################
#	Author: Boya Ren @ Infervision
#	Date: July 20, 2018
#	Functions: Main funtion
#	Last modified: Aug 17, 2018
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
parser.add_argument('--style_C', dest='style_C', default='/home/boya/Data/CT/train/CT_Training_data/dcm', help='path to the unsupervised learning data folder')
parser.add_argument('--style_E', dest='style_E', default='/home/boya/Data/CT/val/STANDARD/dcm', help='path to the unsupervised learning target style data folder')
parser.add_argument('--gpu', dest='gpu', default='1', help='which GPU to use')
parser.add_argument('--out_path', dest='out_path', default='out', help='output_dir')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoints', help='checkpoint_dir')
parser.add_argument('--sample_dir', dest='sample_dir', default='samples', help='checkpoint_dir')
parser.add_argument('--paired', dest='paired', type=bool, default=True, help='whether or not you have paired data')
parser.add_argument('--from_scratch', dest='from_scratch', type=bool, default=True, help='whether or to train the model from scratch')
paired_parser = parser.add_mutually_exclusive_group(required=False)
paired_parser.add_argument('--has-paired', dest='paired', action='store_true')
paired_parser.add_argument('--no-paired', dest='paired', action='store_false')
scratch_parser = parser.add_mutually_exclusive_group(required=False)
scratch_parser.add_argument('--from-scratch', dest='from_scratch', action='store_true')
scratch_parser.add_argument('--continue', dest='from_scratch', action='store_false')
parser.add_argument('--num_supervised', dest='num_supervised', type=int, default=1, help='number of supervised learning iterations per main loop')
parser.add_argument('--num_unsupervised', dest='num_unsupervised', type=int, default=1, help='number of unsupervised learning iterations per main loop')
parser.add_argument('--d_iters', dest='d_iters', type=int, default=7, help='number of discriminator iterations per generator')
parser.add_argument('--w_gan', dest='w_gan', type=float, default=0.01, help='weight for gan loss')
parser.add_argument('--w_rec', dest='w_rec', type=float, default=1, help='weight for reconstruction loss')
parser.add_argument('--w_trans', dest='w_trans', type=float, default=1, help='weight for transformation loss')
parser.add_argument('--w_cc', dest='w_cc', type=float, default=1, help='weight for cycle consistency loss')
parser.add_argument('--w_c', dest='w_c', type=float, default=10, help='weight for code loss')

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
