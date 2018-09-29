Multi-to-One CT imaging style transfer
-----------------------------------------------------

This repository provides an unfinished model that proposes a possible way of disentangled representation. The model works in supervised, unsupervised (still under construction) and semi-supervised ways. The model is able to transfer CT images from multiple domains to a single domain (like standard kernel with normal dose). At this point, we suggest use semi-supervised learning. If you have a GPU with larger RAM (I am using GTX 1080 8GB), we strongly advise you increase fiter numbers in encoder and decoders before advancing to the program.

## Requirements
* Python 2.7
* Cuda 9.0
* Cudnn 7.0
* tensorflow-gpu 1.8
* cv2
* pydicom

## Training
A sample training command:
```bash
$  python main.py --dataset_path '/home/boya/Data/CT-styles' --phase train --gpu 0 --style_A 'LDLungSS0' --style_B 'LDStdSS80' --style_C /home/boya/Data/CT/train/CT_Training_data/dcm --style_E /home/boya/Data/CT/val/STANDARD/dcm --has-paired --from-scratch --num_supervised 1 --num_unsupervised 3 --d_iters 0 --w_gan 0.0
```
some explainations:

* dataset_path: the folder containing paired data, i.e., style_A and style_B.
* style_A: the dataset used for validation
* style_B: target style for supervised learning
* sytle_C: original style for unsupervised learning
* style_D: original styles for supervised learning (change it in model.py)
* style_E: target style for unsupervised learning
* --has-paired or --no-paired: mainly used in testing
* --from-scratch or --continue: whether to train from the scratch or from the last check point
* num_supervised: the number of supervised steps in each training iteration
* num_unsupervised: the number of unsupervised steps in each training iteration
* d_iters: number of discriminator iterations per training iteration.
* w_gan: wgan loss weight (currently it is implemented but not added to the total loss)


## Testing
A sample command:
```bash
$ python main.py --dataset_path '/home/boya/Data/CT/train' --phase test --gpu 0 --style_A 'CT_Training_data/dcm' --out_path 'train_transfer/dcm' --checkpoint_dir 'model1' --no-paired
```

Make sure the checkpoint directory and data folders are all in dataset_path
