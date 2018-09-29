One-to-One CT imaging style transfer
-----------------------------------------------------

This repository provides a Tensorflow implementation of CycleGAN with Wasserstein loss. With images with the same style arranged in the same folder, this model is able to transfer the images in one folder to the other. The generator contains two Unets while the discriminator is cnn+fc. The requirements and detailed usage are listed below.

## Requirements
* Python 2.7
* Cuda 9.0
* Cudnn 7.0
* tensorflow-gpu 1.8
* cv2
* pydicom

## Training
A sample training command can be
```bash
$ python main.py --dataset_path '/home/boya/Data/CT/val' --phase train --gpu 0 --style_A 'guangzhou_yifuyi/dcm' --style_B 'STANDARD/dcm'
```
which tries to learn the mapping from 'guangzhou_yifuyi' to 'STANDARD' so that we can make the 'guangzhou_yifuyi' images look like 'STANDARD' images. Make sure that folders for style_A and style_B are located in dataset_path

## Testing
Sample command:
```bash
$ python main.py --dataset_path '/home/boya/Data/CT/val' --phase test --gpu 0 --style_A 'guangzhou_yifuyi/dcm' --style_B 'STANDARD/dcm' --out_path 'guangzhou_yifuyi_to_STANDARD/dcm'
```

## Tuning parameters and improving the model
The parameters that might be changed are listed in model.py. Some parameters are associated with the data, like if the data is paired, what window to use, etc, while some parameters are associated with the model, like loss function, training algorithm or whether to use learning rate decay. Also we encourage to change the generator or discriminator structures in modules.py. Please used the paired data to tune the model, so that you are able to compare the performances of different models.
