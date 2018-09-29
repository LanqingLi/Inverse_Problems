CT Lung Images Standardization
-----------------------------------------------------
The quality of CT images may differ in many ways, such as dose, noise level, texture, contrast, brightness, etc. In this project folder, we try to leverage all these factors, so as to feed our nodule detector with similar data distributions. In this folder, there are three sub-repos: ct_frcnn_detection, one-to-one and multi-to-one.

##Codes

* [ct_frcnn_detection] The codes came directly from the Model Group that uses fast-RCNN to perform detection. For details please refer to the specific readme.
* [one-to-one] This model use CycleGAN and can transfer one specific domain of images to a specific style. The model works fully in an unsupervised way.
* [multi-to-one] This model can transfer images from multiple domians to one specific domain. The images in the target domain must be homogeneous, i.e., they have exactly the same dose and reconstruction kernel. Domain knowledge and labels for the rest images are not required. The models works in supervised, unsupervised and semi-supervised mannars. However, the fully unsupervised part is still under construction.

##Data

We basically have two kinds of datasets: paired and unpaired. For the paired data, CT images with different doses and/or convolution kernals are perfectly alignecd, while each folder in the unpaired dataset comes from a unique patient.

* The raw data of paired images is located in /home/boya/Data/2018_03_26_WuHanTongJi and is rearranged in /home/boya/Data/CT-styles
* The unpaired data is put in /home/boya/Data/CT which was originally for detection model training and testing.
