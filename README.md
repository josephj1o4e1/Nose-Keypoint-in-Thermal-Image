# Nose-Keypoint-on-Thermal-Image
A Keypoint Detection Implementation on Thermal Image  

This is a Pytorch transfer learning implementation of a convolutional neural network.  
We have two models which uses Mobilenet_v2 and Efficient-net-small as our two backbone choices.  
The output should be 9 flattened 2D points. The default data preprocessing is Rescale/Normalize/Tensor.  
We wanted to random crop, but it has the risk that some target keypoints might be cropped out, which ends up incompatible with our 9 point output model. This task remains to be done.  

Here the main file is train_nosedetection.py  
Please specify the dataset root path argument.  
The command line should be like: python train_nosedetection.py --datasetroot "path to dataset"  


The dataset should be constructed as an "Input" folder and "Target" folder, which they are image.png-target.json pairs for supervised learning.  
Our dataset is from the paper: ["A Thermal Infrared Face Database With Facial Landmarks and Emotion Labels"](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/KCZ18h.pdf) by Marcin Kopaczka, Raphael Kolk  





