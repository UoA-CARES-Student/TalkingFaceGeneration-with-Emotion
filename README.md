# Lip synchronisation and head pose implementation branch
This branch contains lip synchronisation and head pose implementation files. There are two google colab notebooks for ease of execution. 

## Pre-processing step
Before executing lip synchronisation from an output image of emotion generator, two processing steps are needed. 
1. Paste-back operation: pasting the cropped emotional image back to original image for better head pose implementation

2. Image to Video: Lip synchronisation generator takes a video as an input, whereas the output of the image generator is an image. Therefore, image to video conversion is needed.

These steps can be achieved by running "preprocessing_pasteback_vid.ipynb".

## Lip synchronisation 
All the files and instructions how to run lip synchronisation are under "lip_synchronisation" folder. 

## Head pose implementation 
All the files and instructions how to run head pose implementation are under "head_pose" folder. 

## Integration of Lip synchronisation and Head pose implementation
Run "lip_headpose.ipynb".

## Evaluation 
Run "lipSync_evaluation.ipynb" to run PSNR, MSE, SSIM and FID metrics. 