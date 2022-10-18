

## SPT - Wav2Lip model
Adopting the [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) model, the addition of Spatial transformer network was added to increase the quality of lip synchornisation.

[Demo Colab notebook](https://colab.research.google.com/drive/1cvd_ZUBClHlsEx-9szI_zTqqKyHhQMzb?authuser=1#scrollTo=ryz7w34vUAOE)

### Setting up the dataset
Our model was trained using [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) dataset. 

##### Preprocess the dataset for fast training
The dataset is recommended to be preprocessed first using our script.  
```bash
python preprocess.py --data_root data_root/main --preprocessed_root lrs2_preprocessed/
```

To train the model, `filelists/train.txt` & `filelists/val.txt` is needed, where each line in the `.txt` file represents a folder containing frames and audio of the video.


Example.
filelists/train.txt
```
dataset/video1
dataset/video2
...
dataset/video10
```

Where Dataset folder structure
```
dataset
├── video1
│   ├── 0.jpg 
│   ├── 1.jpg 
│   ├── ...
│   ├── 120.jpg
│   └── audio.wav
├── video2
│   ├── 0.jpg 
│   ├── 1.jpg 
│   ├── ...
│   ├── 110.jpg
│   └── audio.wav
...
```

### Training script
You can either train the model without the additional visual quality disriminator (< 1 day of training) or use the discriminator (~2 days). For the former, run: 
```bash
python3 train.py --data_root dataset/ --checkpoint_dir  <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint> --disc_checkpoint_path <path_to_perceptual_disc_checkpoint>
```


### Running the model (inference)
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result will be saved to `results/result_voice.mp4`


### Evalutation
[Evalutation README](../docs/evaluation/)



## DeepFaceLab
To use deepfacelab, simply follow the instruction provided below.
- [Main guide](https://mrdeepfakes.com/forums/thread-guide-deepfacelab-2-0-guide)
    (Specifically, I used scripts found in [DeepFaceLab_Linux](https://github.com/nagadit/DeepFaceLab_Linux) repository.)
- [Google Colab notebook](https://colab.research.google.com/github/chervonij/DFL-Colab/blob/master/DFL_Colab.ipynb)

##StyleGAN emotion modifier
First to download the required dependencies, run these commands in the following order:
<br/>
pip install tensorflow-gpu
<br/>
pip install tensorflow[horovod] 
<br/>
pip install tensorboard==2.10.0
<br/>
This is due to tensorflow 1 now being depcrated, and for many environments, no longer avaiable or downloadable 
https://stackoverflow.com/questions/73215696/did-colab-suspend-tensorflow-1-x
In addition, it is no longer aviable on PyPi.

Then, please download the following weights:
https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2
https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ
Place these weights into "stylegan/weights"
Into TalkingfaceGen
To modifiy images, first encode the images into lantent space, provide the directory of the target photo(s)
```bash
python image_encoder --src_dir [PATH_TO_DIR]
```

After the images are encoded, use "emotion_inference.py" to alter your image's emotion, the --coeff argument takes in numberical inputs, where positive numbers point towards happy, whilst negative numbers point towards sad. the --face_name argument takes in the target photo name that you want the emotion to be altered for. Only pass in the image name with no extension or path.
For example, I want to alter an image I encoded called "demo1.jpg". I also want to change the face so a smile the command would be:
```bash
python emotion_inference.py --face_name demo1 --coeff 0.7
```


Thanks to the following repository for their implementation:
https://github.com/pbaylies/stylegan-encoder
