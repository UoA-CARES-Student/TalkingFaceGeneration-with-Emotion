

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
hq_wav2lip_train.py --data_root dataset/ --checkpoint_dir  <folder_to_save_checkpoints> --syncnet_checkpoint_path <path_to_expert_disc_checkpoint> --disc_checkpoint_path <path_to_perceptual_disc_checkpoint>
```


### Running the model (inference)
```bash
python inference.py --checkpoint_path <ckpt> --face <video.mp4> --audio <an-audio-source> 
```
The result will be saved to `results/result_voice.mp4`


### Evalutation
[Evalutation README](../docs/evaluation/)