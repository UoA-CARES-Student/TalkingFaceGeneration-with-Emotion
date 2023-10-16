# Head pose implementation stage of TalkingFaceGeneration-with-Emotion 
Head pose implementation stage takes a video the lip synchronisation generator and a pose source file to use as a reference. 

## Installation
Use google colab notebook provided (name) 
OR
use Conda environment (python 3.8)
```bash
1. conda create -n pose python=3.8
2. conda activate pose
```
Clone directory and 
```bash
1. pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
2. conda install ffmpeg
3. pip install -r requirements.txt
```

## Download and extract checkpoints
Download checkpoints from 
google drive: https://drive.google.com/file/d/18Bi06ewhcx-1owlJF3F_J3INlXkQ3oX2/view
And put it in ./checkpoints
 ```bash
tar -zxvf checkpoints.tar.gz
```
Google colab demo already includes downloading and placing checkpoints 

## Inference 
 ```bash
python run_demo.py --s_path ./data/s.mp4 \
 		--d_path ./data/d.mp4 \
		--model_path ./checkpoints/dpe.pt \
		--face pose \
		--output_folder ./res
```