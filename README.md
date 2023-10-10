# TalkingFaceGeneration-with-Emotion
README file for emotion generator.

## Installation
```bash
pip install -r requirements.txt
```

## Preparation
This emotion generator uses CREMA-D dataset.

1. Download dataset here [dataset](https://github.com/CheyneyComputerScience/CREMA-D) to download dataset

2. Convert Videos to 25 FPS
```bash
python /data_prep/convertFPS.py -i raw_video_folder -o /output_folder
```

3. Prepare data
```bash
python /data_prep/prepare_data.py -i 25_fps_video_folder/ -o /output_folder --mode 1 --nw 1
```


If you want to use pre-trained model download [pretrained_model](https://drive.google.com/drive/folders/10lDoeIq_68FRFvQEXD4LFjU_JhK1q2Xi?usp=sharing) and put into model folder

## Test

```bash
python generate_emotion.py -m ./model/
```

## Train

```bash
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/mde/ --pre_train 1 --disc_emo 1 --lr_emo 1e-4
```

```bash
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/pre_gen/ --lr_g 1e-4
```

```bash
python train.py -i /train_hdf5_folder/ -v /val_hdf5_folder/ -o ../models/tface_emo/ -m ../models/pre_gen/ -mde ../models/mde/ --disc_frame 0.01 --disc_emo 0.001
```
