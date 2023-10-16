# TalkingFaceGeneration-with-Emotion                                         
Emotion generator generates emotional images(Happy, Disgusted and Neutral) from any single input image(recommend to use no background portrait).

## Installation
```bash
pip install -r requirements.txt
```

## Test with pre-trained model
Use Emotion_generator.ipynb to run in google colab and test

OR follow the next steps

1. If you want to use pre-trained model download [pretrained_model](https://drive.google.com/drive/folders/10lDoeIq_68FRFvQEXD4LFjU_JhK1q2Xi?usp=sharing) and put into model folder or
 ```bash
!gdown https://drive.google.com/uc?id=1xiKfABPX0heyTKvrTBnWTmix2e7KB5rJ&export=download
!unzip model.zip
```
3. Locate to CodeFormer
4. Download the facelib and dlib pretrained models to the weights/facelib folder
 ```bash
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py dlib (only for dlib face detector)
```
4. Download the CodeFormer pretrained models to the weights/CodeFormer folder.
 ```bash
python scripts/download_pretrained_models.py CodeFormer
```
5. Run
```bash
python basicsr/setup.py develop
```
7. Back to main directory and run(default image is Anne)
 ```bash
python generate_emotion.py -im ./input image path -m ./model/
```

You will get emotional images in result folder.

## Preparation for training
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
