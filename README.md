# TalkingFaceGeneration-with-Emotion:
<div style="text-align: right"> by <a href="https://github.com/LemonPepperSeasoning">Takahiro Ishiguro</a> and <a href="https://github.com/forestsky1">Alan lin</a></div>


### Introduction
This repository contains the pytorch implementation emotional talking face generation prefesented in paper :[Talking Face Generation](https://github.com/UoA-CARES-Student/TalkingFaceGeneration-with-Emotion).

### Requirements
-------------
- `Python 3.6` 
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`.



### Repository structure
```
.
├── docs                    # Documentation files
│   └── evaluation.md       
├── emofacegeneration       # Source files
│   ├── model
│   ├── data
│   ├── DeepFaceLab
│   ├── StyleGAN
│   ├── dnnlib              # Helper functions for StyleGAN generator
│   ├── filelists           # describe the directory of train & validation datasets
│   ├── face_detection      # module used to crop face image
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
├── demo                    # demo notebooks
│   ├── DeepFaceLab.ipynb   # notebook for deepfacelab
│   └── SPT-Wav2Lip.ipynb   # notebook for SPT-Wav2Lip with Google TTS
├── LICENSE
├── README.md
└── requirement.txt 

```
### Sample results


### Documentation
- [Training & Inference insturction README](./TalkingFaceGeneration/)
- [Evalutation README](./docs/evaluation/)


### License and Citation
EmoFaceGeneration is released under the [Apache 2.0 license](LICENSE).

### Acknowledgements
Part of the code was adapted from the following projects:
- [to be replaced with link to repo]
