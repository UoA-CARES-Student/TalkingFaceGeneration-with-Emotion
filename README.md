# Talking Face Generation with Emotion:

<p align='center'>
  <img src='media/monalisa_gif.gif' width='80%'/>
</p>

<div style="text-align: right"> by <a href="https://github.com/LemonPepperSeasoning">Takahiro Ishiguro</a> and <a href="https://github.com/forestsky1">Alan lin</a></div>

### Introduction

This repository contains the pytorch implementation emotional talking face generation prefesented in paper : Emotional Talking Facial Video Generation using Single Image and Sentence.

### Requirements

---

- `Python 3.6`
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`.

### Repository structure

```
.
├── docs                    # Documentation files
│   └── evaluation.md
├── TalkingFaceGeneration   # Source files
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

### Documentation

- [Training & Inference insturction README](./TalkingFaceGeneration/)
- [Evalutation README](./docs/evaluation/)

### Sample results

#### Image to Video

<video src="https://user-images.githubusercontent.com/60628111/198815459-f98b1844-faf4-4dfe-bafe-10e3d8a78d0c.mp4
" width='60%'></video>

#### Video to Video

<video src="https://user-images.githubusercontent.com/60628111/198815427-9745b0ac-b61c-4cf4-b20b-cfc8874fb167.mp4" width='60%'></video>

### License and Citation

EmoFaceGeneration is released under the [Apache 2.0 license](LICENSE).

### Acknowledgements

This code borrows heavily from [Wav2Lip](), [StyleGAN]() and [DeepFaceLab](). We thank the authors for releasing their models and codebase. We also like to thank BBC for allowing us to use thier VoxCeleb2 dataset.
