# TalkingFaceGeneration-with-Emotion:
<div style="text-align: right"> by <a href="https://github.com/LemonPepperSeasoning">Takahiro Ishiguro</a> and <a href="https://github.com/forestsky1">Alan lin</a></div>


### Introduction
This repository contains the pytorch implementation emotional talking face generation prefesented in paper :[Talking Face Generation](https://github.com/UoA-CARES-Student/EmoFaceGeneration).

### Requirements
-------------
- `Python 3.6` 
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`. Alternatively, instructions for using a docker image is provided [here](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668). Have a look at [this comment](https://github.com/Rudrabha/Wav2Lip/issues/131#issuecomment-725478562) and comment on [the gist](https://gist.github.com/xenogenesi/e62d3d13dadbc164124c830e9c453668) if you encounter any issues. 
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.


### Repository structure
```
.
├── docs                    # Documentation files
│   └── evaluation.md
├── emofacegeneration   # Source files
│   ├── model
│   ├── data
│   ├── utils
│   ├── train.py
│   └── run.py
├── demo                    # demo notebooks
│   └── SPT-Wav2Lip with Google TTS
├── LICENSE
├── README.md
└── requirement.txt 

```
### Sample results


### Documentation
- [Intrustruction README](./emofacegeneration/)
- [Evalutation README](./docs/evaluation/)


### License and Citation
EmoFaceGeneration is released under the [Apache 2.0 license](LICENSE).

### Acknowledgements
Part of the code was adapted from the following projects:
- [to be replaced with link to repo]
