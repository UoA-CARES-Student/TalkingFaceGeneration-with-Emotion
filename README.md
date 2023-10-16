# TalkingFaceGeneration-with-Emotion
<div style="display: flex; justify-content: center;">
  <img src='media/anne.gif' style="width: 3000%;"/>
  <img src='media/scalett.gif' style="width: 3000%;"/>
</div>

### Introduction
This repository contains the colab files to test overall system with implementation emotional talking face generation : Emotional Talking Facial Video Generation using Single Image and Sentence.
### Test overall system
There are two ipynb files to test step1-step5 and generate emotional talking face video.
```
1.Emotion&preprocessing.ipynb
```
```
2.TTS&Lip_sync&head_pose.ipynb
```

Follow the instruction in ipynb files

### Step-by-step system
1.	Emotional Face Image Generation: A target facial image and emotion are input to produce an emotional image. 

2.	Talking Face Video Generation with Lip Synchronisation: The emotional image is combined with generated audio (from the target dialogue converted via text-to-speech) to produce a video with synced lip movements. 

3.	Paste-back Operation: This stage pastes the cropped emotional image back to its original image.  

4.	Head Pose Implementation: Implementing head movements to make the video more dynamic and realistic. 

5.	Visual Quality Improvement: Enhancements are made to the video's visual quality, resulting
   
### Repository structure

```
├── TalkingFaceGeneration   ├── 2023_all_together ├── CodeFormer
                                                  ├── Emotion
                                                  ├── lip_synchronisation
                                                  ├── head_pose
                                                  ├── 1.Emotion&preprocessing.ipynb
                                                  ├── 2.TTS&Lip_sync&head_pose.ipynb



                            ├── 2023_emotion ├── All files for emotion generator 
                                             
                            ├── 2023_lipsync_headpose ├── All files for Lip Synchronisation and Head Pose


```
