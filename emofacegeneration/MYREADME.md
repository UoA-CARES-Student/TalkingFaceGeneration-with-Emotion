
Not version XXX in requirements.txt
pip install opencv-python==4.5.2.52



# PREPROCESS THE DATA
```
### Default 
python3 preprocess.py --data_root dataset2/test --preprocessed_root voxceleb_preprocessed_5/ --batch_size 8


### Using Hazel & Harddrive
python3 preprocess.py --data_root /home/myuser1/workspace/hard_drive_mount_location/p4p-g24-2022/VoxCeleb2-Dataset/raw_dataset/test --preprocessed_root /home/myuser1/workspace/hard_drive_mount_location/p4p-g24-2022/VoxCeleb2-Dataset/preprocessed_dataset/Wav2Lip_Preprocess --batch_size 8


### Using Hazel & NAS
python3 preprocess.py --data_root /mnt/DataSets/tish386/test --preprocessed_root /mnt/DataSets/tish386/wav2lip_preprocess --batch_size 8


### 2 GPU (crashes hazel)
python3 preprocess.py --data_root /mnt/DataSets/tish386/test --preprocessed_root /mnt/DataSets/tish386/wav2lip_preprocess --batch_size 64 --ngpu 2


### 2 GPU Crashes HAZELL artemis.
Run with 1 GPU.
python3 preprocess_Takahiro.py --data_root /mnt/DataSets/tish386/test_chunk_2 --preprocessed_root /mnt/DataSets/tish386/wav2lip_preprocess_chunk_2

```


# Train expert discriminator
```
python color_syncnet_train.py --data_root voxceleb_preprocessed_5/ --checkpoint_dir <folder_to_save_checkpoints>
```


# Train the model
```
### Default
python wav2lip_train.py --data_root voxceleb_preprocessed_5/ --checkpoint_dir checkpoints --syncnet_checkpoint_path expert/lipsync_expert.pth


python wav2lip_train_Takahiro.py --data_root / --checkpoint_dir checkpoints/tmp --syncnet_checkpoint_path discriminator/lipsync_expert.pth


python hq_wav2lip_train.py --data_root / --checkpoint_dir checkpoint_2 --syncnet_checkpoint_path discriminator/lipsync_expert.pth --disc_checkpoint_path discriminator/visual_quality_disc.pth


### Start from empty discrimnator
python hq_wav2lip_train.py --data_root / --checkpoint_dir checkpoints/attension_v1_jul_12 --syncnet_checkpoint_path discriminator/lipsync_expert.pth --disc_checkpoint_path discriminator/visual_quality_disc.pth
```
