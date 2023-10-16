import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import argparse, os, fnmatch, shutil
from tqdm import tqdm
import cv2

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i1", "--in-file_dis", type=str, help="input video file_dis", default='emo/DIS_generated.mp4_.mp4')
parser.add_argument("-i2", "--in-file_hap", type=str, help="input video file_hap", default='emo/HAP_generated.mp4_.mp4')
parser.add_argument("-i3", "--in-file_neu", type=str, help="input video file_neu", default='emo/NEU_generated.mp4_.mp4')
parser.add_argument("-emo", "--emo_fold", type=str, help="generated video folder", default='emo')
parser.add_argument("-o", "--out-fold", type=str, help="output folder", default='emo_result')
args = parser.parse_args()

in_file_dis = args.in_file_dis
in_file_hap = args.in_file_hap
in_file_neu = args.in_file_neu
out_fold = args.out_fold
emo_fold = args.emo_fold

if not os.path.exists(out_fold):
    os.makedirs(out_fold)
else:
    shutil.rmtree(out_fold)
    os.mkdir(out_fold)

cap1 = cv2.VideoCapture(in_file_dis)
cap2 = cv2.VideoCapture(in_file_hap)
cap3 = cv2.VideoCapture(in_file_neu)

length1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))


for frm_cnt in tqdm(range(length1)):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    # if frame1 is None:
    #     break
    
    if frm_cnt == 11:
        cv2.imwrite(os.path.join(out_fold, 'dis'+'.png'), frame1)
        cv2.imwrite(os.path.join(out_fold, 'hap'+'.png'), frame2)
        cv2.imwrite(os.path.join(out_fold, 'neu'+'.png'), frame3)
        

if os.path.exists(emo_fold):
    shutil.rmtree(emo_fold)

