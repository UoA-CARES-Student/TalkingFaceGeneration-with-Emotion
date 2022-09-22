import os
import sys
import bz2
import argparse
from tensorflow.keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import multiprocessing

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

def align_images(path, cond):
    print(f"condition is {cond}")
    condition = lower(cond)
    print(f"image dir is: {image_dir}")
    aligned_dir = "/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/masks/raw_img/aligned_img"
    output_size = 1024
    x_scale = 1
    y_scale = 1
    em_scale = 0.1
    use_alpha = False
    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))


    landmarks_detector = LandmarksDetector(landmarks_model_path)
    if cond == "dir":
        image_dir = path
        for img_name in os.listdir(image_dir):
            print('Aligning %s ...' % img_name)
            try:
                raw_img_path = os.path.join(image_dir, img_name)
                fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
                if os.path.isfile(fn):
                    continue
                print('Getting landmarks...')
                for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                    try:
                        print('Starting face alignment...')
                        face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                        aligned_face_path = os.path.join(aligned_dir, face_img_name)
                        image_align(raw_img_path, aligned_face_path, face_landmarks, output_size, x_scale, y_scale, em_scale, use_alpha)
                        print('Wrote result %s' % aligned_face_path)
                    except:
                        print("Exception in face alignment!")
            except:
                print("Exception in landmark detection!")
    elif cond == "image":
        try:
                raw_img_path = path
                img_name = raw_img_path.split("/")
                fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name[-1])[0], 1)
                if os.path.isfile(fn):
                    print('Getting landmarks...')
                    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                        try:
                            print('Starting face alignment...')
                            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                            aligned_face_path = os.path.join(aligned_dir, face_img_name)
                            image_align(raw_img_path, aligned_face_path, face_landmarks, output_size, x_scale, y_scale, em_scale, use_alpha)
                            print('Wrote result %s' % aligned_face_path)
                        except:
                            print("Exception in face alignment!")
                else:
                     raise Exception("image does not exist!")
        except:
            print("Exception in landmark detection!")
