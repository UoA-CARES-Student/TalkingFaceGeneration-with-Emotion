import sys


if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

from os import listdir, path

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import argparse, os, cv2, traceback, subprocess
from tqdm import tqdm
from glob import glob
import audio
from hparams import hparams as hp

import face_detection

parser = argparse.ArgumentParser()

parser.add_argument('--ngpu', help='Number of GPUs across which to run in parallel', default=1, type=int)
parser.add_argument('--batch_size', help='Single GPU Face detection batch size', default=32, type=int)
parser.add_argument("--data_root", help="Root folder of the LRS2 dataset", required=True)
parser.add_argument("--preprocessed_root", help="Root folder of the preprocessed dataset", required=True)


args = parser.parse_args()

print(f"NUMBER OF GPUS : {args.ngpu}")



# fa_0 =face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(0))

fa = [0]*2
for id in range(args.ngpu):
	fa[id] = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda:{}'.format(id))
print(len(fa))
print(fa[0])
# fa = [face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
# 									device='cuda:{}'.format(id)) for id in range(args.ngpu)]


template = 'ffmpeg -loglevel panic -y -i {} -strict -2 {}'
# template2 = 'ffmpeg -hide_banner -loglevel panic -threads 1 -y -i {} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}'


def process_video_file(vfile, args, gpu_id):
	video_stream = cv2.VideoCapture(vfile)
	
	frames = []
	while 1:
		still_reading, frame = video_stream.read()
		if not still_reading:
			video_stream.release()
			break
		frames.append(frame)
	
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	batches = [frames[i:i + args.batch_size] for i in range(0, len(frames), args.batch_size)]
	i = -1
	for fb in batches:
		preds = fa[gpu_id].get_detections_for_batch(np.asarray(fb))
		# preds = fa_0.get_detections_for_batch(np.asarray(fb))
		# print(f"len of fb: {len(fb)}")
		
		for j, f in enumerate(preds):
			i += 1
			x1, y1, x2, y2 = f
			cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])

def process_video_file_2(vfile, args, gpu_id):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	# Opens the Video file
	cap= cv2.VideoCapture(vfile)
	i=0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)),frame)
		i+=1

	cap.release()
	cv2.destroyAllWindows()



def process_audio_file(vfile, args):
	vidname = os.path.basename(vfile).split('.')[0]
	dirname = vfile.split('/')[-2]

	fulldir = path.join(args.preprocessed_root, dirname, vidname)
	os.makedirs(fulldir, exist_ok=True)

	wavpath = path.join(fulldir, 'audio.wav')

	command = template.format(vfile, wavpath)
	subprocess.call(command, shell=True)

	
def mp_handler(job):
	vfile, args, gpu_id = job
	try:
		process_video_file(vfile, args, gpu_id)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def main(args):

	print("====== STARTING PREPROCESSING ======")

	print('Started processing for {} with {} GPUs'.format(args.data_root, args.ngpu))
	print(glob(args.data_root))

	print( path.join(args.data_root, '*/*.mp4'))
	filelist = glob(path.join(args.data_root, '*/*.mp4'), recursive=True)
	# print(f"NUMBER OF Video-file to be processed : {len(filelist)}")

	jobs = [(vfile, args, i%args.ngpu) for i, vfile in enumerate(filelist)]
	p = ThreadPoolExecutor(args.ngpu)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

	print('Dumping audios...')

	for vfile in tqdm(filelist):
		print(vfile)
		try:
			process_audio_file(vfile, args)
		except KeyboardInterrupt:
			exit(0)
		except:
			traceback.print_exc()
			continue

if __name__ == '__main__':
	main(args)
