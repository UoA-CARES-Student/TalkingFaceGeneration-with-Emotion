import cv2
import numpy as np
import argparse, shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i_origin", "--img1_path", type=str, help="input original image", default='paste_back_result.jpg')
parser.add_argument("-o", "--output_path", type=str, help="output path", default='img2vid.avi')
args = parser.parse_args()

# Path to the image and desired output video
image_path = args.img1_path
output_path = args.output_path


# Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Failed to load image at {image_path}")
    exit()

# Extract image dimensions
height, width, layers = img.shape
size = (width, height)

# Define video writer
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

# Write the image frame 60 times for 4 seconds of video at 15 fps
for _ in range(60):
    out.write(img)

out.release()