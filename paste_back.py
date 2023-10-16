import cv2
import numpy as np
import dlib
import argparse, shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i_origin", "--img1_path", type=str, help="input original image", default='data/image_samples/resize_Anne.jpg')
parser.add_argument("-i_cropped", "--img2_path", type=str, help="input cropped image", default='results/emo_result_1.0/final_results/dis.png')
parser.add_argument("-o", "--output1_path", type=str, help="output path", default='paste_back_result.jpg')
parser.add_argument("-datfile", "--datFile", type=str, help="dilb file path", default='data/shape_predictor_68_face_landmarks.dat')
args = parser.parse_args()

# Image paths
img1_path = args.img1_path
img2_path = args.img2_path
output1_path = args.output1_path
datFile =args.datFile

# Load images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Initialize dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(datFile)

# Get the landmarks
def get_landmarks(image):
    rects = detector(image, 1)
    if len(rects) != 1:
        return None
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

# Align the image based on the eyes position
def align_face(image, landmarks):
    left_eye = np.mean(landmarks[36:42], axis=0).tolist()[0]
    right_eye = np.mean(landmarks[42:48], axis=0).tolist()[0]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned

# Get landmarks for img1 and aligned face for img2
landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)

if landmarks1 is None or landmarks2 is None:
    print("Couldn't detect face landmarks.")
    exit()

aligned_img2 = align_face(img2, landmarks2)

# Extract faces from the images
x1, y1, w1, h1 = cv2.boundingRect(np.array(landmarks1))
x2, y2, w2, h2 = cv2.boundingRect(np.array(landmarks2))

face1 = img1[y1:y1+h1, x1:x1+w1]
face2 = aligned_img2[y2:y2+h2, x2:x2+w2]

# Resize and mask generation
resize_pct = 0.50
face2 = cv2.resize(face2, (int(face2.shape[1] * resize_pct), int(face2.shape[0] * resize_pct)))

def create_soft_mask(face):
    mask = np.ones_like(face) * 255
    h, w, _ = face.shape
    mask[:15, :, :] = 0
    mask[h-15:, :, :] = 0
    mask[:, :15, :] = 0
    mask[:, w-15:, :] = 0
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    return mask

mask1 = create_soft_mask(face1)
face2_resized = cv2.resize(face2, (w1, h1))
mask2_resized = cv2.resize(create_soft_mask(face2), (w1, h1))
center1 = (x1 + w1 // 2, y1 + h1 // 2)

# Seamless clone
img1_face_swapped = cv2.seamlessClone(face2_resized, img1, mask2_resized, center1, cv2.NORMAL_CLONE)

# Save the results
cv2.imwrite(output1_path, img1_face_swapped)
