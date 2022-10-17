import os
import argparse
import pickle
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import stylegan.config
from .encoder.generator_model import Generator
import cv2
import matplotlib.pyplot as plt
# matplotlib inline


HAPPY_DIR = '/workspace/alan/image_expr_mod/stylegan-encoder/ffhq_dataset/latent_directions/smile.npy'
SAD_DIR = '/workspace/alan/image_expr_mod/stylegan-encoder/latent_directions/emotion_angry.npy'
tflib.init_tf()
with open("/workspace/alan/EmoFaceGeneration/emofacegeneration/stylegan/weights/karras2019stylegan-ffhq-1024x1024.pkl", "rb") as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)

# generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
generator = Generator(Gs_network, 1, clipping_threshold=2, tiled_dlatent=False, model_res=1024, randomize_noise=False)





def generate_image(latent_vector, mask_path, orig_img_path):
    latent_vector = latent_vector.reshape((1, 18, 512))
    # mask = PIL.Image.open(mask_path)

    orig_img = PIL.Image.open(orig_img_path)
    width, height = orig_img.size
    print(f"orig size is {orig_img.size}")

    imask = PIL.Image.open(mask_path).convert('L').resize((width, height))
    imask = imask.filter(ImageFilter.GaussianBlur(8))
    mask = np.array(imask)/255
    mask = np.expand_dims(mask,axis=-1)
    print(f"mask size is {mask.size}")

    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    print(f"img_array size is {img_array.size}")

    img_array = mask*np.array(img_array) + (1.0-mask)*np.array(orig_img)
    img_array = img_array.astype(np.uint8)
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))


def move_and_show(latent_vector, mask_path, orig_img_path, emo_coeffs, emo_cond, gen_dir):

    if emo_cond == "happy":

        direction = np.load('/workspace/alan/EmoFaceGeneration/emofacegeneration/stylegan/latent_directions/smile.npy')    
    if emo_cond == "sad":
        print("I loaded sad")


    for i, coeff in enumerate(emo_coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (new_latent_vector  + coeff*direction)[:8]
        img = np.asarray(generate_image(new_latent_vector, mask_path, orig_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_path = os.path.join(gen_dir,  "coeff_" + str(coeff) + "_" + mask_path.split("/")[-1])
        cv2.imwrite(result_path, img)
        print(f"image written to: {result_path}")
        





def generate_emotion(latent_path, mask_path, orig_img_path, emo_coeff, emo_cond, gen_dir):
    latent_vector = np.load(latent_path)
    move_and_show(latent_vector, mask_path, orig_img_path, emo_coeff, emo_cond, gen_dir)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Given latent representation, emotion direction, mask and original image, change the facial expression sentiment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('--latent_path', help='path to latent representation')
#     parser.add_argument('--mask_path', help='path to image mask')
#     parser.add_argument('--img_path', help='path to original aligned image')
#     parser.add_argument('--emo_cond', type = str, help='emotion condition')
#     parser.add_argument('--coeff', nargs='+', type = float, default = 0, help='coefficent of sentiment, positive increaseing the happiness, negative toward sadness')

#     args, other_args = parser.parse_known_args() 

#     # print(type(args.coeff))
#     # generate_emotion(args.latent_path, args.mask_path, args.img_path, args.coeff, args.emo_cond)


