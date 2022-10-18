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


def move_and_show(latent_vector, mask_path, orig_img_path, emo_coeffs, gen_dir):

    # if emo_cond == "happy":
    #USed to take "emo_condition input, that would load different directions, but results weren't great for other pre-trained directions"

    direction = np.load('./stylegan/latent_directions/smile.npy')    
    # if emo_cond == "sad":
    #     print("I loaded sad")
    #     #used to load a "sad" weight, but no longer in use


    for i, coeff in enumerate(emo_coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (new_latent_vector  + coeff*direction)[:8]
        img = np.asarray(generate_image(new_latent_vector, mask_path, orig_img_path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_path = os.path.join(gen_dir,  "coeff_" + str(coeff) + "_" + mask_path.split("/")[-1])
        cv2.imwrite(result_path, img)
        print(f"image written to: {result_path}")
        





def generate_emotion(latent_path, mask_path, orig_img_path, emo_coeff, gen_dir):
    latent_vector = np.load(latent_path)
    move_and_show(latent_vector, mask_path, orig_img_path, emo_coeff, gen_dir)



