import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
from PIL import ImageFilter
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import stylegan.config
from .encoder.generator_model import Generator
from .encoder.perceptual_model import PerceptualModel, load_images
#from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def encode_image(img_dir):


    encoder_dict = {
        "src_dir": img_dir,
        "generated_images_dir": '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/generated_img',
        "dlatent_dir": '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/latent_rep',
        "data_dir": 'data',
        "model_res": 1024,
        "batch_size": 1,
        "optimizer": 'ggt'

    }

    # Masking params
    mask_dict = {
         "mask_dir": '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/masks',
         "face_mask": True,
         "load_mask": False,
         "use_grabcut": True,
         "scale_mask": 1.4,
         "composite_mask": True,
         "composite_blur": 8
    }



    percep_dict = {
        "image_size": 256,
        "resnet_image_size": 256,
        "lr": 0.25,
        "decay_rate": 0.9,
        "iterations": 100,
        "decay_steps": 4,
        "early_stopping": True,
        "early_stopping_threshold": 0.5,
        "early_stopping_patience": 10,
        "load_effnet": 'data/fintuned_effnet.h5',
        "load_resnet": 'data/finetuned_resnet.h5',
        "use_preprocess_input": True,
        "use_best_loss": True,
        "average_best_loss": 0.25,
        "sharpen_input": True,
        "use_vgg_loss": 0.4,
        "use_vgg_layer": 9,
        "use_pixel_loss": 1.5,
        "use_mssim_loss": 200,
        "use_lpips_loss": 100,
        "use_l1_penalty": 0.5,
        "use_discriminator_loss": 0.5,
        "use_adaptive_loss": False
    }
    # Generator params
    gen_dict = {
        "randomize_noise": False,
        "tile_dlatents": False,
        "clipping_threshold": 2.0
    }


    # args, other_args = parser.parse_known_args()

    percep_dict["decay_steps"] *= 0.01 * percep_dict["iterations"] # Calculate steps as a percent of total iterations

    ref_images = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % img_dir)

    os.makedirs(encoder_dict["data_dir"], exist_ok=True)
    os.makedirs(mask_dict["mask_dir"], exist_ok=True)
    os.makedirs(encoder_dict["generated_images_dir"], exist_ok=True)
    os.makedirs(encoder_dict["dlatent_dir"], exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    with open("/workspace/alan/image_expr_mod/stylegan-encoder/weights/karras2019stylegan-ffhq-1024x1024.pkl", "rb") as f:
        print(f"f is: {f}")
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(Gs_network, encoder_dict["batch_size"], clipping_threshold=gen_dict["clipping_threshold"], tiled_dlatent=gen_dict["tile_dlatents"], model_res=encoder_dict["model_res"], randomize_noise=gen_dict["randomize_noise"])

    perc_model = None
    if (percep_dict["use_lpips_loss"] > 0.00000001):
        # with dnnlib.util.open_url('https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2', cache_dir=config.cache_dir) as f:
        with open("/workspace/alan/EmoFaceGeneration/emofacegeneration/stylegan/weights/vgg16_zhang_perceptual.pkl", "rb") as f:
            perc_model =  pickle.load(f)
        print(f"batch size is " + str(encoder_dict["batch_size"]))
    perceptual_model = PerceptualModel(percep_dict, mask_dict,  batch_size=encoder_dict["batch_size"], perc_model=perc_model)
    perceptual_model.build_perceptual_model(generator, discriminator_network)

    ff_model = None

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch in tqdm(split_to_batches(ref_images, encoder_dict["batch_size"]), total=len(ref_images)//encoder_dict["batch_size"]):
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]

        perceptual_model.set_reference_images(images_batch)
        dlatents = None
        if (ff_model is None):
            if os.path.exists(percep_dict["load_resnet"]):
                from keras.applications.resnet50 import preprocess_input
                print("Loading ResNet Model:")
                ff_model = load_model(percep_dict["load_resnet"])
        if (ff_model is None):
            if os.path.exists(percep_dict["load_effnet"]):
                import efficientnet
                from efficientnet import preprocess_input
                print("Loading EfficientNet Model:")
                ff_model = load_model(percep["load_effnet"])
        if (ff_model is not None): # predict initial dlatents with ResNet model
            if (percep_dict["use_preprocess_input"]):
                dlatents = ff_model.predict(preprocess_input(load_images(images_batch,image_size=percep_dict["resnet_image_size"])))
            else:
                dlatents = ff_model.predict(load_images(images_batch,image_size=percep_dict["resnet_image_size"]))
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=percep_dict["iterations"], use_optimizer=encoder_dict["optimizer"])
        pbar = tqdm(op, leave=False, total=percep_dict["iterations"])
        vid_count = 0
        best_loss = None
        best_dlatent = None
        avg_loss_count = 0
        if percep_dict["early_stopping"]:
            avg_loss = prev_loss = None
        for loss_dict in pbar:
            if percep_dict["early_stopping"]: # early stopping feature
                if prev_loss is not None:
                    if avg_loss is not None:
                        avg_loss = 0.5 * avg_loss + (prev_loss - loss_dict["loss"])
                        if avg_loss < percep_dict["early_stopping_threshold"]: # count while under threshold; else reset
                            avg_loss_count += 1
                        else:
                            avg_loss_count = 0
                        if avg_loss_count > percep_dict["early_stopping_patience"]: # stop once threshold is reached
                            print("")
                            break
                    else:
                        avg_loss = prev_loss - loss_dict["loss"]
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                if best_dlatent is None or percep_dict["average_best_loss"] <= 0.00000001:
                    best_dlatent = generator.get_dlatents()
                else:
                    best_dlatent = 0.25 * best_dlatent + 0.75 * generator.get_dlatents()
                if percep_dict["use_best_loss"]:
                    generator.set_dlatents(best_dlatent)
                best_loss = loss_dict["loss"]
            generator.stochastic_clip_dlatents()
            prev_loss = loss_dict["loss"]
        if not percep_dict["use_best_loss"]:
            best_loss = prev_loss
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        # Generate images from found dlatents and save them
        if percep_dict["use_best_loss"]:
            generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_path, img_name in zip(generated_images, generated_dlatents, images_batch, names):
            mask_img = None
            if mask_dict["composite_mask"] and (mask_dict["load_mask"] or mask_dict["face_mask"]):
                _, im_name = os.path.split(img_path)
                mask_img = os.path.join(mask_dict["mask_dir"], f'{im_name}')
            if mask_dict["composite_mask"] and mask_img is not None and os.path.isfile(mask_img):
                orig_img = PIL.Image.open(img_path).convert('RGB')
                width, height = orig_img.size
                print(f"img_path is {img_path}")
                print(f"orig_img size is {orig_img.size}")
                imask = PIL.Image.open(mask_img).convert('L').resize((width, height))
                imask = imask.filter(ImageFilter.GaussianBlur(mask_dict["composite_blur"]))
                mask = np.array(imask)/255
                mask = np.expand_dims(mask,axis=-1)
                img_array = mask*np.array(img_array) + (1.0-mask)*np.array(orig_img)
                img_array = img_array.astype(np.uint8)
                #img_array = np.where(mask, np.array(img_array), orig_img)
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(encoder_dict["generated_images_dir"], f'{img_name}.png'), 'PNG')
            np.save(os.path.join(encoder_dict["dlatent_dir"], f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()


if __name__ == "__main__":
    encode_image("/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/aligned_img")
