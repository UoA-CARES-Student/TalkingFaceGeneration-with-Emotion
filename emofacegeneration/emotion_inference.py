import argparse, os
from stylegan.encode_images import encode_image
from stylegan.align_images import align_images
from stylegan.generate_emotion import generate_emotion
import glob
def main():
    parser = argparse.ArgumentParser(description='Generate emotional talking face video!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--face_name', help='name of the face file use to encode image')
    parser.add_argument('--coeff', nargs='+', type = float, default = 0, help='coefficent of sentiment, positive increaseing the happiness, negative toward sadness')
    parser.add_argument('--gen_dir', help='path to directory results will be outputted to. by default this path is: /workspace/alan/EmoFaceGeneration/emofacegeneration/faces/generated_img ', default= '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/generated_img', type = str)

    latent_path = '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/latent_rep/'
    mask_path = '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/masks/'
    aligned_path = '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/aligned_img/'



    args, other_args = parser.parse_known_args()
    if os.path.exists(args.gen_dir):



        print("I globbed" + str(glob.glob(mask_path + args.face_name + "*")))
        latent_img_path = glob.glob(latent_path + args.face_name + "*")
        mask_img_path =  glob.glob(mask_path + args.face_name + "*")
        aligned_img_path =  glob.glob(aligned_path + args.face_name + "*")
        if os.path.exists(latent_img_path[0]):
            generate_emotion(latent_img_path[0], mask_img_path[0],aligned_img_path[0], args.coeff, "happy", args.gen_dir)
        else:
            print("face image doesn't exist, check that the extension name is not included, and it has been encoded into latent space. ")
    else:
        print("please enter a valid generated results dir")







if __name__ == "__main__":
    main()