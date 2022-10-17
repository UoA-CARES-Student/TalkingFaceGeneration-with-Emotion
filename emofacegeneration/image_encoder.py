import argparse, os
from stylegan.encode_images import encode_image
from stylegan.align_images import align_images

def main():
    parser = argparse.ArgumentParser(description='Generate emotional talking face video!', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_dir', help='source directory for facial image')
    args, other_args = parser.parse_known_args()
    aligned_path = '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/aligned_img/'

    if os.path.exists(args.src_dir):
        align_images(args.src_dir, "dir")
        encode_image(aligned_path)
        print("Images encoded.")
    else:
        print("ensure that path is to the source directory, and that path exists.")








if __name__ == "__main__":
    main()