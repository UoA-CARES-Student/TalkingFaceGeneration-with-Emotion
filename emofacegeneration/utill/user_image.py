from stylegan.encode_images import encode_image
from stylegan.align_images import align_images



def process_user_image(image_path):
    align_images(image_path, "image")
    encode_image('/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/aligned_img', '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/generated_img', '/workspace/alan/EmoFaceGeneration/emofacegeneration/faces/latent_rep')