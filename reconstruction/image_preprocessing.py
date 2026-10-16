import image_color_correction as icc
import image_illumination_correction as iic
from tqdm import tqdm
import os

def image_preprocessing_job(img_dir, out_dir):
    img_list = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    in_files = [os.path.join(img_dir, f) for f in img_list]


    print("Preprocessing images for illumination correction...")
    iic.preprocess_sequence(in_files, out_dir)

    out_files = [os.path.join(out_dir, f) for f in img_list if os.path.exists(os.path.join(out_dir, f))]
    if len(in_files) != len(out_files):
        print("Warning: Some images were not processed for illumination correction. Please check the output directory.")

    print("Preprocessing images for color correction (overwriting existing files)...")
    for img in tqdm(out_files, desc="Color correction", total=len(out_files), unit="image"):
        icc.correct_image(img, img)

if __name__ == "__main__":
    img_dir_list = [r"F:\01-Reconstructions\data\2022\Coral Garden\images\raw\pl811",
                    r"F:\01-Reconstructions\data\2022\Coral Garden\images\raw\pl814"]
    out_dir = r"F:\01-Reconstructions\data\2022\Coral Garden\images\color_illum_corrected"

    for img_dir in img_dir_list:
        image_preprocessing_job(img_dir, out_dir)