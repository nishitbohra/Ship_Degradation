import cv2
import numpy as np
import os
import pandas as pd
from skimage.metrics import structural_similarity as ssim

base_folder = "E:/MS data/data"
stages = ["1yB", "5Y", "5Yb"]  
np_folder = os.path.join(base_folder, "np")
output_csv = os.path.join(base_folder, "degradation_all.csv")

df_existing = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.DataFrame()

def is_image_file(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg"))

np_images = sorted([f for f in os.listdir(np_folder) if is_image_file(f)])

degradation_data = []

for stage in stages:
    stage_folder = os.path.join(base_folder, stage)
    stage_images = sorted([f for f in os.listdir(stage_folder) if is_image_file(f)])

    comparison_folder = os.path.join(base_folder, f"comparison_{stage}")
    os.makedirs(comparison_folder, exist_ok=True)

    for np_img, stage_img in zip(np_images, stage_images):
        np_path = os.path.join(np_folder, np_img)
        stage_path = os.path.join(stage_folder, stage_img)

        np_image = cv2.imread(np_path, cv2.IMREAD_GRAYSCALE)
        stage_image = cv2.imread(stage_path, cv2.IMREAD_GRAYSCALE)

        if np_image is None or stage_image is None:
            print(f"Error: Could not load image {np_img} or {stage_img}. Skipping...")
            continue

        if np_image.shape != stage_image.shape:
            stage_image = cv2.resize(stage_image, (np_image.shape[1], np_image.shape[0]))

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  
        np_enhanced = clahe.apply(np_image)
        stage_enhanced = clahe.apply(stage_image)

        np_thresh = cv2.adaptiveThreshold(np_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7)
        stage_thresh = cv2.adaptiveThreshold(stage_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 7)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        np_thresh = cv2.morphologyEx(np_thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        stage_thresh = cv2.morphologyEx(stage_thresh, cv2.MORPH_OPEN, kernel, iterations=3)

        diff_mask = cv2.absdiff(np_thresh, stage_thresh)
        change_pixels = np.count_nonzero(diff_mask)
        total_pixels = np_image.shape[0] * np_image.shape[1]
        degradation_percentage = (change_pixels / total_pixels) * 100

        ssim_value, ssim_map = ssim(np_image, stage_image, full=True)
        ssim_map = np.uint8((ssim_map + 1) * 127.5)  

        comparison_image_path = os.path.join(comparison_folder, f"comparison_{np_img}_{stage}.png")
        ssim_image_path = os.path.join(comparison_folder, f"ssim_map_{np_img}_{stage}.png")

        np_overlay = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)
        stage_overlay = cv2.cvtColor(stage_image, cv2.COLOR_GRAY2BGR)

        np_overlay[diff_mask == 255] = [0, 255, 0]  
        stage_overlay[diff_mask == 255] = [0, 0, 255]  

        combined = np.hstack((np_overlay, stage_overlay))
        cv2.imwrite(comparison_image_path, combined)

        cv2.imwrite(ssim_image_path, ssim_map)

        degradation_data.append([np_img, stage, degradation_percentage, ssim_value, comparison_image_path, ssim_image_path])
        print(f"{np_img} ({stage}) -> Degradation: {degradation_percentage:.2f}% | SSIM: {ssim_value:.4f} | Saved: {comparison_image_path}")

df_new = pd.DataFrame(degradation_data, columns=["Image", "Stage", "Degradation_Percentage", "SSIM", "Comparison_Image_Path", "SSIM_Image_Path"])
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv(output_csv, index=False)

print(f"Updated full degradation report saved to {output_csv}")
