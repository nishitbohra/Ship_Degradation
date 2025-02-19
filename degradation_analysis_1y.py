import cv2
import numpy as np
import os
import pandas as pd

np_folder = "E:/MS data/data/np"
y1_folder = "E:/MS data/data/1y"
output_csv = "E:/MS data/degradation_1y.csv"

np_images = sorted(os.listdir(np_folder))
y1_images = sorted(os.listdir(y1_folder))
degradation_data = []


for np_img, y1_img in zip(np_images, y1_images):
    np_path = os.path.join(np_folder, np_img)
    y1_path = os.path.join(y1_folder, y1_img)

    np_image = cv2.imread(np_path, cv2.IMREAD_GRAYSCALE)
    y1_image = cv2.imread(y1_path, cv2.IMREAD_GRAYSCALE)

    if np_image is None:
        print(f"Error: Could not load image at {np_path}")
        continue  

    if y1_image is None:
        print(f"Error: Could not load image at {y1_path}")
        continue  

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    np_enhanced = clahe.apply(np_image)
    y1_enhanced = clahe.apply(y1_image)


    np_thresh = cv2.adaptiveThreshold(np_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    y1_thresh = cv2.adaptiveThreshold(y1_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)


    kernel = np.ones((3, 3), np.uint8)
    np_thresh = cv2.morphologyEx(np_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    y1_thresh = cv2.morphologyEx(y1_thresh, cv2.MORPH_OPEN, kernel, iterations=2)


    diff_mask = cv2.absdiff(np_thresh, y1_thresh)  
    change_pixels = np.count_nonzero(diff_mask)
    total_pixels = np_image.shape[0] * np_image.shape[1]

    degradation_percentage = (change_pixels / total_pixels) * 100
    degradation_data.append([np_img, degradation_percentage])

    print(f"{np_img} -> Degradation: {degradation_percentage:.2f}%")

df = pd.DataFrame(degradation_data, columns=["Image", "Degradation_Percentage"])
df.to_csv(output_csv, index=False)

print(f"Saved degradation report to {output_csv}")
