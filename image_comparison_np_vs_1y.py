import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

np_folder = "E:/MS data/data/np"
y1_folder = "E:/MS data/data/1y"
output_folder = "E:/MS data/data/comparison_1y"
os.makedirs(output_folder, exist_ok=True)

np_images = sorted(os.listdir(np_folder))
y1_images = sorted(os.listdir(y1_folder))

for np_img, y1_img in zip(np_images, y1_images):
    np_path = os.path.join(np_folder, np_img)
    y1_path = os.path.join(y1_folder, y1_img)

    np_image = cv2.imread(np_path)
    y1_image = cv2.imread(y1_path)

    np_gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    y1_gray = cv2.cvtColor(y1_image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    np_enhanced = clahe.apply(np_gray)
    y1_enhanced = clahe.apply(y1_gray)

    np_thresh = cv2.adaptiveThreshold(
        np_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    y1_thresh = cv2.adaptiveThreshold(
        y1_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )

    kernel = np.ones((3, 3), np.uint8)
    np_thresh = cv2.morphologyEx(np_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    y1_thresh = cv2.morphologyEx(y1_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    np_contours, _ = cv2.findContours(np_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y1_contours, _ = cv2.findContours(y1_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    np_filtered = [cnt for cnt in np_contours if 30 < cv2.contourArea(cnt) < 1000]
    y1_filtered = [cnt for cnt in y1_contours if 30 < cv2.contourArea(cnt) < 1000]

    np_mask = np.zeros_like(np_gray)
    y1_mask = np.zeros_like(y1_gray)
    cv2.drawContours(np_mask, np_filtered, -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(y1_mask, y1_filtered, -1, 255, thickness=cv2.FILLED)

    np_overlay = np_image.copy()
    y1_overlay = y1_image.copy()
    np_overlay[np_mask == 255] = [0, 255, 0]  
    y1_overlay[y1_mask == 255] = [0, 0, 255]  

    comparison_output = os.path.join(output_folder, f"comparison_{np_img}")
    combined = np.hstack((np_overlay, y1_overlay))
    cv2.imwrite(comparison_output, combined)

    print(f"Saved comparison image: {comparison_output}")

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title("NP vs 1Y Detection Comparison")
    plt.axis("off")
    plt.show()
