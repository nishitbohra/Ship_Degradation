import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 5)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return thresh

def save_report(degradation_data):
    csv = BytesIO()
    pd.DataFrame(degradation_data).to_csv(csv, index=False)
    csv.seek(0)
    return csv

def plot_degradation(df):
    plt.figure(figsize=(10, 5))
    df_pivot = df.pivot(index="Image", columns="Stage", values="Degradation %")
    for img in df_pivot.index:
        plt.plot(df_pivot.columns, df_pivot.loc[img], marker='o', linestyle='-', label=img)
    plt.xlabel("Time Stage")
    plt.ylabel("Degradation Percentage (%)")
    plt.title("Degradation Progression Over Time")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    st.pyplot(plt)

st.set_page_config(page_title="Ship Part Degradation Analysis", page_icon="🚢", layout="wide")

st.title("🚢 Ship Part Degradation Analysis Dashboard")
st.markdown("Upload microscopic images of different stages to analyze degradation over time.")

col1, col2 = st.columns(2)

with col1:
    np_file = st.file_uploader("Upload Newly Painted (NP) Image", type=["jpg", "png", "jpeg"])
    y1_file = st.file_uploader("Upload 1 Year (1Y) Image", type=["jpg", "png", "jpeg"])
    y1b_file = st.file_uploader("Upload 1 Year Brushed (1YB) Image", type=["jpg", "png", "jpeg"])

with col2:
    y5_file = st.file_uploader("Upload 5 Years (5Y) Image", type=["jpg", "png", "jpeg"])
    y5b_file = st.file_uploader("Upload 5 Years Brushed (5YB) Image", type=["jpg", "png", "jpeg"])

if st.button("Analyze Degradation 🚀"):
    uploaded_files = [np_file, y1_file, y1b_file, y5_file, y5b_file]
    labels = ["Newly Painted", "1 Year", "1 Year Brushed", "5 Years", "5 Years Brushed"]
    degradation_data = []

    st.subheader("🔍 Analysis Results")
    result_cols = st.columns(len(uploaded_files))
    
    for i, file in enumerate(uploaded_files):
        if file:
            image = Image.open(file)
            image = np.array(image)
            processed_image = process_image(image)
            degradation_percentage = np.count_nonzero(processed_image) / (image.shape[0] * image.shape[1]) * 100
            degradation_data.append({"Image": file.name, "Stage": labels[i], "Degradation %": round(degradation_percentage, 2)})
            
            with result_cols[i]:
                st.image(file, caption=f"{labels[i]} - Original", use_container_width=True)
                st.image(processed_image, caption=f"{labels[i]} - Processed", use_container_width=True)
                st.metric(label=f"{labels[i]} Degradation", value=f"{degradation_percentage:.2f}%")
    
    st.subheader("📊 Degradation Report")
    df = pd.DataFrame(degradation_data)
    st.dataframe(df)
    
    csv_file = save_report(degradation_data)
    st.download_button("📥 Download Report", csv_file, "degradation_report.csv", "text/csv")
    
    # Automatically show the degradation over time plot
    st.subheader("📈 Degradation Over Time")
    plot_degradation(df)
