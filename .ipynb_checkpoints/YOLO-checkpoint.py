import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")   # auto download

st.title("🔥 Smart Image Detector (All Objects)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    # Prediction
    results = model(image)

    labels = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

    if len(labels) == 0:
        st.write("❌ Kuch detect nahi hua")
    else:
        unique_labels = list(set(labels))
        st.write("✅ Image me ye objects hain:")
        
        for lbl in unique_labels:
            st.write(f"👉 {lbl}")