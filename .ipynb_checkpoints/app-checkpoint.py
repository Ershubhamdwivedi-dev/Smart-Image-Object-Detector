import streamlit as st
from PIL import Image
import numpy as np

# IMPORTANT FIX: must use headless-safe opencv
import cv2

from ultralytics import YOLO

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
@st.cache_resource
def load_model():
    # safe official model download
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🔥 Smart Image Object Detector")
st.write("Upload any image and detect objects using YOLOv8")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting objects..."):
        results = model(image)

    # draw results
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Objects", use_container_width=True)

    # -----------------------------
    # EXTRACT LABELS
    # -----------------------------
    labels = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            labels.append(model.names[cls_id])

    unique_labels = sorted(set(labels))

    st.subheader("Detected Objects:")

    if len(unique_labels) == 0:
        st.warning("No objects detected")
    else:
        for label in unique_labels:
            st.write("👉", label)