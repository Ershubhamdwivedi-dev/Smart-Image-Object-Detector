import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")   # auto-download
    return model

model = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🔥 Smart Image Object Detector ")
st.write("Upload any image → Detect objects (person, dog, car etc.)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(img)

    results = model(img_np)

    # show image with boxes
    st.image(results[0].plot(), caption="Detected Objects")

    # labels
    st.subheader("Detected Objects:")

    names = results[0].names
    boxes = results[0].boxes

    detected = set()

    for box in boxes:
        cls_id = int(box.cls[0])
        detected.add(names[cls_id])

    for obj in detected:
        st.write(f"👉 {obj}")