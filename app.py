import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLO model (auto download first time)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🔥 Smart Image Object Detector")
st.write("Upload any image → Detect all objects")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run YOLO prediction
    with st.spinner("🔍 Detecting objects..."):
        results = model(image)

    # Draw bounding boxes
    result_img = results[0].plot()

    st.image(result_img, caption="Detected Objects", use_container_width=True)

    # Extract labels
    labels = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

    # Remove duplicates
    unique_labels = list(set(labels))

    # Show results
    st.subheader("✅ Detected Objects:")

    if len(unique_labels) == 0:
        st.write("❌ No objects detected")
    else:
        for lbl in unique_labels:
            st.write(f"👉 {lbl}")