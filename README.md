Smart Image Object Detector is a deep learning-based web application that allows users to upload images and automatically detect objects present in them. The application uses the powerful YOLOv8 model for real-time object detection and is built with Streamlit for an interactive and user-friendly interface.

The system can identify multiple objects in a single image such as people, animals, vehicles, and everyday items. It displays bounding boxes around detected objects and lists their names, making it a practical implementation of computer vision in real-world scenarios.

Technologies Used =>
🐍 Python
🤖 YOLOv8 (Ultralytics)
🌐 Streamlit
🧮 NumPy
🖼️ Pillow (Image Processing)
🎥 OpenCV (Headless for deployment)

Features =>
Upload any image
Detect multiple objects in a single image
Draw bounding boxes around detected objects
Display detected object names
Fast and real-time predictions
Clean and simple UI

How to Run Locally =>
pip install -r requirements.txt
streamlit run app.py

Folder Structure=>

project/
│
├── app.py
├── requirements.txt
└── screenshots/
    ├── ui.png
    ├── upload.png
    ├── result.png
    └── objects.png
