import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO Object Detection Dashboard",
    layout="wide"
)

st.title("🚀 YOLOv8 Object Detection Dashboard")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.sidebar.header("⚙️ Pengaturan")
mode = st.sidebar.selectbox(
    "Pilih Mode",
    ["Webcam", "Upload Gambar", "Upload Video"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.5
)

if mode == "Webcam":
    st.subheader("📷 Deteksi Webcam Real-Time")

    run = st.checkbox("▶️ Jalankan Webcam")
    frame_window = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Tidak bisa mengakses webcam")
            break

        results = model(frame)[0]
        annotated = results.plot()

        frame_window.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            channels="RGB"
        )

    cap.release()

elif mode == "Upload Gambar":
    st.subheader("🖼️ Deteksi Gambar")

    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )
        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        results = model(img, conf=confidence_threshold)[0]

        if len(results.boxes) == 0:
            st.warning("⚠️ Tidak ada objek terdeteksi")
            st.image(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                caption="Gambar Asli",
                use_container_width=True
            )
        else:
            annotated = results.plot()
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Hasil Deteksi",
                use_container_width=True
            )

elif mode == "Upload Video":
    st.subheader("🎞️ Deteksi Video")

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        frame_window = st.image([])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]
            annotated = results.plot()

            frame_window.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB"
            )

        cap.release()
