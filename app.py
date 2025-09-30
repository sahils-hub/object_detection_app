import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os

# NEW imports for webcam streaming
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # lightweight model for speed

# ---------------- APP HEADER ----------------
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("🕵️ Object Detection and Counting App")
st.write("This app lets you **detect objects**, **count them**, and **visualize results** in real-time using YOLOv8.")

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Object filter
all_classes = list(model.names.values())
selected_object = st.sidebar.selectbox("🎯 Select object to detect (or 'All')", ["All"] + all_classes)

# Let user choose input type
upload_type = st.radio("Choose input type:", ["Image", "Video", "Webcam"])

# ---------------- HELPER FUNCTION ----------------
def filter_detections(results):
    """Filter detections based on selected object"""
    detected_classes = [model.names[int(c)] for c in results[0].boxes.cls]
    if selected_object != "All":
        detected_classes = [obj for obj in detected_classes if obj == selected_object]
    return detected_classes

# ---------------- IMAGE UPLOAD ----------------
if upload_type == "Image":
    uploaded_image = st.file_uploader("📷 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run YOLO model
        results = model.predict(np.array(image), conf=confidence)

        # Draw detections
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="Detections", use_container_width=True)

        # Count detected objects
        detected_classes = filter_detections(results)
        if detected_classes:
            counts = pd.Series(detected_classes).value_counts().reset_index()
            counts.columns = ["Object", "Count"]

            # Show table
            st.subheader("📊 Detection Results")
            st.dataframe(counts, use_container_width=True)

            # Show bar chart
            fig = px.bar(counts, x="Object", y="Count", text="Count",
                         title="Object Count Distribution", color="Object")
            st.plotly_chart(fig, use_container_width=True)

            # Save processed image for download
            out_path = "processed_image.jpg"
            cv2.imwrite(out_path, res_plotted)
            with open(out_path, "rb") as f:
                st.download_button("📥 Download Processed Image", f, file_name="detections.jpg")

# ---------------- VIDEO UPLOAD ----------------
elif upload_type == "Video":
    uploaded_video = st.file_uploader("🎬 Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        delay = 1 / fps if fps > 0 else 0.03  # default ~30fps

        # Slider UI
        frame_slider = st.slider("📍 Video Position", 0, total_frames - 1, 0, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)

        stframe = st.empty()
        slider_placeholder = st.empty()
        detected_list = []

        # Play video from current position
        current_frame = frame_slider
        while cap.isOpened() and current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection
            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()

            # Display video frame
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            # Track detections
            detected_classes = filter_detections(results)
            detected_list.extend(detected_classes)

            # Update slider dynamically
            slider_placeholder.slider("📍 Video Position", 0, total_frames - 1,
                                      current_frame, 1, key="progress", disabled=True)

            current_frame += 1
            time.sleep(delay)  # maintain playback speed

        cap.release()

        # Show summary counts after video ends
        if detected_list:
            counts = pd.Series(detected_list).value_counts().reset_index()
            counts.columns = ["Object", "Count"]
            st.subheader("📊 Final Detection Summary")
            st.dataframe(counts, use_container_width=True)
            fig = px.bar(counts, x="Object", y="Count", text="Count",
                         title="Object Count Distribution", color="Object")
            st.plotly_chart(fig, use_container_width=True)

# ---------------- WEBCAM REAL-TIME (with streamlit-webrtc) ----------------
elif upload_type == "Webcam":
    st.subheader("🎥 Live Webcam Detection")

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.detected_list = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # Run YOLO detection
            results = model.predict(img, conf=confidence)
            img = results[0].plot()

            # Track detections
            detected_classes = filter_detections(results)
            self.detected_list.extend(detected_classes)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="object-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
