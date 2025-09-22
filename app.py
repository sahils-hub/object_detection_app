import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # lightweight model for speed

# ---------------- APP HEADER ----------------
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("🕵️ Object Detection and Counting App")
st.write("This app lets you **detect objects**, **count them**, and **visualize results** in real-time using YOLOv8.")

# ---------------- SIDEBAR SETTINGS ----------------
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Object filter (Stage 5)
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

        stframe = st.empty()
        detected_list = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=confidence)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            # Track detections for statistics
            detected_classes = filter_detections(results)
            detected_list.extend(detected_classes)

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

# ---------------- WEBCAM REAL-TIME ----------------
elif upload_type == "Webcam":
    st.write("🎥 Starting webcam... (press **Stop** to end)")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    detected_list = []

    stop_button = st.button("🛑 Stop Webcam")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        # Track detections
        detected_classes = filter_detections(results)
        detected_list.extend(detected_classes)

    cap.release()

    # Show summary after stopping webcam
    if detected_list:
        counts = pd.Series(detected_list).value_counts().reset_index()
        counts.columns = ["Object", "Count"]
        st.subheader("📊 Webcam Session Summary")
        st.dataframe(counts, use_container_width=True)
        fig = px.bar(counts, x="Object", y="Count", text="Count",
                     title="Object Count Distribution", color="Object")
        st.plotly_chart(fig, use_container_width=True)
