import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

# Load the YOLO model once
model = YOLO("yolov8n.pt")

st.title("Object Detection App")

# Sidebar options
option = st.sidebar.selectbox("Choose Input Type", ["Image", "Video", "Webcam"])


# ================= Image Section =================
if option == "Image":
    st.subheader("Upload an Image")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        file_bytes = uploaded_image.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file_bytes)
        image_path = temp_file.name

        image = cv2.imread(image_path)

        results = model(image)
        annotated_image = results[0].plot()

        st.image(annotated_image, channels="BGR", use_column_width=True)


# ================= Video Section =================
elif option == "Video":
    st.subheader("Upload a Video")

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        video_path = temp_file.name

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames // fps if fps > 0 else 0

        # Skip slider
        skip_seconds = st.slider("Skip to (seconds)", 0, duration, 0)

        # Set video to start at skip position
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_seconds * fps)

        stframe = st.empty()

        # Inject JS to force autoplay
        st.markdown(
            """
            <script>
            var video = window.parent.document.querySelector('video');
            if(video) {
                video.play();
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()


# ================= Webcam Section =================
elif option == "Webcam":
    st.subheader("Webcam Live Feed")
    run = st.checkbox("Run")
    stframe = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture from webcam.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
