🕵️ Object Detection and Counting App

An interactive Streamlit web app that uses YOLOv8 for object detection, counting, and real-time visualization on images, videos, and webcam streams.
It allows users to upload files, adjust detection confidence, and view detection statistics with dynamic charts.


🚀 Features

📷 Image Detection – Upload an image and detect all objects in it.
🎬 Video Detection – Upload a video, play it frame-by-frame, and see live detections.
🎥 Webcam Mode – Detect and count objects in real time using your webcam.
📊 Automatic Object Counting – Displays total detected objects with bar charts.
💾 Download Processed Images – Save annotated results for reference.
⚙️ Adjustable Confidence Threshold – Control detection sensitivity.
📈 Visual Insights – Get object distribution with interactive Plotly charts.


🧠 Tech Stack

Component	Description
Python -	Programming language
Streamlit -	Web app framework
OpenCV -	Image & video processing
Ultralytics YOLOv8 -	Object detection model
Pandas -	Data manipulation
Plotly -	Data visualization
streamlit-webrtc -	Real-time webcam streaming


📁 Project Structure

📂 object_detection_app/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Dependencies
├── yolov8n.pt             # Pre-trained YOLOv8 model (auto-downloaded if missing)
└── README.md              # Project documentation


🧩 How It Works

The YOLOv8 model detects objects in each frame or image.
Detected labels and bounding boxes are drawn using OpenCV.
Results are displayed in Streamlit with object counts and bar charts.
For webcam input, frames are processed in real time via streamlit-webrtc.


⚡ Tips for Better Performance

Use YOLOv8n.pt (Nano model) for faster results.
On Streamlit Cloud, large videos may lag — prefer short clips or local runs.
Use smaller frame sizes (640×480) to improve FPS.


👨‍💻 Author

Sahil Kamble
📧 Email: sahilkamble0134@gmail.com
🌐 GitHub: github.com/sahils-hub
