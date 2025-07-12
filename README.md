# car-and-pedstrain-tracking-using-machine-learning-
# YOLOv5 Traffic Object Detection

This project demonstrates traffic object detection using the YOLOv5 model with Ultralytics' API. It processes video input, identifies key traffic-related objects — cars, people, and bicycles — and overlays real-time counts and bounding boxes.

---

## Features

- Uses YOLOv5s pretrained model
- Detects and counts:
  - Cars
  - People
  - Bicycles
- Displays annotated video frames using OpenCV
- Custom class filtering and live object counting
- Google Colab compatible

---

## How It Works

1. **Zip Extraction**: The script expects a `.zip` file containing a `.mp4` video (`dhaka_traffic.mp4.zip`). It extracts the video file into a temporary directory.
2. **Model Loading**: YOLOv5s model is loaded using the Ultralytics API.
3. **Detection**: Frame-by-frame object detection is performed.
4. **Visualization**: Bounding boxes and real-time object counts are displayed with `cv2_imshow()`.

---

## Dependencies

Install required packages (for Google Colab):

```bash
!pip install -q ultralytics opencv-python
.
├── traffic_detection.py      # Main detection script
├── dhaka_traffic.mp4.zip     # Compressed input video (upload in Colab)
└── README.md                 # Project overview
