!pip install -q ultralytics

import cv2
import os
import zipfile
import tempfile
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

def extract_zip(zip_path):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

model = YOLO('yolov5s.pt')

zip_file_path = '/content/dhaka_traffic.mp4.zip'

if not os.path.exists(zip_file_path):
    raise FileNotFoundError(f"{zip_file_path} not found. Please upload the zip file to Colab.")

extracted_dir = extract_zip(zip_file_path)

target_classes = {'car': 0, 'person': 0, 'bicycle': 0}

for video_file in os.listdir(extracted_dir):
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(extracted_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (640, 360))

        results = model(resized_frame)[0]

        counts = {'car': 0, 'person': 0, 'bicycle': 0}

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = model.names[cls_id]

            if class_name in counts and conf > 0.3:
                counts[class_name] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 255) if class_name == 'car' else (0, 0, 255) if class_name == 'person' else (255, 0, 0)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(resized_frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        y_offset = 30
        for cls, cnt in counts.items():
            color = (0, 255, 255) if cls == 'car' else (0, 0, 255) if cls == 'person' else (255, 0, 0)
            cv2.putText(resized_frame, f"{cls.capitalize()}: {cnt}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 40

        cv2_imshow(resized_frame)

    cap.release()

cv2.destroyAllWindows()
