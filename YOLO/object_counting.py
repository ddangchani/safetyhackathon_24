# Object counting using YOLO

import cv2
from ultralytics import YOLO, solutions

# Load YOLO

model = YOLO("yolov8n.pt")
cap =