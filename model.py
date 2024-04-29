import cv2 as cv
import numpy as np
import easyocr
import torch
from ultralytics import YOLO

# Create reader instance for EasyOCR
reader = easyocr.Reader(['en', 'ru'])

# Open the video capture
cap = cv.VideoCapture(0)
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

while True:
    status, img = cap.read()
    if not status:
        print("Failed to grab frame")
        break

    results = model(img)

    # OCR Processing
    text_results = reader.readtext(img)

    for (bbox, text, prob) in text_results:
        # EasyOCR returns bbox as [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        # Extract top-left and bottom-right coordinates for cv.rectangle
        top_left = tuple(map(int, bbox[0]))  # Converts float to int and list to tuple
        bottom_right = tuple(map(int, bbox[2]))  # Converts float to int and list to tuple

        # Draw rectangles and text
        cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv.putText(img, text, (top_left[0], top_left[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the results
    cv.imshow('Result', np.squeeze(results.render()))

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
