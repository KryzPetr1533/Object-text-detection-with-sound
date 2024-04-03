import torch
import cv2 as cv
import numpy as np
import logging as log
from ultralytics import YOLO


camera_ind = 0
cap = cv.VideoCapture(camera_ind)

model_path = "./yolov5s.pt"

if cap is None or not cap.isOpened():
    log.error('Camera is not found.')
    log.info('Exit...')
else:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)


    while True:
        status, img = cap.read()
        if status is False:
            log.error('No image is read.')
        else:
            results = model(img)
            cv.imshow("Result", np.squeeze(results.render()))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break