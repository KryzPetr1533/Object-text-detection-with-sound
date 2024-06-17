import torch
import logging as log

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
log.indo("Model been downloaded.")