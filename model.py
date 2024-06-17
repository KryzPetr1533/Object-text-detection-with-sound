import cv2 as cv
import numpy as np
import easyocr
import torch
from ultralytics import YOLO
import pyttsx3
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Create reader instance for EasyOCR
reader = easyocr.Reader(lang_list=['en', 'ru'], gpu=True)

# Initialize TTS engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# Initialize a queue to hold text for TTS
text_queue = queue.Queue()

# Set to hold spoken texts
spoken_texts = set()

# Open the video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv.CAP_PROP_FPS, 30)

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).to('cuda')

def speak_text():
    logging.debug("TTS thread started.")
    while True:
        text = text_queue.get()
        if text is None:
            text_queue.task_done()  # Mark the None task as done before breaking
            break
        logging.debug(f"Speaking text: {text}")
        tts.say(text)
        tts.runAndWait()
        text_queue.task_done()
    logging.debug("TTS thread finished.")

def draw_text(img, text, pos, font_scale=0.8, font_thickness=2, font=cv.FONT_HERSHEY_COMPLEX):
    # Text display configuration
    font_color = (0, 255, 0)
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = pos[0]
    text_y = pos[1] - 10
    box_coords = ((text_x, text_y), (text_x + text_size[0] + 2, text_y - text_size[1] - 2))
    cv.rectangle(img, box_coords[0], box_coords[1], font_color, cv.FILLED)
    cv.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

# Start the TTS thread
tts_thread = threading.Thread(target=speak_text, daemon=True)
tts_thread.start()

try:
    while True:
        status, img = cap.read()
        if not status:
            print("Failed to grab frame")
            break

        # Convert the image to grayscale
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        results = model(img)

        # OCR Processing
        text_results = reader.readtext(gray_img)

        for (bbox, text, prob) in text_results:
            # EasyOCR returns bbox as [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            # Extract top-left and bottom-right coordinates for cv.rectangle
            top_left = tuple(map(int, bbox[0]))  # Converts float to int and list to tuple
            bottom_right = tuple(map(int, bbox[2]))

            # Draw rectangles and text
            cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            draw_text(img, text, top_left)

            # Add text to the queue if it hasn't been spoken yet
            if text not in spoken_texts:
                text_queue.put(text)
                spoken_texts.add(text)

        # Display the results
        cv.imshow('Result', np.squeeze(results.render()))

        # Break the loop if 'ESC' is pressed
        if cv.waitKey(1) == 27:
            logging.debug("ESC key pressed. Exiting.")
            break
finally:
    # Clean up
    logging.debug("Releasing video capture and closing windows.")
    cap.release()
    cv.destroyAllWindows()
    
    # Signal the TTS thread to exit and wait for it to finish
    logging.debug("Signaling TTS thread to exit.")
    text_queue.put(None)
    tts_thread.join()
    logging.debug("Application terminated.")
