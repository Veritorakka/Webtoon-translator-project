import torch
from PIL import Image
import cv2
import pytesseract  # type: ignore
import os
import numpy as np
import yolov5
import platform
import pathlib

# Disable PIL's image size limit (useful for large images)
Image.MAX_IMAGE_PIXELS = None

# Letterbox function to resize image to 1280x1280 while keeping the aspect ratio
def letterbox_image(img, new_shape=(1280, 1280), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width and height difference
    dw, dh = dw // 2, dh // 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_letterboxed = cv2.copyMakeBorder(img_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    
    return img_letterboxed, ratio, dw, dh

# Function to handle platform-specific path compatibility (Windows vs Linux)
def ensure_path_compatibility():
    plt = platform.system()
    if plt == 'Windows':
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath

# Function to run YOLOv5 and detect speech bubbles
def detect_speech_bubbles(image_path, model_path='Bubbledetect.pt'):
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]

    # Letterbox the image to 1280x1280 while preserving aspect ratio
    img_letterboxed, ratio, dw, dh = letterbox_image(img)

    # Ensure path compatibility across platforms
    ensure_path_compatibility()

    # Load the YOLOv5 model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = yolov5.load(model_path, device=device)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # Run inference on the model
    try:
        print("Running inference on the model...")
        results = model(img_letterboxed)  # Call model directly with image
        print("Inference completed.")
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    # Extract detection results and rescale coordinates back to original image size
    try:
        detections = results.xyxy[0]  # Get the predictions for the first image
        detections = detections.cpu().numpy()  # Convert to numpy array
        detections[:, [0, 2]] = (detections[:, [0, 2]] - dw) / ratio  # Rescale x-coordinates
        detections[:, [1, 3]] = (detections[:, [1, 3]] - dh) / ratio  # Rescale y-coordinates
        print(f"Detections found: {len(detections)}")
    except Exception as e:
        print(f"Error processing detections: {e}")
        return []

    return detections


def extract_text_from_image(image_path, output_txt_path, lang='eng'):
    img = Image.open(image_path)
    
    # Muunna kuva harmaasävyksi ja tee binääristäminen
    img = img.convert('L')  # Harmaasävy
    img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binääristäminen

    # Poimi teksti
    text = pytesseract.image_to_string(img, lang=lang)
    
    # Tallenna teksti tiedostoon
    with open(output_txt_path, 'w') as f:
        f.write(text)

