import torch
from PIL import Image, ImageEnhance
import cv2
import pytesseract  # type: ignore
import os
import numpy as np
from ultralytics import YOLO  # YOLOv8
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

# Function to run YOLOv8 and detect speech bubbles
def detect_speech_bubbles(image_path, model_path='Bubbledetect.pt'):
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]

    # Letterbox the image to 640x640 while preserving aspect ratio
    img_letterboxed, ratio, dw, dh = letterbox_image(img)

    # Ensure path compatibility across platforms
    ensure_path_compatibility()

    # Load the YOLOv8 model
    device = torch.device('cpu')
    
    try:
        model = YOLO(model_path)  # YOLOv8 model loading
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # Run inference on the model
    try:
        results = model(img_letterboxed)  # Call model directly with image
    except Exception as e:
        print(f"Error during inference: {e}")
        return []

    # Extract detection results and rescale coordinates back to original image size
    try:
        detections = results[0].boxes.xyxy.cpu().numpy()  # Get the predictions for the first image
        detections[:, [0, 2]] = (detections[:, [0, 2]] - dw) / ratio  # Rescale x-coordinates
        detections[:, [1, 3]] = (detections[:, [1, 3]] - dh) / ratio  # Rescale y-coordinates
    except Exception as e:
        print(f"Error processing detections: {e}")
        return []

    return detections


def resize_image_if_needed(img, min_size=(500, 500)):
    """
    Check if the image width or height is below the given threshold (500x500 by default).
    If so, resize the image so that the shorter side is at least min_size.
    """
    width, height = img.size
    if width < min_size[0] or height < min_size[1]:
        # Calculate the scaling factor while preserving the aspect ratio
        scale_factor = max(min_size[0] / width, min_size[1] / height)
        new_size = (int(width * scale_factor), int(height * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)  # Use LANCZOS filter for resizing
    return img


def extract_text_from_image(image, output_txt_path=None, lang='eng', try_multiple_psm=True):
    # Check if the input is a file path or a direct PIL image
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image

    # Resize the image if it is smaller than 500x500 pixels
    img = resize_image_if_needed(img, min_size=(500, 500))
    
    # Enhance the contrast and sharpness of the image
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(2.0)  # Improve contrast
    
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(2.0)  # Improve sharpness

    # Convert the image to grayscale and apply binarization
    img = img.convert('L')  # Grayscale
    img = img.point(lambda x: 0 if x < 128 else 255, '1')  # Binarization

    # Try extracting the text without specifying a PSM mode first
    text = pytesseract.image_to_string(img, lang=lang)

    # If multiple PSM modes are allowed, try them if no text is found
    if try_multiple_psm and not text.strip():
        for psm_mode in [6, 3, 4]:  # Try PSM 6, 3, and 4
            config = f'--psm {psm_mode}'
            text = pytesseract.image_to_string(img, lang=lang, config=config)
            if text.strip():
                break

    # If an output file path is provided, write the text to a file
    if output_txt_path:
        with open(output_txt_path, 'w') as f:
            f.write(text)

    return text
