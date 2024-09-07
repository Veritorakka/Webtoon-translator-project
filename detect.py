import torch
from PIL import Image
import cv2
import pytesseract  # type: ignore
import os
import numpy as np

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

# Function to run YOLOv5 and detect speech bubbles
def detect_speech_bubbles(image_path, model_path='Bubbledect.pt'):
    img = cv2.imread(image_path)
    original_shape = img.shape[:2]  # original image height and width

    # Load the YOLOv5 model from the local repository
    model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Letterbox the image
    img_letterboxed, ratio, dw, dh = letterbox_image(img)

    # Convert letterboxed image to PIL Image and pass to model
    img_pil = Image.fromarray(img_letterboxed)
    results = model(img_pil)

    # Extract detection results and rescale coordinates back to original image size
    detections = results.xyxy[0]  # Coordinates (x1, y1, x2, y2)
    detections[:, [0, 2]] = (detections[:, [0, 2]] - dw) / ratio  # Rescale x-coordinates
    detections[:, [1, 3]] = (detections[:, [1, 3]] - dh) / ratio  # Rescale y-coordinates
    return detections

# Function to extract text from an image using Tesseract, with language selection
def extract_text_from_image(image_path, output_txt_path, lang='chi_sim'):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    with open(output_txt_path, 'w') as f:
        f.write(text)

# Main function to process images, can be called externally
def process_images(input_dir="input_images", temp_dir="temp_images", output_dir="output_texts", lang='chi_sim'):
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        
        # Detect speech bubbles in the image
        detections = detect_speech_bubbles(img_path)
        
        for i, det in enumerate(detections):
            # Extract bounding box coordinates
            x1, y1, x2, y2, conf, cls = det

            # Crop the speech bubble from the original image using rescaled coordinates
            img = cv2.imread(img_path)
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]

            # Save the cropped image as a temp file
            temp_img_path = os.path.join(temp_dir, f"{img_name}_bubble_{i}.png")
            cv2.imwrite(temp_img_path, crop_img)

            # Extract text from the temp image using the specified language
            temp_txt_path = os.path.join(output_dir, f"{img_name}_bubble_{i}.txt")
            extract_text_from_image(temp_img_path, temp_txt_path, lang=lang)
