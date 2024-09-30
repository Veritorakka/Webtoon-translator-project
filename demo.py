from detect import detect_speech_bubbles, extract_text_from_image
from translate import translate_text
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Dictionary for Tesseract OCR languages
ocr_languages = {
    "zh-en": "chi_sim",   # Chinese Simplified for OCR
    "ja-en": "jpn",       # Japanese for OCR
    "ko-en": "kor"        # Korean for OCR
}

def process_image(image_path, model_path='Bubbledetect.pt', lang='zh-en'):
    # Step 1: Detect speech bubbles
    print("Detecting speech bubbles...")
    detections = detect_speech_bubbles(image_path, model_path)

    if detections is None or len(detections) == 0:
        print("No speech bubbles detected.")
        return

    # Step 2: Extract text from each speech bubble
    ocr_lang = ocr_languages.get(lang, 'eng')  # Use the correct OCR language based on translation language
    texts = []
    translated_texts = []  # List to store translated texts

    for i, bubble in enumerate(detections):
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(int, bubble[:4])
        
        # Load the image and crop it to the bubble area
        img = Image.open(image_path)
        bubble_img = img.crop((x1, y1, x2, y2))

        # Extract text from the cropped bubble image
        print(f"Extracting text from bubble {i + 1}...")
        text = extract_text_from_image(bubble_img, lang=ocr_lang)
        texts.append(text)
        print(f"Extracted text: {text}")

        # Step 3: Translate each bubble's text separately
        print(f"Translating text from bubble {i + 1}...")
        translated_text = translate_text(text, lang)
        translated_texts.append(translated_text)
        print(f"Translated text: {translated_text}")

    # Step 4: Print the final translated texts
    print("Final Translated Texts:")
    for i, translated in enumerate(translated_texts):
        print(f"Bubble {i + 1}: {translated}")

# Example usage
process_image("demo.png", model_path="Bubbledetect.pt", lang="zh-en")
