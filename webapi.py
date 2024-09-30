import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw
from detect import detect_speech_bubbles, extract_text_from_image
from translate import translate_text
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create folders if they do not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Dictionary for Tesseract OCR languages
ocr_languages = {
    "zh-en": "chi_sim",   # Chinese Simplified for OCR
    "ja-en": "jpn",       # Japanese for OCR
    "ko-en": "kor"        # Korean for OCR
}

@app.route('/')
def index():
    return render_template('WebUi.html')  # Load the UI

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    lang = request.form.get('language')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Step 1: Detect speech bubbles
    detections = detect_speech_bubbles(file_path, model_path="Bubbledetect.pt")

    if detections is None or len(detections) == 0:
        return jsonify({"error": "No speech bubbles detected."}), 400

    ocr_lang = ocr_languages.get(lang, 'eng')  # Use the correct OCR language
    texts = []
    translated_texts = []
    img = Image.open(file_path)
    draw = ImageDraw.Draw(img)

    # Step 2: Process each speech bubble
    for i, bubble in enumerate(detections):
        # Get the coordinates of the speech bubble
        x1, y1, x2, y2 = map(int, bubble[:4])

        # Crop the image to the bubble area
        bubble_img = img.crop((x1, y1, x2, y2))

        # Extract text from the bubble area
        text = extract_text_from_image(bubble_img, lang=ocr_lang)
        texts.append(text)

        # Translate the text in the speech bubble
        translated_text = translate_text(text, lang)
        translated_texts.append(translated_text)

        # Draw the bubble location and name on the original image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"Bubble {i + 1}", fill="red")

    # Save the marked image
    marked_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"marked_{filename}")
    img.save(marked_image_path)

    # Return the bubbles and translations to the webpage
    result = {
        "message": "Processing complete",
        "filename": filename,
        "marked_image": f"outputs/marked_{filename}",
        "bubbles": [
            {"bubble_id": i + 1, "text": texts[i], "translated_text": translated_texts[i]}
            for i in range(len(detections))
        ]
    }

    return jsonify(result)

@app.route('/outputs/<filename>')
def send_marked_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    
    app.run(debug=True, host='0.0.0.0')  # Start Flask in the Docker container
