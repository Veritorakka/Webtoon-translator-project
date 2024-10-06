from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageDraw
from detect import detect_speech_bubbles, extract_text_from_image
from translate import translate_text
from werkzeug.utils import secure_filename
import os

# Yritetään tuoda llm.py tiedoston funktio
try:
    from llm import translate_and_provide_context
    llm_available = True
except ImportError:
    llm_available = False

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Dictionary for Tesseract OCR languages
ocr_languages = {
    "zh-en": "chi_sim",
    "ja-en": "jpn",
    "ko-en": "kor"
}

def calculate_overlap_area(box1, box2):
    """Laskee kahden suorakulmion päällekkäisyyden prosenttiosuuden."""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Laske päällekkäisen alueen leveys ja korkeus
    x_overlap = max(0, min(x2, x2_2) - max(x1, x1_2))
    y_overlap = max(0, min(y2, y2_2) - max(y1, y1_2))

    # Jos päällekkäisyyttä ei ole
    if x_overlap == 0 or y_overlap == 0:
        return 0

    # Päällekkäisen alueen pinta-ala
    overlap_area = x_overlap * y_overlap

    # Alkuperäisten suorakulmioiden pinta-alat
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Päällekkäisyyden prosenttiosuus suhteessa pienempään suorakulmioon
    smaller_area = min(area1, area2)
    overlap_percent = (overlap_area / smaller_area) * 100

    return overlap_percent

def merge_boxes(box1, box2):
    """Yhdistää kaksi suorakulmiota, jos ne menevät tarpeeksi päällekkäin."""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return [min(x1, x1_2), min(y1, y1_2), max(x2, x2_2), max(y2, y2_2)]

def combine_overlapping_detections(detections, overlap_threshold=40):
    """Yhdistää tunnistukset, jos niiden päällekkäisyys ylittää annetun kynnyksen."""
    combined_detections = []
    
    # Muutetaan detections listaksi
    detections = detections.tolist()
    
    while len(detections) > 0:
        current_box = detections.pop(0)
        merged = False
        
        for i, existing_box in enumerate(combined_detections):
            if calculate_overlap_area(current_box[:4], existing_box[:4]) > overlap_threshold:
                # Yhdistä tunnistukset ja päivitä olemassa oleva laatikko
                combined_detections[i] = merge_boxes(current_box[:4], existing_box[:4])
                merged = True
                break

        if not merged:
            combined_detections.append(current_box)
    
    return combined_detections



@app.route('/')
def index():
    return render_template('WebUi.html')  # Lataa UI

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

    # Combine overlapping detections
    combined_detections = combine_overlapping_detections(detections)

    ocr_lang = ocr_languages.get(lang, 'eng')
    texts = []
    translated_texts = []
    contexts = []
    img = Image.open(file_path)
    draw = ImageDraw.Draw(img)

    # Step 2: Process each speech bubble
    for i, bubble in enumerate(combined_detections):
        x1, y1, x2, y2 = map(int, bubble[:4])
        bubble_img = img.crop((x1, y1, x2, y2))
        text = extract_text_from_image(bubble_img, lang=ocr_lang)
        texts.append(text)

        if llm_available:
            try:
                translated_text, context = translate_and_provide_context(text, lang)
                translated_texts.append(translated_text)
                contexts.append(context)
            except Exception as e:
                print(f"LLM translation failed: {e}")
                translated_text = translate_text(text, lang)
                translated_texts.append(translated_text)
                contexts.append("Context not available.")
        else:
            translated_text = translate_text(text, lang)
            translated_texts.append(translated_text)
            contexts.append("Context not available.")

        # Draw the bubble location and name on the original image
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"Bubble {i + 1}", fill="red")

    marked_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"marked_{filename}")
    img.save(marked_image_path)

    result = {
        "message": "Processing complete",
        "filename": filename,
        "marked_image": f"outputs/marked_{filename}",
        "bubbles": [
            {"bubble_id": i + 1, "text": texts[i], "translated_text": translated_texts[i], "context": texts[i]}
            for i in range(len(combined_detections))
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
    
    app.run(debug=True, host='0.0.0.0')
