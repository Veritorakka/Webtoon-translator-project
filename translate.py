import torch
from transformers import MarianMTModel, MarianTokenizer  # type: ignore

model_names = {
    "zh-en": 'Helsinki-NLP/opus-mt-zh-en',   # Chinese to English
    "ja-en": 'Helsinki-NLP/opus-mt-ja-en',   # Japanese to English
    "ko-en": 'Helsinki-NLP/opus-mt-ko-en'    # Korean to English
}

def translate_text(input_text, lang):
    # Tarkistetaan GPU ja käytetään sitä, jos se on saatavilla
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Lataa MarianMT-malli ja siirrä GPU:lle tai CPU:lle
    tokenizer = MarianTokenizer.from_pretrained(model_names[lang])
    model = MarianMTModel.from_pretrained(model_names[lang]).to(device1)

    # Tokenisoi teksti ja siirrä GPU:lle tai CPU:lle
    tokens = tokenizer(input_text, return_tensors="pt", padding=True).to(device1)

    # Generoi käännös
    translated = model.generate(**tokens)

    # Purkaa käännös tekstiksi
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text
