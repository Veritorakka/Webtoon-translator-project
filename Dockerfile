# Käytä Python 3.8 -pohjakuvaa
FROM python:3.8-slim

# Aseta työhakemisto
WORKDIR /app

# Asenna tarvittavat työkalut, kuten Tesseract ja tarvittavat kielet (kiina, japani, korea) sekä OpenGL
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \  
    tesseract-ocr-jpn \      
    tesseract-ocr-kor \
    tesseract-ocr-eng \      
    libtesseract-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Aseta Tesseractin kielitiedoston sijainti
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Kopioi requirements-tiedosto konttiin
COPY requirements.txt .

# Asenna Python-riippuvuudet (Flask, PyTorch, ja muut kirjastot)
RUN pip install --no-cache-dir -r requirements.txt

# Kopioi projektin tiedostot ja testidata konttiin
COPY . .

# Luo kansio ladatuille kuville
RUN mkdir -p /app/uploads

# Aseta Flaskin ympäristömuuttujat
ENV FLASK_APP=webapi.py
ENV FLASK_RUN_HOST=0.0.0.0

# Exponoi portti 5000 Flask-sovellukselle
EXPOSE 5000

# Aseta oletuskomento Flask-sovelluksen käynnistämiseen
CMD ["flask", "run"]
