# Käytä Python 3.8 -pohjakuvaa
FROM python:3.8-slim

# Aseta työhakemisto
WORKDIR /app

# Asenna tarvittavat työkalut
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-eng \
    libtesseract-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Aseta Tesseractin kielitiedoston sijainti
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Asenna Ollama virallisen asennusskriptin avulla
RUN curl -fsSL https://ollama.com/install.sh | bash

# Kopioi requirements-tiedosto konttiin
COPY requirements.txt .

# Asenna Python-riippuvuudet (Flask, PyTorch, ja muut kirjastot)
RUN pip install --no-cache-dir -r requirements.txt

# Kopioi projektin tiedostot konttiin
COPY . .

# Luo kansio ladatuille kuville
RUN mkdir -p /app/uploads

# Kopioi koko .ollama-kansio Windows-laitteelta Docker-konttiin /root/.ollama
COPY ollama_models/. /root/.ollama/

# Poista ollama_models-kansio app-hakemistosta
RUN rm -rf /app/ollama_models

# Aseta Flaskin ympäristömuuttujat
ENV FLASK_APP=webapi.py
ENV FLASK_RUN_HOST=0.0.0.0

# Exponoi portti 5000 Flask-sovellukselle
EXPOSE 5000

# Luo käynnistysskripti
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
flask run --host=0.0.0.0 --port=5000' > /app/start.sh && chmod +x /app/start.sh

# Aseta käynnistysskripti
CMD ["/app/start.sh"]
