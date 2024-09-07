# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.0-base

# Set working directory
WORKDIR /app

# Install necessary tools like git, Tesseract, Marian-MT, and dependencies
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5.git

# Move to the YOLOv5 directory
WORKDIR /app/yolov5

# Copy the requirements file
COPY requirements.txt .

# Install CUDA-compatible PyTorch, Marian-MT, and other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Return to the main app directory
WORKDIR /app

# Copy the Webtoon-translator project files to the container, including Bubbledect.pt
COPY . .

# Set the entry point command
ENTRYPOINT ["python", "main.py"]
