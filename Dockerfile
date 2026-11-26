FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# System libraries required by OpenCV & PyTorch on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU PyTorch first for a leaner container, then locked Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0 \
    && pip install -r requirements.txt

# Pre-download YOLO11n-pose weights during image build to prevent cold-start delay
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"

COPY main.py .
COPY templates ./templates

# Ensure non-root runtimes (e.g., Hugging Face Spaces UID 1000) can read model weights
RUN chmod -R a+rX /app

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
