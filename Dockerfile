FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/best.pt

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# OpenCV is imported by Ultralytics during model initialisation. The slim
# Python image does not include these shared libraries by default. This layer
# follows Python dependencies so changing it does not invalidate the large
# cached PyTorch installation.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
