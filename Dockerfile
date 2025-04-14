FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    poppler-utils \
    ffmpeg \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt\
    && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
