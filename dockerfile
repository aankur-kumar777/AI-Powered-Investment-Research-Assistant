# Dockerfile
FROM python:3.10-slim

# system deps for pdf processing and tesseract; adjust as needed for your base image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# copy project
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
