FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./api/

RUN pip install --no-cache-dir -r api/requirements.txt

COPY api/ ./api/

COPY models/resnet50_stage2.keras ./models/

EXPOSE 5000

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/resnet50_stage2.keras

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

CMD ["python", "api/app.py"]

