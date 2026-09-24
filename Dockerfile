FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY anonymizer ./anonymizer
ENV HOST=0.0.0.0 DATA_DIR=/app/data MODEL_DIR=/app/models DEVICE=cpu PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "anonymizer"]
