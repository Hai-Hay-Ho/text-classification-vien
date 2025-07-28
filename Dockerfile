FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

COPY textClassification.py .

CMD ["uvicorn", "textClassification:app", "--host", "0.0.0.0", "--port", "7860"]
