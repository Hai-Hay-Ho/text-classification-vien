import os

# Chỉ định thư mục cache nằm trong /app (nơi bạn có quyền ghi)
os.environ["HF_HOME"] = "/app/hf_cache"
os.makedirs("/app/hf_cache", exist_ok=True)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, MarianMTModel, MarianTokenizer

app = FastAPI()

classifier = None
model = None
tokenizer = None

@app.on_event("startup")
def load_models():
    global classifier, model, tokenizer
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    model_name = 'Helsinki-NLP/opus-mt-vi-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

class Message(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(message: Message):
    src_text = [message.text]
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    result = classifier(translated_text)[0]
    return {
        "input": message.text,
        "translated": translated_text,
        "sentiment": result
    }
