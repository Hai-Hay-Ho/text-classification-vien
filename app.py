import os

# Tạo thư mục cache để lưu mô hình
os.environ["HF_HOME"] = "/app/hf_cache"
os.makedirs("/app/hf_cache", exist_ok=True)

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, MarianMTModel, MarianTokenizer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
app = FastAPI()

# Bật CORS để cho phép gọi từ các trang web khác
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bạn có thể chỉ định cụ thể domain nếu muốn an toàn hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khai báo biến toàn cục cho pipeline và model
classifier = None
model = None
tokenizer = None

# Tải mô hình khi khởi động app
@app.on_event("startup")
def load_models():
    global classifier, model, tokenizer
    print("Đang tải mô hình...")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    model_name = 'Helsinki-NLP/opus-mt-vi-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print("Hoàn tất tải mô hình.")

# Khai báo kiểu dữ liệu input
class Message(BaseModel):
    text: str

# API chính
@app.post("/analyze")
def analyze_sentiment(message: Message):
    src_text = [message.text]
    # Dịch sang tiếng Anh
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    # Phân tích cảm xúc
    result = classifier(translated_text)[0]
    return {
        "input": message.text,
        "translated": translated_text,
        "sentiment": result
    }
@app.get("/", response_class=FileResponse)
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))