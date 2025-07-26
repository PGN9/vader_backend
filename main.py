from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
import os
import psutil
import requests
import traceback
import platform
import json
import time
import asyncio

# === Global Load Flag ===
model_loaded = False

# === Model Info ===
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"
ONNX_MODEL_URL = "https://huggingface.co/dakyswr/lxyuan-distilbert-sentiment-onnx/resolve/main/model-quant.onnx"

# === Download ONNX Model if Needed ===
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        print("Downloading ONNX model...")
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
        with open(ONNX_MODEL_PATH, "wb") as f:
            f.write(requests.get(ONNX_MODEL_URL).content)
        print("Download complete.")

download_model()

# === App Init ===
app = FastAPI()
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# === Classes ===
class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

# === Helpers ===
def get_size_in_kb(data: str) -> float:
    return len(data.encode("utf-8")) / 1024

@app.get("/")
def root():
    return {"message": "Welcome! The multilingual ONNX model is live."}

# === Health Check Route ===
@app.get("/health")
def health():
    return {"status": "ok"}

# === Startup Event: Load Tokenizer & Model ===
@app.on_event("startup")
async def load_model():
    global tokenizer, session, model_loaded
    try:
        print("🔄 Loading tokenizer and ONNX model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        
        # 🔥 Warm-up with a dummy comment
        dummy_input = tokenizer(["Hello world!"], return_tensors="np", truncation=True, padding=True)
        _ = session.run(None, {
            "input_ids": dummy_input["input_ids"],
            "attention_mask": dummy_input["attention_mask"]
        })

        model_loaded = True
        print("✅ Model ready and warm-up complete.")
    except Exception as e:
        print("❌ Model load failed:", str(e))
        traceback.print_exc()


# === Predict Endpoint ===
@app.post("/predict")
async def predict_sentiment(request: CommentsRequest):
    if not model_loaded:
        return JSONResponse(status_code=503, content={"error": "Model is not yet loaded."})

    try:
        print("📨 Received /predict request")
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        input_json = json.dumps(request.dict())
        total_data_size_kb = get_size_in_kb(input_json)

        texts = [comment.body for comment in request.comments]
        ids = [comment.id for comment in request.comments]

        inputs = tokenizer(texts, return_tensors="np", truncation=True, padding=True, max_length=512)
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        start_time = time.time()
        logits = session.run(None, onnx_inputs)[0]  # Shape: (batch_size, 3)
        elapsed = time.time() - start_time
        print(f"⏱️ Inference time for {len(texts)} comments: {round(elapsed, 3)} seconds")

        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        label_ids = np.argmax(probs, axis=1)

        results = []
        for i, label_id in enumerate(label_ids):
            results.append({
                "id": ids[i],
                "body": texts[i],
                "sentiment": LABEL_MAP.get(int(label_id), "unknown"),
                "sentiment_score": round(float(probs[i][label_id]), 4)
            })

        current_memory_mb = process.memory_info().rss / 1024 / 1024
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        elif platform.system() == "Darwin":
            import resource
            peak_memory_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        else:
            peak_memory_mb = current_memory_mb

        return {
            "model_used": MODEL_NAME,
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2),
            "total_data_size_kb": round(total_data_size_kb, 2),
            "total_return_size_kb": round(get_size_in_kb(json.dumps(results)), 2)
        }

    except Exception as e:
        print("❌ Error during inference:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# === Run Server ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
