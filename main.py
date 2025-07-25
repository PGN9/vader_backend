from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import os
import psutil
from fastapi.responses import JSONResponse
import traceback
import platform
import json

# Load tokenizer and quantized ONNX model
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
try:
    session = ort.InferenceSession(ONNX_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model at {ONNX_MODEL_PATH}: {e}")

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

app = FastAPI()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data: str) -> float:
    return len(data.encode("utf-8")) / 1024

@app.get("/")
def root():
    return {"message": f"Sentiment Classification using {MODEL_NAME}"}

@app.post("/predict")
def predict_sentiment(request: CommentsRequest):
    try:
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        input_json = json.dumps(request.dict())
        total_data_size_kb = get_size_in_kb(input_json)

        results = []

        for comment in request.comments:
            text = comment.body

            inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True, max_length=512)
            onnx_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }

            logits = session.run(None, onnx_inputs)[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            label_id = int(np.argmax(probs))
            confidence = float(probs[0][label_id])

            results.append({
                "id": comment.id,
                "body": text,
                "sentiment": LABEL_MAP.get(label_id, "unknown"),
                "sentiment_score": round(confidence, 4)
            })

        current_memory_mb = process.memory_info().rss / 1024 / 1024

        # Platform-specific peak memory
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
        print("Error:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
