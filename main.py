from fastapi import FastAPI
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

# Load quantized ONNX model
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"  # Define local path for the model file
ONNX_MODEL_URL = "https://huggingface.co/dakyswr/lxyuan-distilbert-sentiment-onnx/resolve/main/model-quant.onnx"

def download_model():
    model_path = ONNX_MODEL_PATH
    if not os.path.exists(model_path):
        print("Downloading ONNX model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(requests.get(ONNX_MODEL_URL).content)
        print("Download complete.")

download_model()


# Load tokenizer
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
        print("Received /predict request")
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024

        input_json = json.dumps(request.dict())
        total_data_size_kb = get_size_in_kb(input_json)

        # --- Batch inference starts here ---
        texts = [comment.body for comment in request.comments]
        ids = [comment.id for comment in request.comments]

        inputs = tokenizer(
            texts, return_tensors="np", truncation=True,
            padding=True, max_length=512
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        import time
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
        # --- End of batch inference ---

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
