from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import requests
import os
import json
import gc
import logging
import psutil
import time

# === Config ===
MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
ONNX_MODEL_URL = "https://huggingface.co/Ndi2020/bhadresh-emotion-onnx/resolve/main/model-quant.onnx"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
BATCH_SIZE = 32         # Safe for <=512MB RAM
THRESHOLD = 0.3
TIMEOUT_SECONDS = 300   # Render hard timeout is 300s max

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-model")

# === Ensure Model Exists ===
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        logger.info("Downloading quantized ONNX model...")
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
        response = requests.get(ONNX_MODEL_URL, timeout=60)
        response.raise_for_status()
        with open(ONNX_MODEL_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Download complete.")

download_model()

# === Load Tokenizer and Model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

# === FastAPI App ===
app = FastAPI()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

@app.get("/")
def health_check():
    return {
        "status": "backend is alive",
        "message": "Emotion ONNX model is running."
    }


@app.post("/predict")
def predict(request: CommentsRequest):
    try:
        # === Memory usage before ===
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        logger.info(f"Received {len(request.comments)} comments for prediction.")

        # === Measure request data size ===
        request_json = request.model_dump()
        request_bytes = json.dumps(request_json).encode("utf-8")
        request_size_kb = len(request_bytes) / 1024

        texts = [c.body for c in request.comments]
        ids = [c.id for c in request.comments]

        results = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_ids = ids[i:i + BATCH_SIZE]

            logger.info(f"Processing batch {i // BATCH_SIZE + 1} - size {len(batch_texts)}")

            inputs = tokenizer(
                batch_texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512
            )
            logger.info("Tokenization complete.")

            onnx_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }

            logits = session.run(None, onnx_inputs)[0]
            logger.info("ONNX model inference complete.")

            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            logger.info("Softmax probabilities computed.")

            for j, p in enumerate(probs):
                emotion_list = [label for k, label in enumerate(LABELS) if p[k] > THRESHOLD]
                emotion_scores = [{"label": label, "score": round(float(p[k]), 4)} for k, label in enumerate(LABELS)]

                results.append({
                    "id": batch_ids[j],
                    "emotions": emotion_list,
                    "emotion_scores": emotion_scores
                })

            logger.info(f"Batch {i // BATCH_SIZE + 1} processed. Results so far: {len(results)}")
            del batch_texts, batch_ids, inputs, onnx_inputs, logits, probs
            gc.collect()

        # === Measure return data size ===
        response_payload = {
            "model_used": MODEL_ID,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "total_data_size_kb": round(request_size_kb, 2),
            "total_return_size_kb": round(len(json.dumps({"results": results}).encode("utf-8")) / 1024, 2),
            "results": results
        }

        logger.info(f"Final response: {len(results)} results, response size: {response_payload['total_return_size_kb']} KB")

        return response_payload

    except Exception as e:
        logger.error("Exception during prediction", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
