from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import requests
import os
import traceback
import json

MODEL_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
ONNX_MODEL_URL = "https://huggingface.co/Ndi2020/bhadresh-emotion-onnx/resolve/main/model-quant.onnx"
ONNX_MODEL_PATH = "./onnx_model/model-quant.onnx"

LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Ensure model is available
def download_model():
    if not os.path.exists(ONNX_MODEL_PATH):
        print("Downloading quantized ONNX model...")
        os.makedirs(os.path.dirname(ONNX_MODEL_PATH), exist_ok=True)
        with open(ONNX_MODEL_PATH, "wb") as f:
            f.write(requests.get(ONNX_MODEL_URL).content)
        print("Download complete.")

download_model()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
session = ort.InferenceSession(ONNX_MODEL_PATH)

# Define API
app = FastAPI()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

@app.get("/")
def root():
    return {"message": "Emotion ONNX model is running."}

@app.post("/predict")
def predict(request: CommentsRequest):
    try:
        texts = [c.body for c in request.comments]
        ids = [c.id for c in request.comments]
        inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True, max_length=512)

        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        logits = session.run(None, onnx_inputs)[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        THRESHOLD = 0.4  # adjust this as needed

        results = []
        for i, p in enumerate(probs):
            score_dict = {label: round(float(p[j]), 4) for j, label in enumerate(LABELS)}

            # Pick all emotions with score above threshold
            emotion_list = [label for j, label in enumerate(LABELS) if p[j] > THRESHOLD]

            results.append({
                "id": ids[i],
                "body": texts[i],
                "emotions": emotion_list,              # now a list, not just one
                "emotion_scores": score_dict
            })


        return {"model": MODEL_ID, "results": results}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
