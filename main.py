from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import psutil
from fastapi.responses import JSONResponse
import traceback
import platform
import json
torch.set_num_threads(1)

# Load Hugging Face model
MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Label mapping based on HuggingFace model card
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

app = FastAPI()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data: str) -> float:
    return len(data.encode('utf-8')) / 1024

@app.get("/")
def root():
    return {"message": "Sentiment Classification using lxyuan/distilbert-base-multilingual-cased-sentiments-student"}

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

            # Hugging Face classification
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                label_id = torch.argmax(probs).item()
                confidence = probs[0][label_id].item()

            sentiment = LABEL_MAP.get(label_id, "unknown")

            results.append({
                "id": comment.id,
                "body": text,
                "sentiment": sentiment,
                "sentiment_score": round(confidence, 4)
            })

        current_memory_mb = process.memory_info().rss / 1024 / 1024

        # Platform-specific memory peak check
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_kb / 1024
        elif platform.system() == "Darwin":
            import resource
            peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_bytes / 1024 / 1024
        else:
            peak_memory_mb = current_memory_mb

        return_data = {
            "model_used": MODEL_NAME,
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2),
            "total_data_size_kb": round(total_data_size_kb, 2),
            "total_return_size_kb": round(get_size_in_kb(json.dumps(results)), 2)
        }

        return return_data

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

