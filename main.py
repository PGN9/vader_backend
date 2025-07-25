from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import os
import time
import psutil
import traceback
import platform
import json


request_count = 0
start_time = time.time()

app = FastAPI()

emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
sentiment_analyzer = SentimentIntensityAnalyzer()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

def get_size_in_kb(data):
    return len(data.encode('utf-8')) / 1024  # size in KB

@app.get("/")
def root():
    return {"message": "model backend is running."}


@app.post("/predict")
def predict_sentiment(request: CommentsRequest):
    try:
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024 
        # Track input size
        input_json = json.dumps(request.model_dump()) if hasattr(request, "json") else str(request)
        total_data_size_kb = get_size_in_kb(input_json)

        results = []
        for comment in request.comments:
            body = comment.body

            # VADER sentiment
            scores = sentiment_analyzer.polarity_scores(body)
            sentiment = (
                "positive" if scores["compound"] > 0.05 else
                "negative" if scores["compound"] < -0.05 else
                "neutral"
            )

            # Emotion classification via Hugging Face
            emotion_results = emotion_classifier(body)[0]  # Get all emotion scores
            top_emotion = max(emotion_results, key=lambda x: x["score"])["label"]
            emotion_scores = {res["label"]: round(res["score"], 4) for res in emotion_results}

            results.append({
                "id": comment.id,
                "body": body,
                "sentiment": sentiment,
                "sentiment_score": round(scores["compound"], 4),
                "emotion": top_emotion,
                "emotion_scores": emotion_scores
            })

        # Memory usage check
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        # Peak memory usage (cross-platform)
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_kb / 1024
        elif platform.system() == "Darwin":  # macOS
            import resource
            peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_bytes / 1024 / 1024
        else:
            peak_memory_mb = current_memory_mb  # fallback

        
        return_data = {
            "model_used": "vader",
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2)
        }
        # add data size info
        total_return_size_kb = get_size_in_kb(json.dumps(return_data))
        return_data["total_data_size_kb"] = round(total_data_size_kb, 2)
        return_data["total_return_size_kb"] = round(total_return_size_kb, 2)
        return return_data

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
