from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time
import psutil  # <-- NEW IMPORT

request_count = 0
start_time = time.time()

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]

@app.get("/")
def root():
    return {"message": "vader backend is running."}

@app.post("/predict")
def predict_sentiment(request: CommentsRequest):
    # debug
    #print("Received comments:", request.comments)

    results = []
    for comment in request.comments:
        body = comment.body
        scores = analyzer.polarity_scores(body)
        sentiment = (
            "positive" if scores["compound"] > 0.05 else
            "negative" if scores["compound"] < -0.05 else
            "neutral"
        )
        results.append({
            "id": comment.id,
            "body": body,
            "sentiment": sentiment,
            "sentiment_score": scores["compound"]
        })

    # Memory usage in MB
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage_mb = mem_info.rss / 1024 / 1024

    return {
        "results": results,
        "memory_usage_mb": round(memory_usage_mb, 2)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
