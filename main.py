from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time

request_count = 0
start_time = time.time()

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

# New Pydantic model to accept list of comments with id and body
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
    return results

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
