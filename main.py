from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time
import psutil
from fastapi.responses import JSONResponse
import traceback
import tracemalloc


request_count = 0
start_time = time.time()
# Get process for memory monitoring
tracemalloc.start()
process = psutil.Process(os.getpid())
initial_memory_mb = process.memory_info().rss / 1024 / 1024

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
    try:        
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

        # Memory usage check
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        current, peak = tracemalloc.get_traced_memory()
        peak_memory_mb = peak / 1024 / 1024
        tracemalloc.stop()

        return {
            "results": results,
            "proxy_memory_initial_mb": round(initial_memory_mb, 2),
            "proxy_memory_usage_mb": round(current_memory_mb, 2),
            "proxy_memory_peak_mb": round(peak_memory_mb, 2)
        }

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
