from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import time
import psutil
from fastapi.responses import JSONResponse
import traceback
import platform


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
    try:
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024 

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
