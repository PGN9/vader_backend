from fastapi import FastAPI, Request
from pydantic import BaseModel
# a tool to analyze sentiment (positive / negative / neutral) of text.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os
import time

request_count = 0
start_time = time.time()

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

# define a data model called TextRequest (which extends BaseModel)
# make sure the incoming JSON data has a field called text and that text is a string.
class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "vader backend is running."}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    scores = analyzer.polarity_scores(request.text)
    sentiment = "positive" if scores['compound'] > 0.05 else \
                "negative" if scores['compound'] < -0.05 else "neutral"
    return {"sentiment": sentiment, "scores": scores}

@app.middleware("http")
async def count_requests(request: Request, call_next):
    global request_count, start_time
    request_count += 1
    response = await call_next(request)

    elapsed = time.time() - start_time
    if elapsed > 10:
        print(f"[Request Stats] Last 10s - Requests: {request_count}, RPS: {request_count / elapsed:.2f}", flush=True)
        request_count = 0
        start_time = time.time()

    return response


# Fix to help Render to find a port
if __name__ == "__main__":
    import uvicorn

    # Use PORT from Render, default to 8000 for local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)