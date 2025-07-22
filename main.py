from fastapi import FastAPI
from pydantic import BaseModel
# a tool to analyze sentiment (positive / negative / neutral) of text.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import os

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

# define a data model called TextRequest (which extends BaseModel)
# make sure the incoming JSON data has a field called text and that text is a string.
class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "VADER backend is running."}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    scores = analyzer.polarity_scores(request.text)
    sentiment = "positive" if scores['compound'] > 0.05 else \
                "negative" if scores['compound'] < -0.05 else "neutral"
    return {"sentiment": sentiment, "scores": scores}


if __name__ == "__main__":
    import uvicorn

    # Use PORT from Render, default to 8000 for local
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)