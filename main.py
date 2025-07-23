from fastapi import FastAPI
from pydantic import BaseModel
# a tool to analyze sentiment (positive / negative / neutral) of text.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

# define a data model called TextRequest (which extends BaseModel)
# make sure the incoming JSON data has a field called text and that text is a string.
class TextListRequest(BaseModel):
    text: List[str]

@app.post("/predict")
def predict_sentiment(request: TextListRequest):
    results = []
    for text in request.texts:
        scores = analyzer.polarity_scores(text)
        sentiment = (
            "positive" if scores["compound"] > 0.05 else
            "negative" if scores["compound"] < -0.05 else
            "neutral"
        )
        results.append({
            "text": text,
            "sentiment": sentiment,
            "compound": scores["compound"]
        })
    return results
