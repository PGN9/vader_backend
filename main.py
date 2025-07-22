from fastapi import FastAPI
from pydantic import BaseModel
# a tool to analyze sentiment (positive / negative / neutral) of text.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

# define a data model called TextRequest (which extends BaseModel)
# make sure the incoming JSON data has a field called text and that text is a string.
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    scores = analyzer.polarity_scores(request.text)
    sentiment = "positive" if scores['compound'] > 0.05 else \
                "negative" if scores['compound'] < -0.05 else "neutral"
    return {"sentiment": sentiment, "scores": scores}
