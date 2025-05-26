# NewsGuardAI-A-Real-Time-Misinformation-Detection-Trustworthiness-Scoring-Platform
NewsGuardAI is a web-based application that helps users analyze the trustworthiness, bias, and factual accuracy of online news articles, social media posts, and headlines in real-time. 
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Optional
import uvicorn
import requests

app = FastAPI(title="NewsGuardAI - Misinformation Detection")

# Load Hugging Face pipelines
text_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
summarizer = pipeline("summarization")
zero_shot_classifier = pipeline("zero-shot-classification")
sentence_similarity = pipeline("sentence-similarity")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")  # English to Spanish for demo

# Example fact-checking labels
CATEGORIES = ["fake news", "reliable", "bias", "neutral"]

class ArticleInput(BaseModel):
    url: Optional[str] = None
    content: Optional[str] = None

@app.post("/analyze")
async def analyze_article(input: ArticleInput):
    if not input.content and not input.url:
        raise HTTPException(status_code=400, detail="Provide article content or URL")

    content = input.content

    # If URL is provided, fetch the article content
    if input.url:
        try:
            response = requests.get(input.url)
            content = response.text[:3000]  # naive: take first 3000 chars
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to fetch article")

    # Run classification
    label_result = text_classifier(content[:512])

    # Summarization
    summary = summarizer(content[:1000])[0]['summary_text']

    # Zero-shot classification to check topic
    topic_result = zero_shot_classifier(content[:512], candidate_labels=CATEGORIES)

    # Translate for demo purposes (e.g., English to Spanish)
    translation = translator(content[:1000])[0]['translation_text']

    return {
        "classification": label_result,
        "summary": summary,
        "topic_score": topic_result,
        "translated": translation,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


