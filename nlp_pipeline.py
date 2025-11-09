# nlp_pipeline.py
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import google.generativeai as genai
import os

# ensure punkt etc loaded (for NLTK tokenizers if needed)
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def gemini_emotion_analysis(text: str) -> dict:
    """
    Uses Gemini (Google Generative AI) to analyze emotions in text.
    Returns a dict like: {'Happy': 0.8, 'Sad': 0.1, ...}
    """
    if not text.strip():
        return {}

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Analyze the following text and return a JSON dictionary of emotions with confidence scores between 0 and 1.
    The emotions should include: Happy, Sad, Angry, Fearful, Surprised, Neutral, and Curious.
    Text: "{text}"
    """

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Try to safely extract dictionary from the response
        import json, re
        json_str = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group())
            # Normalize keys (capitalize)
            emo = {k.capitalize(): float(v) for k, v in data.items() if isinstance(v, (int, float))}
            return emo
        else:
            print("Gemini response not in JSON format:", raw_text)
            return {}
    except Exception as e:
        print("Gemini emotion analysis failed:", e)
        return {}

class EmotionSentimentPipeline:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self._load_models()

    def _load_models(self):
        try:
            self.emotion_pipe = pipeline("text-classification", model=self.model_name, return_all_scores=True)
            self.sentiment_pipe = pipeline("sentiment-analysis")
            self.use_transformer = True
        except Exception as e:
            print("Transformer pipelines failed to load; falling back to Gemini + VADER.", e)
            self.use_transformer = False
            self.vader = SentimentIntensityAnalyzer()

    def predict(self, text):
        text = text or ""
        if self.use_transformer and text.strip():
            try:
                emo_scores = self.emotion_pipe(text)[0]
                emo = {item['label']: float(item['score']) for item in emo_scores}
            except Exception as e:
                print("Emotion transform failed:", e)
                emo = gemini_emotion_analysis(text)
            try:
                sent_scores = self.sentiment_pipe(text)
                sent = {sent_scores[0]['label']: float(sent_scores[0]['score'])}
            except Exception as e:
                print("Sentiment transform failed:", e)
                vs = self.vader.polarity_scores(text)
                compound = vs['compound']
                lab = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
                sent = {lab: float(abs(compound))}
            return emo, sent
        else:
            emo = gemini_emotion_analysis(text)
            vs = self.vader.polarity_scores(text)
            compound = vs['compound']
            lab = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
            sent = {lab: float(abs(compound))}
            return emo, sent
