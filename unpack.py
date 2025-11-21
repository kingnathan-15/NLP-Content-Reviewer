import re
import json
import joblib
from langdetect import detect, LangDetectException
from translate import Translator
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

lda = joblib.load('./model/topic_model_lda.pkl')
vectorizer = joblib.load('./model/topic_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
    topic_labels = json.load(f)
topic_labels = {int(k): v for k, v in topic_labels.items()}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_topic_and_sentiment_for_comment(comment: str, top_n_words: int = 10):
    cleaned = clean_text(comment)
    X = vectorizer.transform([cleaned])
    topic_probs = lda.transform(X)[0]
    topic_id = int(np.argmax(topic_probs))

    feature_names = vectorizer.get_feature_names_out()
    topic = lda.components_[topic_id]
    top_idx = topic.argsort()[-top_n_words:][::-1]
    top_words = [feature_names[i] for i in top_idx]

    sentiment_add_result, sentiment_confidence = sentiment_add(comment)

    if sentiment_add_result is None:
        sentiment_add_result = 1  # default to neutral

    return {
        'topic_id': topic_id,
        'topic_label': topic_labels.get(topic_id, 'Unlabeled'),
        'topic_probability': float(topic_probs[topic_id]),
        'top_words': top_words,
        'sentiment': sentiment_add_result,
        'sentiment_confidence': sentiment_confidence
    }

def sentiment_add(text):
    # Load persisted artifacts
    best_model = joblib.load('./model/sentiment_classifier.pkl')
    tfidf = joblib.load('./model/topic_vectorizer_using_tfidf.pkl')

    try:
        # Clean and vectorize text
        cleaned_text = clean_text_for_sa(text)
        text_vector = tfidf.transform([cleaned_text])

        # Predict and get confidence
        pred = best_model.predict(text_vector)[0]
        proba = best_model.predict_proba(text_vector).max()

        return pred, round(proba, 3)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, 0.0

def clean_text_for_sa(text):
    # Lowercase conversion
    text = text.lower()

    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]

    # Apply stemming
    stemmed = [stemmer.stem(word) for word in filtered]

    return " ".join(stemmed)

def english_check(text):
    if not text:
        return True
    try:
        return detect(text) == 'en'
    except:
        return True

def detect_lang(text):
    if not text:
        return 'en'
    try:
        return detect(text)
    except:
        return 'en'

def translate_to_english(text):
    if not text:
        return ""
    try:
        lang1 = detect_lang(text)
        if lang1 == 'en':
            return text
        translator = Translator(from_lang=lang1, to_lang="en")
        translation = translator.translate(text)
        return translation
    except LangDetectException:
        return ""