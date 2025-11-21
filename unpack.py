import re
import json
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK Resource Downloads (Addressing the Punkt_tab Error) ---
# Ensure necessary NLTK packages are downloaded
try:
    nltk.download('stopwords')
    # FIX: Explicitly download 'punkt' and 'punkt_tab' as requested by the error message.
    # We wrap this in a try/except because the download itself can fail in restricted environments.
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    print(f"Warning: NLTK download failed, possibly due to environment restrictions: {e}")


# --- Model and Configuration Loading ---
try:
    # Load LDA Topic Model components
    lda = joblib.load('./model/topic_model_lda.pkl')
    vectorizer = joblib.load('./model/topic_vectorizer.pkl')

    # Load Sentiment Analysis (SA) components (assuming they are used elsewhere)
    sa_model = joblib.load('./model/sa_model.pkl')
    sa_tfidf = joblib.load('./model/sa_vectorizer.pkl')
    
    with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
    topic_labels = {int(k): v for k, v in topic_labels.items()}
    
except FileNotFoundError as e:
    print(f"Error loading model files: {e}. Ensure all model files are in the './model' directory.")
    # Assign dummy variables if models fail to load, to prevent subsequent NameErrors
    lda, vectorizer, sa_model, sa_tfidf, topic_labels = None, None, None, None, {}

# Initialize global resources
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Helper Functions (Sentiment Analysis components) ---

def clean_text_for_sa(text):
    # Lowercase conversion
    text = text.lower()

    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and remove stopwords
    # Added a check for NLTK resource availability
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        print("Warning: NLTK Punkt resource missing. Falling back to simple split.")
        tokens = text.split()

    filtered = [word for word in tokens if word not in stop_words]

    # Apply stemming
    stemmed = [stemmer.stem(word) for word in filtered]

    return " ".join(stemmed)

def get_prediction_and_confidence(text, best_model, tfidf):
    """Predicts sentiment for a given text and returns prediction and confidence."""
    # Check if models were loaded
    if best_model is None or tfidf is None:
        print("Prediction error: Sentiment models not loaded.")
        return 1, 0.0 # Default to Neutral with 0 confidence
        
    try:
        # Clean and vectorize text
        cleaned_text = clean_text_for_sa(text)
        text_vector = tfidf.transform([cleaned_text])

        # Predict and get confidence
        pred = best_model.predict(text_vector)[0]
        # Use np.max on probabilities to get confidence
        proba = best_model.predict_proba(text_vector).max()

        return pred, round(proba, 3)
    except Exception as e:
        print(f"Prediction error in get_prediction_and_confidence: {str(e)}")
        # Return a safe default tuple on failure
        return 1, 0.0 # Default to Neutral with 0 confidence


# --- Main Analysis Function ---

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_topic_and_sentiment_for_comment(comment: str, top_n_words: int = 10):
    """
    Analyzes a comment to determine its dominant topic and sentiment.
    
    Returns:
        tuple: (topic_id, sentiment_int, topic_confidence, sentiment_confidence)
        None: If any critical error occurs during analysis.
    """
    # Check if models were loaded (Handles FileNotFoundError from above)
    if lda is None or vectorizer is None or sa_model is None or sa_tfidf is None:
        print("Analysis error: One or more NLP models failed to load.")
        return None
        
    try:
        # 1. Topic Analysis
        cleaned = clean_text(comment)
        if not cleaned:
            # Handle empty comments gracefully
            return None 

        X = vectorizer.transform([cleaned])
        topic_probs = lda.transform(X)[0]
        topic_id = int(np.argmax(topic_probs))
        topic_confidence = round(topic_probs.max(), 3)

        # 2. Sentiment Analysis
        sentiment_int, sentiment_confidence = get_prediction_and_confidence(
            comment, sa_model, sa_tfidf
        )
        
        # Return the expected 4-item tuple
        return (topic_id, sentiment_int, topic_confidence, sentiment_confidence)
    
    except Exception as e:
        # Catch any residual analysis error and return None
        print(f"Critical error in get_topic_and_sentiment_for_comment: {str(e)}")
        return None