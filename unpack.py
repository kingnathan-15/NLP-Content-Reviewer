import re
import json
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK Resource Downloads ---
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) 
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

# Initialize global resources
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: Could not load stopwords: {e}")
    stop_words = set()

stemmer = PorterStemmer()

# --- Sentiment String to Integer Mapping ---
SENTIMENT_STRING_TO_INT = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

# --- Model and Configuration Loading ---
try:
    # Load LDA Topic Model components
    lda = joblib.load('./model/topic_model_lda.pkl')
    vectorizer = joblib.load('./model/topic_vectorizer.pkl')

    # Load Sentiment Analysis (SA) components
    sa_model = joblib.load('./model/sa_model.pkl')
    sa_tfidf = joblib.load('./model/sa_vectorizer.pkl')
    
    with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
    topic_labels = {int(k): v for k, v in topic_labels.items()}
    
    print("✓ All models loaded successfully")
    
except FileNotFoundError as e:
    print(f"Error loading model files: {e}. Ensure all model files are in the './model' directory.")
    lda, vectorizer, sa_model, sa_tfidf, topic_labels = None, None, None, None, {}

# --- Helper Functions ---

def clean_text_for_sa(text):
    """Clean text for sentiment analysis."""
    try:
        # Lowercase conversion
        text = text.lower()

        # Remove special characters/numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize using simple split (bypassing NLTK punkt_tab)
        tokens = text.split() 

        # Remove stopwords
        filtered = [word for word in tokens if word not in stop_words]

        # Apply stemming
        stemmed = [stemmer.stem(word) for word in filtered]

        return " ".join(stemmed)
    except Exception as e:
        print(f"Error in clean_text_for_sa: {e}")
        return ""

def get_prediction_and_confidence(text, best_model, tfidf):
    """Predicts sentiment for a given text and returns prediction and confidence."""
    # Check if models were loaded
    if best_model is None or tfidf is None:
        print("Prediction error: Sentiment models not loaded.")
        return 1, 0.0 # Default to Neutral with 0 confidence
        
    try:
        # Clean and vectorize text
        cleaned_text = clean_text_for_sa(text)
        
        # Check for empty cleaned text
        if not cleaned_text.strip():
            return 1, 0.0

        text_vector = tfidf.transform([cleaned_text])

        # Predict and get confidence
        pred = best_model.predict(text_vector)[0]
        proba = best_model.predict_proba(text_vector).max()

        # Convert string prediction to int
        if isinstance(pred, str):
            pred_lower = pred.lower().strip()
            pred_int = SENTIMENT_STRING_TO_INT.get(pred_lower, 1)  # Default to neutral if unknown
        elif isinstance(pred, (int, np.integer)):
            pred_int = int(pred)
        else:
            # Fallback for unexpected types
            pred_int = 1
        
        return pred_int, round(float(proba), 3)
        
    except Exception as e:
        print(f"Prediction error in get_prediction_and_confidence: {str(e)}")
        return 1, 0.0

# --- Main Analysis Function ---

def clean_text(text):
    """Clean text for topic analysis."""
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
    # Check if models were loaded
    if lda is None or vectorizer is None or sa_model is None or sa_tfidf is None:
        print("Analysis error: One or more NLP models failed to load.")
        return None
        
    try:
        # 1. Topic Analysis
        cleaned = clean_text(comment)
        
        if not cleaned:
            return None 

        X = vectorizer.transform([cleaned])
        topic_probs = lda.transform(X)[0]
        topic_id = int(np.argmax(topic_probs))
        topic_confidence = round(float(topic_probs.max()), 3)

        # 2. Sentiment Analysis
        sentiment_int, sentiment_confidence = get_prediction_and_confidence(
            comment, sa_model, sa_tfidf
        )
        
        # Return the expected 4-item tuple
        return (topic_id, sentiment_int, topic_confidence, sentiment_confidence)
    
    except Exception as e:
        print(f"Critical error in get_topic_and_sentiment_for_comment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None