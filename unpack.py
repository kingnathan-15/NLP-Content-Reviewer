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

# --- Model and Configuration Loading ---
print("=" * 60)
print("ATTEMPTING TO LOAD MODELS...")
print("=" * 60)

lda = None
vectorizer = None
sa_model = None
sa_tfidf = None
topic_labels = {}

try:
    # Load LDA Topic Model components
    print("Loading LDA model...")
    lda = joblib.load('./model/topic_model_lda.pkl')
    print("✓ LDA model loaded successfully")
    
    print("Loading vectorizer...")
    vectorizer = joblib.load('./model/topic_vectorizer.pkl')
    print("✓ Vectorizer loaded successfully")

    # Load Sentiment Analysis (SA) components
    print("Loading sentiment model...")
    sa_model = joblib.load('./model/sa_model.pkl')
    print("✓ Sentiment model loaded successfully")
    
    print("Loading sentiment vectorizer...")
    sa_tfidf = joblib.load('./model/sa_vectorizer.pkl')
    print("✓ Sentiment vectorizer loaded successfully")
    
    print("Loading topic labels...")
    with open('./model/topic_labels.json', 'r', encoding='utf-8') as f:
        topic_labels = json.load(f)
    topic_labels = {int(k): v for k, v in topic_labels.items()}
    print("✓ Topic labels loaded successfully")
    print(f"  Available topics: {list(topic_labels.keys())}")
    
    print("=" * 60)
    print("ALL MODELS LOADED SUCCESSFULLY!")
    print("=" * 60)
    
except FileNotFoundError as e:
    print("=" * 60)
    print(f"❌ ERROR: Model file not found: {e}")
    print("=" * 60)
    print("\nPlease ensure the following files exist:")
    print("  - ./model/topic_model_lda.pkl")
    print("  - ./model/topic_vectorizer.pkl")
    print("  - ./model/sa_model.pkl")
    print("  - ./model/sa_vectorizer.pkl")
    print("  - ./model/topic_labels.json")
    
except Exception as e:
    print("=" * 60)
    print(f"❌ ERROR loading models: {e}")
    print(f"Error type: {type(e).__name__}")
    print("=" * 60)
    import traceback
    traceback.print_exc()

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
        print("❌ Prediction error: Sentiment models not loaded.")
        return 1, 0.0 # Default to Neutral with 0 confidence
        
    try:
        # Clean and vectorize text
        cleaned_text = clean_text_for_sa(text)
        
        # Check for empty cleaned text
        if not cleaned_text.strip():
            print("⚠️ Warning: Cleaned text is empty")
            return 1, 0.0

        text_vector = tfidf.transform([cleaned_text])

        # Predict and get confidence
        pred = best_model.predict(text_vector)[0]
        proba = best_model.predict_proba(text_vector).max()

        print(f"✓ Sentiment prediction RAW: {pred} (type: {type(pred)}), confidence: {proba:.3f}")
        
        # Convert prediction to int to ensure consistency
        pred_int = int(pred)
        print(f"✓ Sentiment prediction INT: {pred_int}")
        
        return pred_int, round(proba, 3)
        
    except Exception as e:
        print(f"❌ Prediction error in get_prediction_and_confidence: {str(e)}")
        import traceback
        traceback.print_exc()
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
    print("\n" + "=" * 60)
    print(f"ANALYZING COMMENT: {comment[:50]}...")
    print("=" * 60)
    
    # Check if models were loaded
    if lda is None or vectorizer is None or sa_model is None or sa_tfidf is None:
        print("❌ Analysis error: One or more NLP models failed to load.")
        print(f"  lda: {lda is not None}")
        print(f"  vectorizer: {vectorizer is not None}")
        print(f"  sa_model: {sa_model is not None}")
        print(f"  sa_tfidf: {sa_tfidf is not None}")
        return None
        
    try:
        # 1. Topic Analysis
        print("\n1. Topic Analysis:")
        cleaned = clean_text(comment)
        
        if not cleaned:
            print("❌ Error: Cleaned comment is empty")
            return None 
        
        print(f"  Cleaned text: {cleaned[:50]}...")

        X = vectorizer.transform([cleaned])
        topic_probs = lda.transform(X)[0]
        topic_id = int(np.argmax(topic_probs))
        topic_confidence = round(topic_probs.max(), 3)
        
        print(f"  ✓ Topic ID: {topic_id}")
        print(f"  ✓ Topic confidence: {topic_confidence}")
        print(f"  ✓ Topic label: {topic_labels.get(topic_id, 'Unknown')}")

        # 2. Sentiment Analysis
        print("\n2. Sentiment Analysis:")
        sentiment_int, sentiment_confidence = get_prediction_and_confidence(
            comment, sa_model, sa_tfidf
        )
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print(f"Result: ({topic_id}, {sentiment_int}, {topic_confidence}, {sentiment_confidence})")
        print("=" * 60)
        
        # Return the expected 4-item tuple
        return (topic_id, sentiment_int, topic_confidence, sentiment_confidence)
    
    except Exception as e:
        print(f"❌ Critical error in get_topic_and_sentiment_for_comment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None