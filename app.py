from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import warnings
from unpack import get_topic_and_sentiment_for_comment

from database import init_db, insert_review, get_all_reviews, populate_original_data, sentiment_retrieval, limit_reviews
warnings.filterwarnings('ignore')

app = Flask(__name__)

with app.app_context():
    # this runs once at startup
    init_db() 
    populate_original_data()

@app.route('/')
def home():
    return render_template('interface.html')

@app.route('/admin')
def admin():
    topic_labels = {
        0: "Need for additional guidance from lecturer during labs",
        1: "Practical applications of taught content",
        2: "Miscellaneous comments regarding CATs and explanations of content",
        3: "Negative opinions related to the amount of content within the slides used in teaching"
    }
    
    details = limit_reviews()
    
    return render_template('admin.html', reviews=details, topic_labels=topic_labels)

@app.route("/api/sentiments")
def get_sentiments():
    data = sentiment_retrieval()
    print("Sentiment data retrieved:", data)
    return jsonify(data)

@app.route("/api/topics")
def get_topics():
    details = get_all_reviews()
    topic_counts = pd.Series([review['topic'] for review in details]).value_counts().to_dict()
    
    topic_labels = {
        0: "Need for additional guidance from lecturer during labs",
        1: "Practical applications of taught content",
        2: "Miscellaneous comments regarding CATs and explanations of content",
        3: "Negative opinions related to the amount of content within the slides used in teaching"
    }
    
    topics_data = {
        'labels': [topic_labels.get(topic_id, "Unknown Topic") for topic_id in topic_counts.keys()],
        'counts': list(topic_counts.values())
    }
    
    print("Topic data retrieved:", topics_data)
    return jsonify(topics_data)


@app.route('/submit-review', methods=['POST'])
def submit():
    if request.method == 'POST':
        review = request.form['text-input']
        result = get_topic_and_sentiment_for_comment(review)
        print("Review processed:", result)
        insert_review(review, result['topic_id'], result['sentiment'])
        
    return render_template('next_user.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
