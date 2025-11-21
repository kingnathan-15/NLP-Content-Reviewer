import sqlite3
import pandas as pd

DB_NAME = "reviews.db"

SENTIMENT_MAP = {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        Drop TABLE IF EXISTS reviews;
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review TEXT NOT NULL,
        topic INTEGER NOT NULL,
        sentiment INTEGER DEFAULT 1
        );
''')
    print("Database initialized with 'reviews' table.")
    conn.commit()
    conn.close()

def insert_review(review, topic, sentiment):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO reviews (review, topic, sentiment) VALUES (?, ?, ?)', (review, topic, sentiment))
    conn.commit()
    conn.close()

def get_all_reviews():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews')
    rows = cursor.fetchall()
    conn.close()
    return rows

def limit_reviews(limit=10):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews ORDER BY id DESC LIMIT 10;')
    rows = cursor.fetchall()
    conn.close()
    return rows

def populate_original_data():
    conn = get_db()
    cursor = conn.cursor()
    
    data = pd.read_csv('./data/finalized_reviews.csv', encoding='utf-8')
    count = 0
    for _, row in data.iterrows():
        review_text = row['future_recommendation']
    
        if pd.notna(review_text):
            sentiment_str = str(row['sentiment_initial']).lower().strip()
            sentiment_int = SENTIMENT_MAP.get(sentiment_str, 1)  # default to neutral

            cursor.execute(
                '''
                INSERT INTO reviews (review, topic, sentiment)
                VALUES (?, ?, ?)
                ''',
                (review_text, row['dominant_topic'], sentiment_int)
            )
            count += 1

    print("Original data populated into the database. Total records inserted:", count)
    conn.commit()
    conn.close()


def sentiment_retrieval():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT sentiment, COUNT(*) as count FROM reviews GROUP BY sentiment;')
    rows = cursor.fetchall()
    conn.close()
    
    int_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    counts = {'positive': 0, 'neutral': 0, 'negative': 0}

    for sentiment_int, count in rows:
        label = int_to_label.get(sentiment_int)
        if label:
            counts[label] = count


    return counts