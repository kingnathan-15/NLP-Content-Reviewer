import streamlit as st
import pandas as pd
import numpy as np
# Assuming database.py and unpack.py are available in the directory
from database import init_db, populate_original_data, insert_review, limit_reviews, sentiment_retrieval, get_all_reviews
from unpack import get_topic_and_sentiment_for_comment

# --- Configuration and Initialization ---

# Set wide layout for better dashboard viewing
st.set_page_config(layout="wide", page_title="Course Review Analysis")

TOPIC_LABELS = {
    0: "Need for additional guidance from lecturer during labs",
    1: "Practical applications of taught content",
    2: "Miscellaneous comments regarding CATs and explanations of content",
    3: "Negative opinions related to the amount of content within the slides used in teaching"
}
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Use Streamlit's cache to initialize the database only once.
@st.cache_resource
def setup_database():
    """Initializes the database and populates initial data."""
    init_db()
    populate_original_data()
    return True

# Run setup
setup_database()


# --- Data Retrieval Functions (Using Cache) ---

@st.cache_data(show_spinner="Fetching recent reviews...")
def get_recent_reviews_df():
    """Fetches limited reviews and converts them to a DataFrame."""
    # limit_reviews() returns sqlite3.Row objects. We must convert them to dicts
    # before passing to Pandas to ensure column names are correctly mapped, 
    # which fixes the original KeyError: 'topic'.
    reviews_rows = limit_reviews()
    
    # CRUCIAL FIX for KeyError: 'topic'
    df = pd.DataFrame([dict(row) for row in reviews_rows])
    
    # Map the integer topic ID to the descriptive label
    df['Topic'] = df['topic'].apply(lambda t: TOPIC_LABELS.get(t, "Unknown"))
    df['Sentiment'] = df['sentiment'].apply(lambda s: SENTIMENT_LABELS.get(s, "Unknown"))
    
    # Select and rename columns for display
    df_display = df[['review', 'Topic', 'Sentiment']]
    df_display.columns = ['Feedback', 'Topic', 'Sentiment']
    
    return df_display

@st.cache_data(show_spinner="Calculating sentiment statistics...")
def get_sentiment_metrics():
    """Fetches sentiment counts."""
    # sentiment_retrieval() returns a dictionary like {'positive': X, 'neutral': Y, 'negative': Z}
    return sentiment_retrieval()

@st.cache_data(show_spinner="Analyzing topic distribution...")
def get_topic_distribution():
    """Calculates topic counts for all reviews."""
    all_reviews = get_all_reviews()
    
    # CRUCIAL: Convert sqlite3.Row objects to dicts
    topic_ids = [dict(row)['topic'] for row in all_reviews]
    
    topic_counts = pd.Series(topic_ids).value_counts().sort_index()
    
    df_topics = pd.DataFrame({
        'Topic ID': topic_counts.index,
        'Count': topic_counts.values
    })
    
    # Map ID to label for chart readability
    df_topics['Topic'] = df_topics['Topic ID'].apply(lambda t: TOPIC_LABELS.get(t, "Unknown Topic"))
    
    return df_topics


# --- View Functions ---

def show_review_form():
    """Displays the user review submission form."""
    st.title("📝 Course Evaluation Submission")
    st.markdown("---")
    
    with st.form(key='review_form'):
        st.write("What are your future recommendations regarding the course?")
        review_text = st.text_area(
            label="Enter your feedback here:",
            height=200,
            key='text-input'
        )
        
        # Every form must have a submit button.
        submit_button = st.form_submit_button(label='Send your Review')
    
    if submit_button:
        if not review_text.strip():
            st.error("Please enter some feedback before submitting.")
            return

        with st.spinner('Analyzing and submitting review...'):
            # 1. Analyze review
            result = get_topic_and_sentiment_for_comment(review_text)
            
            # result is (topic_id, sentiment_int, topic_confidence, sentiment_confidence)
            if result:
                topic_id, sentiment_int, _, _ = result[:4]
                
                # 2. Insert into database
                insert_review(review_text, topic_id, sentiment_int)
                
                # 3. Clear data caches so the admin dashboard updates immediately
                st.cache_data.clear()
                
                # 4. Success message
                st.success("🎉 Thank you for your review!")
                st.info(f"""
                    **Your Feedback Analysis:**
                    - **Predicted Topic:** {TOPIC_LABELS.get(topic_id, 'Unknown')}
                    - **Predicted Sentiment:** {SENTIMENT_LABELS.get(sentiment_int, 'Unknown')}
                """)
            else:
                st.error("Could not process the review. Please check the review text.")

def show_admin_dashboard():
    """Displays the administrative dashboard with review analysis."""
    st.title("📊 Admin Dashboard")
    st.markdown("Welcome, Admin! View the analytics and recent reviews below.")
    st.markdown("---")

    # 1. Sentiment Metrics (Top Row)
    st.header("Overall Sentiment")
    
    metrics = get_sentiment_metrics()
    positive = metrics.get('positive', 0)
    neutral = metrics.get('neutral', 0)
    negative = metrics.get('negative', 0)
    total = positive + neutral + negative
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Reviews", total)
    col2.metric("Positive", positive, delta_color="normal")
    col3.metric("Neutral", neutral)
    col4.metric("Negative", negative, delta_color="inverse")
    
    st.markdown("---")

    # 2. Topic Distribution Chart
    st.header("Topic Distribution")
    df_topics = get_topic_distribution()
    
    # Ensure there is data to display charts
    if not df_topics.empty and df_topics['Count'].sum() > 0:
        # Create a horizontal bar chart for topics
        st.bar_chart(df_topics.set_index('Topic')['Count'])
    else:
        st.info("No data available to display topic distribution.")

    st.markdown("---")

    # 3. Recent Reviews Table
    st.header("Recent Reviews")
    df_recent = get_recent_reviews_df()
    
    if not df_recent.empty:
        # Style the dataframe for better readability
        st.dataframe(
            df_recent, 
            column_config={
                "Feedback": st.column_config.TextColumn(
                    "Feedback",
                    help="The raw text submitted by the user.",
                    width="large",
                ),
            },
            hide_index=True,
        )
        st.caption(f"Showing the {len(df_recent)} most recent reviews.")
    else:
        st.info("No reviews have been submitted yet.")


# --- Main Navigation ---

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Review Submission", "Admin Dashboard")
)

# Render the selected page
if page == "Review Submission":
    show_review_form()
elif page == "Admin Dashboard":
    show_admin_dashboard()
from database import (
    init_db, insert_review, get_all_reviews, populate_original_data,
    sentiment_retrieval, limit_reviews
)

warnings.filterwarnings("ignore")

# -----------------------------
# INITIAL DATABASE SETUP
# -----------------------------
init_db()
populate_original_data()

topic_labels = {
    0: "Need for additional guidance from lecturer during labs",
    1: "Practical applications of taught content",
    2: "Miscellaneous comments regarding CATs and explanations of content",
    3: "Negative opinions related to the amount of content within the slides used in teaching"
}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="NLP Content Reviewer", layout="wide")

st.title("NLP Content Reviewer")

tabs = st.tabs(["Submit Review", "Admin Dashboard"])

# =============================================================
# TAB 1: USER SUBMISSION PAGE
# =============================================================
with tabs[0]:
    st.header("Submit a Review")
    review_text = st.text_area("Enter your comment:", height=150)

    if st.button("Submit"):
        if review_text.strip():
            result = get_topic_and_sentiment_for_comment(review_text)

            insert_review(
                review_text,
                result["topic_id"],
                result["sentiment"]
            )

            st.success("Your review has been submitted!")
            st.write("### Topic Prediction:")
            st.write(topic_labels[result["topic_id"]])

            st.write("### Sentiment:")
            st.write(result["sentiment"])
        else:
            st.error("Please enter some text.")

# =============================================================
# TAB 2: ADMIN PAGE
# =============================================================
with tabs[1]:
    st.header("Admin Dashboard")

    # ---- Show limited reviews ----
    st.subheader("Latest Reviews")
    reviews = limit_reviews()
    if reviews:
        df_reviews = pd.DataFrame(reviews)
        df_reviews["topic"] = df_reviews["topic"].apply(
            lambda t: topic_labels.get(t, "Unknown")
        )
        st.dataframe(df_reviews)
    else:
        st.info("No reviews found.")

    # ---- Sentiment Pie Chart ----
    st.subheader("Sentiment Distribution")
    sentiments = sentiment_retrieval()
    if sentiments:
        df_sent = pd.DataFrame(sentiments)
        st.bar_chart(df_sent.set_index("sentiment"))
    else:
        st.info("Sentiment data not available.")

    # ---- Topic Distribution ----
    st.subheader("Topic Distribution")
    all_reviews = get_all_reviews()
    if all_reviews:
        topic_counts = pd.Series([r["topic"] for r in all_reviews]).value_counts()
        df_topics = pd.DataFrame({
            "Topic": [topic_labels[int(t)] for t in topic_counts.index],
            "Count": topic_counts.values
        })
        st.bar_chart(df_topics.set_index("Topic"))
    else:
        st.info("Topic data not available.")
