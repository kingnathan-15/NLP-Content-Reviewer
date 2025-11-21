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

def get_sentiment_label(sentiment_int):
    """Safely get sentiment label with debugging."""
    label = SENTIMENT_LABELS.get(sentiment_int, f"Unknown (value: {sentiment_int})")
    if "Unknown" in label:
        print(f"⚠️ WARNING: Unexpected sentiment value: {sentiment_int} (type: {type(sentiment_int)})")
    return label
    
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
    # Convert list of sqlite3.Row objects to list of dictionaries
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
    
    # CRUCIAL: Convert sqlite3.Row objects to dicts before processing
    topic_ids = [dict(row)['topic'] for row in all_reviews]
    
    topic_counts = pd.Series(topic_ids).value_counts().sort_index()
    
    df_topics = pd.DataFrame({
        'Topic ID': topic_counts.index,
        'Count': topic_counts.values
    })
    
    # Map ID to label for chart readability
    df_topics['Topic'] = df_topics['Topic ID'].apply(lambda t: TOPIC_LABELS.get(t, "Unknown Topic"))
    
    return df_topics


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
            
            # Check if result is None before unpacking
            if result is not None:
                topic_id, sentiment_int, topic_confidence, sentiment_confidence = result
                
                # 2. Insert into database
                insert_review(review_text, topic_id, sentiment_int)
                
                # 3. Clear data caches so the admin dashboard updates immediately
                st.cache_data.clear()
                
                # Get labels safely
                topic_label = TOPIC_LABELS.get(topic_id, f"Unknown Topic (ID: {topic_id})")
                sentiment_label = SENTIMENT_LABELS.get(sentiment_int, f"Unknown Sentiment (value: {sentiment_int})")
                
                # 4. Success message
                st.success("🎉 Thank you for your review!")
                st.info(f"""
                    **Your Feedback Analysis:**
                    - **Predicted Topic:** {topic_label}
                    - **Predicted Sentiment:** {sentiment_label}
                    - **Topic Confidence:** {topic_confidence:.1%}
                    - **Sentiment Confidence:** {sentiment_confidence:.1%}
                """)
            else:
                st.error("Could not process the review. Please check that the NLP models are properly loaded and the review text is valid.")

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