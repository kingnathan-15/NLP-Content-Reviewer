import streamlit as st
import pandas as pd
import warnings
from unpack import get_topic_and_sentiment_for_comment
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
