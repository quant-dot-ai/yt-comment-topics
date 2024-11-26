import streamlit as st

# Streamlit UI
st.title("YouTube Comments Topic Analyzer")
st.subheader("Extract topics from YouTube comments using BERTopic")

# Input field for YouTube video ID
video_id = st.text_input("Enter YouTube Video ID", "")

if st.button("Analyze Topics"):
    if video_id.strip():
        st.info("Fetching comments...")
        # comments = fetch_comments(video_id)
        st.text(video_id)
