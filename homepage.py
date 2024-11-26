import streamlit as st
import os
import pandas as pd
import googleapiclient.discovery

# Fetch YouTube API key from environment variable
api_key = os.getenv('YOUTUBE_API_KEY')

# Initialize YouTube API client
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# Streamlit UI
st.title("YouTube Comments Topic Analyzer")
st.subheader("Extract topics from YouTube comments")

# Input field for YouTube video ID
video_id = st.text_input("Enter YouTube Video ID", "")

def get_comments(video_id, max_results=10):
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_page_token,
                order="relevance"
            )
            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": comment["authorDisplayName"],
                    "published_at": comment["publishedAt"],
                    "like_count": comment["likeCount"],
                    "text": comment["textDisplay"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
        except Exception as e:
            st.error(f"Error fetching comments: {e}")
            break

    return pd.DataFrame(comments)

if st.button("Analyze Topics"):
    if video_id.strip():
        with st.spinner("Fetching comments..."):
            df = get_comments(video_id, max_results=10)
        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No comments found for the given video ID.")
    else:
        st.error("Please enter a valid YouTube video ID.")
