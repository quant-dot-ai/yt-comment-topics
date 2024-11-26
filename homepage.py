import streamlit as st
import os
import pandas as pd

api_key = os.getenv('YOUTUBE_API_KEY')

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

def main():
    # Streamlit UI
    st.title("YouTube Comments Topic Analyzer")
    st.subheader("Extract topics from YouTube comments using BERTopic")
    
    # Input field for YouTube video ID
    video_id = st.text_input("Enter YouTube Video ID", "")
    
    if st.button("Analyze Topics"):
        if video_id.strip():
            st.info("Fetching comments...")
            # comments = fetch_comments(video_id)
            get_comments(video_id)
            st.dataframe(df_with_n_relevant_comments)

def get_comments(video_id, next_page_token=None):
    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=10,
        pageToken=next_page_token,
        order="relevance"
    )
    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]
        comments.append([
            comment["authorDisplayName"],
            comment["publishedAt"],
            comment["updatedAt"],
            comment["likeCount"],
            comment["textDisplay"]
        ])
    
    if "nextPageToken" in response:
        get_comments(video_id, response["nextPageToken"])

    df_with_n_relevant_comments=pd.DataFrame(comments, columns=["author", "published_at", "updated_at", "like_count", "text"])
    return df_with_n_relevant_comments

if __name__ == "__main__":
    main()
