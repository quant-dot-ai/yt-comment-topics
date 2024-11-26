import requests
import streamlit as st
import os
import pandas as pd
import googleapiclient.discovery
from fastopic import FASTopic
from topmost.preprocessing import Preprocessing

# Keys
api_key=st.secrets["api_keys"]["YOUTUBE_API_KEY"] 
token = st.secrets["HUGGINGFACE_TOKEN]["token"]

# Objects
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": "Bearer {token}"}



def get_comments(video_id, next_page_token=None):
    comments = []
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100,
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
        
    df=pd.DataFrame(comments, columns=["author", "published_at", "updated_at", "like_count", "text"])
    return df     


def get_topics_from_fasTopic(comments_text):
    preprocessing = Preprocessing(stopwords='English')
    model = FASTopic(num_topics=5, preprocessing=preprocessing)
    topic_top_words, doc_topic_dist = model.fit_transform(comments_text) # Needs to be fit_transform to get embeddings of the doc.

    # Display topic words in table form
    st.table(topic_top_words)

    st.table(model.get_beta())

    fig_topics = model.visualize_topic(top_n=5)
    st.plotly_chart(fig_topics)

    fig_topic_wts = model.visualize_topic_weights(top_n=20, height=500)
    st.plotly_chart(fig_topic_wts)


def comment_section_sentiment(comment_texts):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    output = query({
        "inputs": comment_texts,
    })

def main():
    # Streamlit UI
    st.title("YouTube Comments Topic Analyzer")
    st.subheader("Extract topics from YouTube comments using BERTopic")
    
    # Input field for YouTube video ID
    video_id = st.text_input("Enter YouTube Video ID", "")
    
    if st.button("Analyze Topics"):
        if video_id.strip():
            st.info("Fetching comments...")
            comments_text=get_comments(video_id)
            st.dataframe(comments_text)
            st.subhead("Overall Sentiment:")
            comment_section_sentiment(comments_text['text'].tolist())
            get_topics_from_fasTopic(comments_text['text'].tolist())
            

if __name__ == "__main__":
    main()
