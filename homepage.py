import requests
import streamlit as st
import os
import pandas as pd
import googleapiclient.discovery
from fastopic import FASTopic
from topmost.preprocessing import Preprocessing
from collections import Counter
from urllib.parse import urlparse, parse_qs

# Keys
api_key=st.secrets["api_keys"]["YOUTUBE_API_KEY"] 
token = st.secrets["HUGGINGFACE_TOKEN"]["token"]

# Objects
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {token}", 
           "x-wait-for-model": "true"}



def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube video URL.

    Args:
        youtube_url (str): The YouTube video URL.

    Returns:
        str: The video ID if found, or None if the URL is invalid.
    """
    try:
        # Parse the URL
        parsed_url = urlparse(youtube_url)

        # Check if the URL is a valid YouTube link
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            # Extract video ID from the 'v' query parameter
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]

        elif parsed_url.hostname in ['youtu.be']:
            # Extract video ID from the path for shortened URLs
            return parsed_url.path[1:]

        else:
            return None
    except Exception as e:
        print(f"Error parsing YouTube URL: {e}")
        return None

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
    model = FASTopic(num_topics=20, preprocessing=preprocessing)
    topic_top_words, doc_topic_dist = model.fit_transform(comments_text) # Needs to be fit_transform to get embeddings of the doc.

    # Display topic words in table form
    st.table(topic_top_words)

    st.table(model.get_beta())

    fig_topics = model.visualize_topic(top_n=5)
    st.plotly_chart(fig_topics)

    fig_topic_wts = model.visualize_topic_weights(top_n=20, height=500)
    st.plotly_chart(fig_topic_wts)



def comment_section_sentiment(comment_texts):
    # Make API request to Hugging Face
    response = requests.post(API_URL, headers=headers, json={"inputs": comment_texts})
    
    # Check if the response is successful
    if response.status_code == 200:
        sentiment_results = response.json()

        if isinstance(sentiment_results, list):
            # Count the occurrences of positive and negative sentiments
            sentiment_counter = Counter()

            for sentiment in sentiment_results:
                sentiment_label = sentiment[0]['label']
                sentiment_counter[sentiment_label] += 1

            # Calculate percentages
            total_comments = sum(sentiment_counter.values())
            positive_percentage = (sentiment_counter["POSITIVE"] / total_comments) * 100
            negative_percentage = (sentiment_counter["NEGATIVE"] / total_comments) * 100

            # Display results in Streamlit
            st.subheader("Overall Sentiment Analysis")
            st.write(f"**Positive Sentiment:** {positive_percentage:.2f}%")
            st.write(f"**Negative Sentiment:** {negative_percentage:.2f}%")
        else:
            st.error("Failed to parse sentiment analysis response.")
    else:
        st.error("Failed to get sentiment analysis. Please check the API call or your API key.")
    
def main():
    # Streamlit UI
    st.title("YouTube Comments Topic Analyzer")
    st.subheader("Extract topics from YouTube comments using BERTopic")
    
    # Input field for YouTube video ID
    video_url = st.text_input("Enter YouTube URL", "")
    video_id = extract_video_id(video_url)
    
    if st.button("Analyze Topics"):
        if video_id.strip():
            st.info("Fetching comments...")
            comments_text=get_comments(video_id)
            st.dataframe(comments_text)
            st.subheader("Overall Sentiment:")
            comment_section_sentiment(comments_text['text'].tolist())
            get_topics_from_fasTopic(comments_text['text'].tolist())
            

if __name__ == "__main__":
    main()
