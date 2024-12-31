import requests
import streamlit as st
import pandas as pd
import googleapiclient.discovery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from urllib.parse import urlparse, parse_qs
import numpy as np
from transformers import pipeline

# Keys and setup remain the same
api_key = st.secrets["api_keys"]["YOUTUBE_API_KEY"]
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

# Add sentiment analyzer
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def extract_video_id(youtube_url):
    # Your existing extract_video_id function remains the same
    try:
        parsed_url = urlparse(youtube_url)
        if parsed_url.hostname in ['www.youtube.com', 'youtube.com', 'm.youtube.com']:
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        elif parsed_url.hostname in ['youtu.be']:
            return parsed_url.path[1:]
        else:
            return None
    except Exception as e:
        print(f"Error parsing YouTube URL: {e}")
        return None

def get_comments(video_id, next_page_token=None):
    # Your existing get_comments function remains the same
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
        
    return pd.DataFrame(comments, columns=["author", "published_at", "updated_at", "like_count", "text"])

def extract_topics_llm(comments_df, num_clusters=5):
    """Extract topics using LLM and clustering"""
    # Step 1: Convert comments to TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(comments_df['text'])
    
    # Step 2: Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    # Step 3: Get representative comments for each cluster
    comments_df['cluster'] = cluster_labels
    cluster_centers = kmeans.cluster_centers_
    
    topics = []
    for i in range(num_clusters):
        # Get top terms for cluster
        cluster_center = cluster_centers[i]
        top_terms_idx = np.argsort(cluster_center)[-5:]  # Get top 5 terms
        top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_idx]
        
        # Get representative comment (closest to cluster center)
        cluster_comments = comments_df[comments_df['cluster'] == i]
        if len(cluster_comments) > 0:
            # Get comment closest to cluster center
            cluster_vectors = vectorizer.transform(cluster_comments['text'])
            distances = np.sqrt(np.sum((cluster_vectors.toarray() - cluster_center) ** 2, axis=1))
            representative_idx = distances.argmin()
            representative_comment = cluster_comments.iloc[representative_idx]['text']
            
            topics.append({
                'cluster_id': i,
                'size': len(cluster_comments),
                'top_terms': top_terms,
                'representative_comment': representative_comment
            })
    
    return topics, cluster_labels

def display_topics(topics, comments_df):
    """Display topics and their analysis"""
    st.subheader("Comment Topics Analysis")
    
    for topic in topics:
        with st.expander(f"Topic {topic['cluster_id']+1} ({topic['size']} comments)"):
            st.write("**Key Terms:**", ", ".join(topic['top_terms']))
            st.write("**Representative Comment:**")
            st.write(topic['representative_comment'])
            
            # Get sentiment for cluster
            cluster_comments = comments_df[comments_df['cluster'] == topic['cluster_id']]
            sentiments = sentiment_analyzer(cluster_comments['text'].tolist()[:50])  # Analyze up to 50 comments
            sentiment_counts = Counter(s['label'] for s in sentiments)
            total = sum(sentiment_counts.values())
            
            st.write("**Sentiment Analysis:**")
            st.write(f"Positive: {sentiment_counts['POSITIVE']/total*100:.1f}%")
            st.write(f"Negative: {sentiment_counts['NEGATIVE']/total*100:.1f}%")

def main():
    st.title("YouTube Comments Topic Analyzer")
    st.subheader("Extract topics from YouTube comments using LLM and Clustering")
    
    video_url = st.text_input("Enter YouTube URL", "")
    num_clusters = st.slider("Number of Topics", min_value=3, max_value=10, value=5)
    
    if st.button("Analyze Topics"):
        video_id = extract_video_id(video_url)
        if video_id:
            with st.spinner("Fetching and analyzing comments..."):
                comments_df = get_comments(video_id)
                topics, cluster_labels = extract_topics_llm(comments_df, num_clusters)
                display_topics(topics, comments_df)
                
                # Display clustered comments
                comments_df['cluster'] = cluster_labels
                st.subheader("Clustered Comments")
                st.dataframe(comments_df[['text', 'cluster', 'like_count']])

if __name__ == "__main__":
    main()
