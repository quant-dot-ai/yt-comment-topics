import requests
import streamlit as st
import pandas as pd
import numpy as np
import googleapiclient.discovery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from urllib.parse import urlparse, parse_qs
from transformers import pipeline
import plotly.express as px
from sklearn.decomposition import PCA
import openai

# Keys and setup
api_key = st.secrets["api_keys"]["YOUTUBE_API_KEY"]
openai.api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def extract_video_id(youtube_url):
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

def get_topic_title(terms, comments_df, cluster_id):
    """Generate descriptive topic summary using top comments"""
    cluster_comments = comments_df[comments_df['cluster'] == cluster_id]
    top_comments = cluster_comments.nlargest(3, 'like_count')['text'].tolist()
    
    prompt = f"""Here are the top 3 most-liked comments from a cluster:
    1. {top_comments[0]}
    2. {top_comments[1] if len(top_comments) > 1 else ''}
    3. {top_comments[2] if len(top_comments) > 2 else ''}
    
    Write a single clear sentence (max 10-12 words) that captures the main discussion topic.
    Be specific and natural. Use active voice. Avoid starting with 'Comments about' or 'Discussion of'.
    Response should be just the sentence, nothing else."""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except:
        return f"Cluster {cluster_id + 1}"

def extract_topics_llm(comments_df, num_clusters=5):
    """Extract topics using LLM and clustering"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        max_df=0.95,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(comments_df['text'])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(tfidf_matrix.toarray())
    
    comments_df['cluster'] = cluster_labels
    comments_df['x'] = coords[:, 0]
    comments_df['y'] = coords[:, 1]
    
    cluster_centers = kmeans.cluster_centers_
    
    topics = []
    for i in range(num_clusters):
        cluster_center = cluster_centers[i]
        top_terms_idx = np.argsort(cluster_center)[-5:]
        top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_idx]
        
        cluster_comments = comments_df[comments_df['cluster'] == i]
        if len(cluster_comments) > 0:
            cluster_vectors = vectorizer.transform(cluster_comments['text'])
            distances = np.sqrt(np.sum((cluster_vectors.toarray() - cluster_center) ** 2, axis=1))
            representative_idx = distances.argmin()
            representative_comment = cluster_comments.iloc[representative_idx]['text']
            
            topic_title = get_topic_title(top_terms, comments_df, i)
            
            topics.append({
                'cluster_id': i,
                'title': topic_title,
                'size': len(cluster_comments),
                'top_terms': top_terms,
                'representative_comment': representative_comment
            })
    
    return topics, cluster_labels, comments_df

def visualize_clusters(comments_df, topics):
    """Create user-friendly cluster visualization"""
    cluster_titles = {topic['cluster_id']: topic['title'].strip('Topic ()') for topic in topics}
    comments_df['topic'] = comments_df['cluster'].map(cluster_titles)
    
    fig = px.scatter(
        comments_df,
        x='x', y='y',
        color='topic',
        hover_data=['text'],
        title='Comments Grouped by Topic',
        labels={'x': '', 'y': ''},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_traces(
        hovertemplate="<b>Comment:</b> %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    st.plotly_chart(fig)

def display_comments_table(comments_df):
    """Display comments as interactive cards"""
    sorted_df = comments_df.sort_values('like_count', ascending=False)
    n_clusters = len(sorted_df['cluster'].unique())
    colors = px.colors.qualitative.Set3[:n_clusters]
    
    for _, row in sorted_df.iterrows():
        with st.container():
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(
                    f"""
                    <div style="
                        padding: 1rem;
                        border-radius: 10px;
                        margin: 0.5rem 0;
                        background-color: {colors[row['cluster']]}40;
                        border-left: 5px solid {colors[row['cluster']]};
                    ">
                        {row['text']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        padding: 1rem;
                        margin: 0.5rem 0;
                    ">
                        <span style="font-size: 1.2em; font-weight: bold;">👍 {row['like_count']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

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
                topics, cluster_labels, comments_df = extract_topics_llm(comments_df, num_clusters)
                
                visualize_clusters(comments_df, topics)
                display_topics(topics, comments_df)
                
                st.subheader("Comments by Popularity")
                display_comments_table(comments_df[['text', 'cluster', 'like_count']])

if __name__ == "__main__":
    main()
