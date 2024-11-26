import requests
import streamlit as st
import os
import pandas as pd
import googleapiclient.discovery
from fastopic import FASTopic
from topmost.preprocessing import Preprocessing

# Keys
api_key=st.secrets["api_keys"]["YOUTUBE_API_KEY"] 


# Objects
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)



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
    topic_top_words, doc_topic_dist = model.fit_transform(comments_text)
    # st.table(topic_top_words)

    # Display topic words in table form
    st.table(topic_top_words)
    
    # Visualize topics using available visualization methods in fastopic
    st.subheader("Topic Distribution")
    fig1 = model.plot_topic_distributions()  # Plot distribution of topics across all documents
    st.pyplot(fig1)
    
    st.subheader("Topic Heatmap")
    fig2 = model.plot_topic_heatmap()  # Plot heatmap to show the intensity of topics across the comments
    st.pyplot(fig2)

    st.subheader("Topic Word Scores")
    fig3 = model.plot_word_scores()  # Plot the word scores per topic to show word significance
    st.pyplot(fig3)

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
            get_topics_from_fasTopic(comments_text['text'].tolist())

if __name__ == "__main__":
    main()
