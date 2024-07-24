import streamlit as st
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token=HF_API_TOKEN,
)

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

def get_transcript(video_id):
    """Get transcript of YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def summarize_to_bullet_points(text, num_points=6):
    """Summarize text into a specified number of bullet points using Phi-3-mini-4k-instruct"""
    prompt = f"Summarize the following text into {num_points} main points:\n\n{text}"
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            stream=False
        )
        summary = response.choices[0].message.content
        
        # Split the summary into bullet points
        bullet_points = summary.split("\n")
        return [point.strip().lstrip("•-").strip() for point in bullet_points if point.strip()]
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None


st.title("YouTube Video Summarizer")
youtube_url = st.text_input("Enter YouTube video URL:")

button = st.button("Summarize")

if button:
    try:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL")
        else:
            transcript = get_transcript(video_id)
            if transcript:
                st.info("Summarizing the transcript into main points...")
                bullet_points = summarize_to_bullet_points(transcript)
                
                if bullet_points:
                    st.write("Summary:")
                    for point in bullet_points:
                        st.write(f"• {point}")
                else:
                    st.error("Failed to generate summary.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")