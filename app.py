from flask import Flask, request, jsonify, render_template
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os
from dotenv import load_dotenv
from groq import Groq
import concurrent.futures
import requests

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

app = Flask(__name__)

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
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        return None

def chunk_transcript(transcript, chunk_size=4000):
    """Split transcript into chunks of specified size"""
    chunks = []
    current_chunk = []
    current_chunk_duration = 0

    for entry in transcript:
        current_chunk.append(entry)
        current_chunk_duration += entry['duration']

        if current_chunk_duration >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_chunk_duration = 0

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def generate_short_summary(text, video_title, chunk_index, total_chunks):
    """Generate a short descriptive summary of the text using GROQ API"""
    prompt = f"""Summarize the following chunk of transcript from the video titled '{video_title}'. 
    This is chunk {chunk_index + 1} out of {total_chunks}. 
    Provide exactly 5-6 bullet points that fit into the context of the entire video. Do not include any introductory text or headers, only the bullet points:

    {text}

    Bullet Points (5-6):"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
        )
        summary = chat_completion.choices[0].message.content
        
        # Process the summary to ensure it's in bullet point format
        points = summary.split('\n\n')
        formatted_points = []
        for point in points:
            point = point.strip()
            if point and not point.startswith('•'):
                point = '• ' + point
            if point:
                formatted_points.append(point)
        
        return '\n'.join(formatted_points)
    except Exception as e:
        print(f"Error in summarization: {str(e)}")
        return None

def format_timestamp(seconds):
    """Format seconds into a human-readable timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_chunk(chunk, i, video_title, total_chunks):
    chunk_text = " ".join([entry['text'] for entry in chunk])
    chunk_start_time = chunk[0]['start']
    chunk_end_time = chunk[-1]['start'] + chunk[-1]['duration']
    summary = generate_short_summary(chunk_text, video_title, i, total_chunks)
    return {
        "chunk_index": i,
        "start_time": format_timestamp(chunk_start_time),
        "end_time": format_timestamp(chunk_end_time),
        "summary": summary
    }

def get_video_title(video_id):
    """Get the title of a YouTube video using the oEmbed API"""
    url = f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = requests.get(url)
        data = response.json()
        return data['title']
    except Exception as e:
        print(f"Error fetching video title: {str(e)}")
        return f"Video {video_id}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    youtube_url = request.json['url']
    video_id = get_video_id(youtube_url)
    
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    transcript = get_transcript(video_id)
    if not transcript:
        return jsonify({"error": "Failed to fetch transcript"}), 400
    
    video_title = get_video_title(video_id)
    
    chunks = chunk_transcript(transcript)
    total_chunks = len(chunks)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: process_chunk(x[1], x[0], video_title, total_chunks), enumerate(chunks)))
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)