import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils.audio_utils import extract_audio_text
from langchain.schema import Document
import os

def extract_video_text(uploaded_video):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    clip = VideoFileClip(temp_video_path)
    clip.audio.write_audiofile("temp_audio.wav")
    text = extract_audio_text(open("temp_audio.wav", "rb"))
    clip.audio.close()
    os.remove("temp_audio.wav")
    os.remove(temp_video_path)

    return text
