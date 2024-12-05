import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip
from utils.audio_utils import extract_audio_text
import os

def extract_video_text(uploaded_video):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    try:
        clip = VideoFileClip(temp_video_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        clip.audio.write_audiofile(temp_audio_path)
        clip.audio.close()

        # Extract text
        with open(temp_audio_path, "rb") as audio_file:
            text = extract_audio_text(audio_file)

    except Exception as e:
        print(f"Error processing video: {e}")
        text = None

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return text
