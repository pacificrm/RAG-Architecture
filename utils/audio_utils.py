import tempfile
import os
import speech_recognition as sr
from pydub import AudioSegment

recognizer = sr.Recognizer()

def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Audio could not be understood."
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service."

def extract_audio_text(uploaded_audio, chunk_duration=30):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(uploaded_audio.read())
        temp_audio_path = temp_audio.name

    audio = AudioSegment.from_file(temp_audio_path)
    duration_seconds = len(audio) // 1000

    chunks = [audio[start * 1000:(start + chunk_duration) * 1000]
              for start in range(0, duration_seconds, chunk_duration)]

    full_text = []
    for chunk in chunks:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
            chunk.export(chunk_file.name, format="wav")
            text = transcribe_audio(chunk_file.name)
            full_text.append(text)
            os.remove(chunk_file.name)

    os.remove(temp_audio_path)
    return " ".join(full_text)
