import streamlit as st 
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

st.title("Speech Transcribing")

load_dotenv(override=True)
client = OpenAI()

st.write("Record your message here... ")
audio_value = st.audio_input("Record a voice message")

if audio_value:
    st.audio(audio_value)
    # Save the uploaded audio to a temporary file

    #with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
    #    tmp_file.write(audio_value.getbuffer())
    #    tmp_file_path = tmp_file.name

    # Transcribe using OpenAI Whisper via GPT-4o endpoint
    #with open(audio_value, "rb") as audio_file:
    with st.spinner("Transcribing..."):
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",  # OpenAI uses Whisper for transcription
            file=audio_value.getvalue()
        )

    #st.subheader("Transcription:")
    #st.write(transcript["text"])


