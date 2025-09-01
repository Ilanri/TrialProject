import requests
import time
import os
import logging

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"

HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

DEBUG = os.environ.get("DEBUG", "0") == "1"
logger = logging.getLogger("assemblyai_utils")

def transcribe_audio_assemblyai(file_path: str) -> str:
    try:
        # Upload audio file
        with open(file_path, 'rb') as f:
            response = requests.post(f"{ASSEMBLYAI_URL}/upload", headers=HEADERS, files={"file": f})
        response.raise_for_status()
        audio_url = response.json()["upload_url"]

        # Start transcription
        transcript_response = requests.post(
            f"{ASSEMBLYAI_URL}/transcript",
            headers=HEADERS,
            json={"audio_url": audio_url}
        )
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json()["id"]

        # Poll for completion
        while True:
            poll = requests.get(f"{ASSEMBLYAI_URL}/transcript/{transcript_id}", headers=HEADERS)
            poll.raise_for_status()
            status = poll.json()["status"]
            if status == "completed":
                transcript = poll.json()["text"]
                logger.info(f"Transcribed audio file with AssemblyAI: {file_path}")
                return transcript
            elif status == "failed":
                logger.error(f"Failed to transcribe audio file with AssemblyAI: {file_path}")
                return "[Transcription failed]"
            time.sleep(3)
    except Exception as e:
        logger.error(f"Error in transcribing audio file with AssemblyAI: {file_path}: {e}")
        raise
