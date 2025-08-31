import requests
import time
import os

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_URL = "https://api.assemblyai.com/v2"

HEADERS = {"authorization": ASSEMBLYAI_API_KEY}

def transcribe_audio_assemblyai(file_path: str) -> str:
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
            return poll.json()["text"]
        elif status == "failed":
            return "[Transcription failed]"
        time.sleep(3)
