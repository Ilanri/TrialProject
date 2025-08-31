import os
from typing import List
from pypdf import PdfReader

from pydub import AudioSegment
import tempfile
from assemblyai_utils import transcribe_audio_assemblyai

SUPPORTED_TEXT = [".txt"]
SUPPORTED_PDF = [".pdf"]
SUPPORTED_AUDIO = [".mp3", ".wav", ".ogg", ".m4a"]


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_audio(file_path: str) -> str:
    # Use AssemblyAI for transcription
    return transcribe_audio_assemblyai(file_path)


def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    save_path = os.path.join(save_dir, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_TEXT:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext in SUPPORTED_PDF:
        return extract_text_from_pdf(file_path)
    elif ext in SUPPORTED_AUDIO:
        return extract_text_from_audio(file_path)
    else:
        return ""
