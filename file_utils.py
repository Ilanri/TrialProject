import os
from typing import List
from pypdf import PdfReader
from pydub import AudioSegment
import tempfile
from assemblyai_utils import transcribe_audio_assemblyai
import re
import unicodedata

# Supported file types
SUPPORTED_TEXT = [".txt"]
SUPPORTED_PDF = [".pdf"]
SUPPORTED_AUDIO = [".mp3", ".wav", ".ogg", ".m4a"]

# File I/O functions
def save_uploaded_file(uploaded_file, save_dir: str) -> str:
    """Save an uploaded file to the specified directory."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    save_path = os.path.join(save_dir, uploaded_file.name)
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        raise IOError(f"Error saving file {uploaded_file.name}: {e}")
    return save_path

# Text cleaning function for PDFs
def clean_pdf_text(text: str) -> str:
    """Clean and normalize extracted text from PDFs."""
    bullets = ["•", "*", "·", "o", "●", "▪", "■", "►", "‣", "○"]
    for b in bullets:
        text = text.replace(b, " ")
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = '\n'.join([line for line in text.splitlines() if line.strip()])
    text = unicodedata.normalize("NFKC", text)
    return text

# PDF text extraction function
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return clean_pdf_text(text)
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF {file_path}: {e}")

# Audio transcription function using AssemblyAI
def extract_text_from_audio(file_path: str) -> str:
    """Transcribe audio to text using AssemblyAI."""
    try:
        return transcribe_audio_assemblyai(file_path)
    except Exception as e:
        raise ValueError(f"Error transcribing audio {file_path}: {e}")

# Text chunking function
def chunk_text_by_paragraphs(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Chunk text by paragraphs or fixed-size chunks."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    if len(paragraphs) <= 1 or max(len(p) for p in paragraphs) > chunk_size * 2:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += chunk_size - overlap
        return chunks
    return paragraphs

# Function to extract text based on file type
def extract_text_from_file(file_path: str) -> List[str]:
    """Extract text from supported file types."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in SUPPORTED_TEXT:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        elif ext in SUPPORTED_PDF:
            text = extract_text_from_pdf(file_path)
        elif ext in SUPPORTED_AUDIO:
            text = extract_text_from_audio(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return chunk_text_by_paragraphs(text)
    except Exception as e:
        raise ValueError(f"Error extracting text from file {file_path}: {e}")
