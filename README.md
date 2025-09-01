# Streamlit RAG Chatbot

This project is a minimal Streamlit chatbot app using Retrieval-Augmented Generation (RAG) with Groq as the LLM and FAISS as the vector store for document retrieval.

## Features

- Chatbot interface powered by Streamlit
- RAG architecture: retrieves relevant context from your documents
- Uses Groq for LLM responses
- FAISS for fast vector search

## Frontend Features

- Modern, unified color scheme and fonts for a clean look
- Sidebar with persona/tone selection and knowledge base management
- Drag-and-drop file upload for PDFs and audio (mp3, wav, ogg, m4a)
- Visual separator for clear sectioning
- LLM-driven suggested questions for user guidance
- Context/tone switcher for dynamic conversation style
- Sidebar image for branding or personalization
- Responsive layout for desktop and mobile

## Setup

Use source .venv/bin/activate

1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`

## Configuration

- Set your Groq API key in the environment variable `GROQ_API_KEY`.
- Place your knowledge base files in the `data/` directory.

 If you update intro.txt, simply delete persona_prompt.txt to force a new persona to be generated.

## Debugging

To enable debugging and log important actions:

1. Set the environment variable before running Streamlit:

   ```bash
   export DEBUG=1
   streamlit run app.py
   ```

   To disable debugging, use:

   ```bash
   export DEBUG=0
   streamlit run app.py
   ```

2. All logs will be written to `debug.log` in the project directory. This file will contain only essential information: user queries, what was retrieved, and whether major actions succeeded or failed.

3. No debug information is shown in the UI.

4. To keep your log file clean, logs from third-party libraries (like Watchdog) are suppressed unless they are errors.

5. You can view the log file with:

   ```bash
   tail -f debug.log
   ```

## System Design Choices

### Architecture

- **Retrieval-Augmented Generation (RAG):** The system combines a vector database (FAISS) for fast semantic search with a large language model (Groq) for answer generation. This allows the chatbot to ground its answers in your own documents, improving factuality and relevance.

### Technology Choices

- **Streamlit:** Chosen for rapid prototyping and interactive UI, making it easy to deploy and use the chatbot in a browser.
- **FAISS:** Provides efficient similarity search over document embeddings, enabling scalable retrieval even with large knowledge bases.
- **SentenceTransformers:** Used to embed text chunks for semantic search. The 'all-MiniLM-L6-v2' model offers a good balance of speed and quality.
- **Groq LLM API:** Used for generating answers and suggested questions. Groq offers high-quality, fast LLM inference.
- **AssemblyAI:** Used for audio transcription, allowing the knowledge base to include audio files (mp3, wav, ogg, m4a) as well as text and PDF.
- **pypdf:** For robust PDF text extraction and cleaning.
- **File chunking and cleaning:** Documents are chunked by paragraph or fixed window, and PDF text is cleaned to remove artifacts, ensuring high-quality retrieval.
- **Persona system:** The chatbot's persona is dynamically generated from `intro.txt` using the LLM, and cached for efficiency. This allows the chatbot to answer in a consistent, user-defined style.
- **UI/UX:** The app uses a modern, unified color scheme and fonts, with a sidebar for persona/tone selection and knowledge base management. User images and visual separators improve clarity and engagement.
- **Exclusions:** Internal files like `embedded_files.txt` and `persona_prompt.txt` are excluded from the knowledge base to prevent polluting search results.

### Why not alternatives?

- **No pyaudio/pydub:** Audio is handled via AssemblyAI, which is more robust and cloud-based, avoiding local audio decoding issues.
- **No Chroma/Weaviate:** FAISS is lightweight, fast, and easy to use for local vector search.
- **No OpenAI API:** Groq is used for LLM inference, but the system is modular and could be adapted to other LLM APIs.

---

For further details, see comments in the code or contact the author.
