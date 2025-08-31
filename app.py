


import os
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from db_utils import save_faiss_index, load_faiss_index, save_docs_and_embeddings, load_docs_and_embeddings
from file_utils import save_uploaded_file, extract_text_from_file, SUPPORTED_TEXT, SUPPORTED_PDF, SUPPORTED_AUDIO

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Placeholder, update as needed
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

DB_DIR = "db"
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss.index")
DOCS_EMB_PATH = os.path.join(DB_DIR, "docs_emb.pkl")

os.makedirs(DB_DIR, exist_ok=True)

@st.cache_resource
def load_model():
	return SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts, model):
	return model.encode(texts, show_progress_bar=False)

def create_faiss_index(embeddings):
	dim = embeddings.shape[1]
	index = faiss.IndexFlatL2(dim)
	index.add(embeddings)
	return index

def groq_chat(prompt, context=""):
	headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
	data = {
		"model": "llama3-8b-8192",  # Example model name, update as needed
		"messages": [
			{"role": "system", "content": context},
			{"role": "user", "content": prompt}
		]
	}
	response = requests.post(GROQ_API_URL, headers=headers, json=data)
	if response.status_code == 200:
		return response.json()['choices'][0]['message']['content']
	else:
		return f"[Groq API error: {response.status_code}]"

st.title("RAG Chatbot with Groq and FAISS")


def try_load_db():
	docs, embeddings, index = [], None, None
	if os.path.exists(DOCS_EMB_PATH) and os.path.exists(FAISS_INDEX_PATH):
		try:
			docs, embeddings = load_docs_and_embeddings(DOCS_EMB_PATH)
			index = load_faiss_index(FAISS_INDEX_PATH)
		except Exception as e:
			st.warning(f"Could not load persistent DB: {e}")
	return docs, embeddings, index


def load_txt_files_from_db():
	txts = []
	for fname in os.listdir(DB_DIR):
		if fname.endswith('.txt'):
			with open(os.path.join(DB_DIR, fname), 'r', encoding='utf-8') as f:
				txts.append(f.read())
	return txts

if 'docs' not in st.session_state:
	docs, embeddings, index = try_load_db()
	# If no persistent DB, try to load .txt files from db/
	if not docs:
		docs = load_txt_files_from_db()
		if docs:
			st.session_state.model = load_model()
			embeddings = embed_texts(docs, st.session_state.model)
			index = create_faiss_index(embeddings)
			save_docs_and_embeddings(docs, embeddings, DOCS_EMB_PATH)
			save_faiss_index(index, FAISS_INDEX_PATH)
	st.session_state.docs = docs
	st.session_state.embeddings = embeddings
	st.session_state.index = index
	if 'model' not in st.session_state:
		st.session_state.model = load_model()

st.sidebar.header("Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
	"Upload files (txt, pdf, audio)",
	type=["txt", "pdf", "mp3", "wav", "ogg", "m4a"],
	accept_multiple_files=True
)
if uploaded_files:
	new_docs = []
	for uploaded_file in uploaded_files:
		save_path = save_uploaded_file(uploaded_file, DB_DIR)
		text = extract_text_from_file(save_path)
		if text.strip():
			new_docs.append(text)
			# Also add to session state and persist
			st.session_state.docs.append(text)
	if new_docs:
		# Re-embed all docs for simplicity (could optimize)
		embeddings = embed_texts(st.session_state.docs, st.session_state.model)
		st.session_state.embeddings = embeddings
		st.session_state.index = create_faiss_index(embeddings)
		save_docs_and_embeddings(st.session_state.docs, embeddings, DOCS_EMB_PATH)
		save_faiss_index(st.session_state.index, FAISS_INDEX_PATH)
		st.sidebar.success(f"Added and saved {len(new_docs)} new documents.")

def retrieve_context(query, k=2):
	if st.session_state.index is None:
		return ""
	query_emb = embed_texts([query], st.session_state.model)
	D, I = st.session_state.index.search(query_emb, k)
	return "\n".join([st.session_state.docs[i] for i in I[0] if i < len(st.session_state.docs)])

st.write("Ask a question about your uploaded documents:")
user_input = st.text_input("You:")

if st.button("Send") and user_input:
	context = retrieve_context(user_input)
	with st.spinner("Groq is thinking..."):
		answer = groq_chat(user_input, context)
	st.markdown(f"**Bot:** {answer}")
