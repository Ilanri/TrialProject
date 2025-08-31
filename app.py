


import os
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

from db_utils import save_faiss_index, load_faiss_index, save_docs_and_embeddings, load_docs_and_embeddings
from file_utils import save_uploaded_file, extract_text_from_file, SUPPORTED_TEXT, SUPPORTED_PDF, SUPPORTED_AUDIO
from persona_utils import construct_persona_from_intro


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Placeholder, update as needed
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Config: path to intro file for persona
DB_DIR = "db"
INTRO_FILE = os.path.join(DB_DIR, "intro.txt")
PERSONA_CACHE_FILE = os.path.join(DB_DIR, "persona_prompt.txt")

def get_or_create_persona():
	# If persona already constructed and cached, use it
	if os.path.exists(PERSONA_CACHE_FILE):
		with open(PERSONA_CACHE_FILE, "r", encoding="utf-8") as f:
			return f.read().strip()
	# Otherwise, construct persona from intro.txt using LLM
	if os.path.exists(INTRO_FILE):
		with open(INTRO_FILE, "r", encoding="utf-8") as f:
			intro_text = f.read().strip()
		persona = construct_persona_from_intro(intro_text, GROQ_API_URL, GROQ_API_KEY)
		with open(PERSONA_CACHE_FILE, "w", encoding="utf-8") as f:
			f.write(persona)
		return persona
	return ""
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
	persona = get_or_create_persona()
	# Compose system prompt: persona + context
	if persona:
		system_prompt = f"{persona}"
	else:
		system_prompt = "You are the person the questions are aabout."
		
	user_prompt = f"{prompt}\n\nUse the following context to answer the question informally, concisely and in first person form. Strictly stay within the provided context. \n{context}"

	headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
	data = {
		"model": "llama-3.3-70b-versatile",  # Example model name, update as needed
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt}
		]
	}
	response = requests.post(GROQ_API_URL, headers=headers, json=data)
	if response.status_code == 200:
		return response.json()['choices'][0]['message']['content']
	else:
		return f"[Groq API error: {response.status_code}] - {response.text}"
st.sidebar.markdown(f"**Persona config file:** `{INTRO_FILE}`")

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





def load_all_supported_files_from_db():
	from file_utils import extract_text_from_file, SUPPORTED_TEXT, SUPPORTED_PDF, SUPPORTED_AUDIO
	docs = []
	file_list = []
	file_map = {}
	chunk_file_map = []  # Track which file each chunk comes from
	for fname in os.listdir(DB_DIR):
		ext = os.path.splitext(fname)[1].lower()
		if ext in SUPPORTED_TEXT + SUPPORTED_PDF + SUPPORTED_AUDIO:
			fpath = os.path.join(DB_DIR, fname)
			chunks = extract_text_from_file(fpath)
			for chunk in chunks:
				if chunk.strip():
					docs.append(chunk)
					chunk_file_map.append(fname)
			if chunks:
				file_list.append(fname)
			file_map[fname] = chunks
	print(f"[RAG] Included files: {file_list}")
	return docs, file_list, file_map, chunk_file_map



# Always scan and embed all supported files in db/ on startup, but avoid re-embedding

# Always scan and embed all supported files in db/ on startup, but avoid re-embedding
if 'docs' not in st.session_state:
	st.session_state.model = load_model()
	docs, file_list, file_map, chunk_file_map = load_all_supported_files_from_db()
	embedded_files = []
	# Try to load previous embeddings and file list
	prev_docs, prev_emb, prev_index = [], None, None
	if os.path.exists(DOCS_EMB_PATH) and os.path.exists(FAISS_INDEX_PATH):
		try:
			prev_docs, prev_emb = load_docs_and_embeddings(DOCS_EMB_PATH)
			prev_index = load_faiss_index(FAISS_INDEX_PATH)
		except Exception as e:
			st.warning(f"Could not load persistent DB: {e}")
	# Map previous docs to filenames if possible (assume order is preserved)
	prev_file_list = []
	if os.path.exists(os.path.join(DB_DIR, "embedded_files.txt")):
		with open(os.path.join(DB_DIR, "embedded_files.txt"), "r") as f:
			prev_file_list = [line.strip() for line in f.readlines()]
	# Only embed new files
	new_files = [fname for fname in file_list if fname not in prev_file_list]
	all_files = prev_file_list + new_files
	# Flatten all chunks for all files
	all_docs = []
	all_chunk_file_map = []
	for fname in all_files:
		if fname in file_map:
			for chunk in file_map[fname]:
				if chunk.strip():
					all_docs.append(chunk)
					all_chunk_file_map.append(fname)
		elif prev_file_list and fname in prev_file_list:
			idx = prev_file_list.index(fname)
			if prev_docs and idx < len(prev_docs):
				all_docs.append(prev_docs[idx])
				all_chunk_file_map.append(fname)
	if new_files:
		# Only embed new chunks, then concatenate with previous embeddings
		new_chunks = []
		for f in new_files:
			new_chunks.extend([c for c in file_map[f] if c.strip()])
		new_emb = embed_texts(new_chunks, st.session_state.model)
		if prev_emb is not None:
			embeddings = np.vstack([prev_emb, new_emb])
		else:
			embeddings = new_emb
	else:
		embeddings = prev_emb
	if embeddings is not None:
		index = create_faiss_index(embeddings)
		save_docs_and_embeddings(all_docs, embeddings, DOCS_EMB_PATH)
		save_faiss_index(index, FAISS_INDEX_PATH)
		# Save file list for future runs
		with open(os.path.join(DB_DIR, "embedded_files.txt"), "w") as f:
			for fname in all_files:
				f.write(fname + "\n")
	else:
		index = None
	st.session_state.docs = all_docs
	st.session_state.embeddings = embeddings
	st.session_state.index = index
	st.session_state.embedded_files = all_files
	st.session_state.chunk_file_map = all_chunk_file_map


st.sidebar.header("Knowledge Base")
st.sidebar.markdown("**Embedded files:**")
if 'embedded_files' in st.session_state:
	for fname in st.session_state.embedded_files:
		st.sidebar.write(f"- {fname}")


uploaded_files = st.sidebar.file_uploader(
	"Upload files (txt, pdf, audio)",
	type=["txt", "pdf", "mp3", "wav", "ogg", "m4a"],
	accept_multiple_files=True
)
if uploaded_files:
	new_chunks = []
	new_chunk_file_map = []
	for uploaded_file in uploaded_files:
		save_path = save_uploaded_file(uploaded_file, DB_DIR)
		chunks = extract_text_from_file(save_path)
		for chunk in chunks:
			if chunk.strip():
				new_chunks.append(chunk)
				new_chunk_file_map.append(uploaded_file.name)
				st.session_state.docs.append(chunk)
				st.session_state.chunk_file_map.append(uploaded_file.name)
	if new_chunks:
		embeddings = embed_texts(st.session_state.docs, st.session_state.model)
		st.session_state.embeddings = embeddings
		st.session_state.index = create_faiss_index(embeddings)
		save_docs_and_embeddings(st.session_state.docs, embeddings, DOCS_EMB_PATH)
		save_faiss_index(st.session_state.index, FAISS_INDEX_PATH)
		st.sidebar.success(f"Added and saved {len(new_chunks)} new chunks.")

def retrieve_context(query, k=10):
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
