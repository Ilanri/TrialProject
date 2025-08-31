
import os
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Placeholder for Groq LLM API (replace with actual API call)
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Placeholder, update as needed
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

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

if 'docs' not in st.session_state:
	st.session_state.docs = []
	st.session_state.embeddings = None
	st.session_state.index = None
	st.session_state.model = load_model()

st.sidebar.header("Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
if uploaded_files:
	docs = []
	for file in uploaded_files:
		docs.append(file.read().decode("utf-8"))
	st.session_state.docs = docs
	embeddings = embed_texts(docs, st.session_state.model)
	st.session_state.embeddings = embeddings
	st.session_state.index = create_faiss_index(embeddings)
	st.sidebar.success(f"Loaded {len(docs)} documents.")

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
