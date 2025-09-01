



import os
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests

# Inject custom CSS for theme
with open("style.css") as f:
	st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from db_utils import save_faiss_index, load_faiss_index, save_docs_and_embeddings, load_docs_and_embeddings
from file_utils import save_uploaded_file, extract_text_from_file, SUPPORTED_TEXT, SUPPORTED_PDF, SUPPORTED_AUDIO
from persona_utils import construct_persona_from_intro


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Placeholder, update as needed
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Config: path to intro file for persona
DB_DIR = "db"
INTRO_FILE = os.path.join(DB_DIR, "intro.txt")
PERSONA_CACHE_FILE = os.path.join(DB_DIR, "persona_prompt.txt")

# --- Sidebar User Image ---
import streamlit as st
import os
image_path = os.path.join("db", "me.jpg")
if os.path.exists(image_path):
	st.sidebar.image(image_path, caption="This is Ilanri!", use_container_width=True)

# --- Context Switch: Tone Selector ---
TONE_OPTIONS = {
	"Interview Mode": "Answer concisely, professionally, and highlight achievements as if in a job interview.",
	"Fast Facts": "Answer in bullet points or TL;DR style for quick reference.",
	"Mentor Mode": "Answer like a helpful mentor—encouraging, insightful, and guiding.",
	"Playful Mode": "Answer with light humor, metaphors, or fun comparisons while staying informative.",
	"Casual Chat": "Answer like you would in a relaxed conversation with a peer—natural, friendly, and relatable.",
	"Debug Mode": "Answer step-by-step, like explaining your reasoning while debugging code.",
	"Analogy Mode": "Always explain with analogies and metaphors.",
	"Concise": "Answer as briefly and to the point as possible, with no extra fluff."
}

st.sidebar.markdown("## Choose Response Tone")
selected_tone = st.sidebar.radio(
	"",
	list(TONE_OPTIONS.keys()),
	index=0,
	key="tone_selector"
)

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
	# Get selected tone from session state (set by sidebar radio)
	tone = st.session_state.get("tone_selector", "Friendly")
	tone_instruction = TONE_OPTIONS.get(tone, "")
	# Compose system prompt: persona + tone
	if persona:
		system_prompt = f"{persona}\n\n{tone_instruction}"
	else:
		system_prompt = f"You are the person the questions are about.\n\n{tone_instruction}"
	user_prompt = f"{prompt}\n\nUse the following context to answer the question in first person. Strictly stay within the provided context.\n{context}"

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


# Use accent color and heading font for title
st.markdown(
	'<h1 style="color:#4A6572; font-family:Century Gothic, sans-serif;">AI</h1>',
	unsafe_allow_html=True
)


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
	EXCLUDE_FILES = {"embedded_files.txt", "persona_prompt.txt"}
	for fname in os.listdir(DB_DIR):
		if fname in EXCLUDE_FILES:
			continue
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


st.sidebar.markdown("## Knowledge Base")
if 'embedded_files' in st.session_state:
	for fname in st.session_state.embedded_files:
		st.sidebar.write(f"- {fname}")






def retrieve_context(query, k=10):
	if st.session_state.index is None:
		return ""
	query_emb = embed_texts([query], st.session_state.model)
	D, I = st.session_state.index.search(query_emb, k)
	return "\n".join([st.session_state.docs[i] for i in I[0] if i < len(st.session_state.docs)])

st.markdown(
	'<h3 style="color:#4A6572; font-family:Century Gothic, sans-serif;">Ask any question <b>About Ilanri</b>:</h3>',
	unsafe_allow_html=True
)
user_input = st.text_input("You:", value=st.session_state.get('user_input', ''), key="main_user_input")

if st.button("Send", key="main_send_button") and user_input:
	context = retrieve_context(user_input)
	with st.spinner("I'm thinking..."):
		answer = groq_chat(user_input, context)
	st.session_state['user_input'] = ""
	st.markdown(
		f'<div style="background: linear-gradient(135deg, #F2F6F8 0%, #E6ECF0 100%); '
		'border: 1.5px solid #4A6572; border-radius: 12px; padding: 1.2em 1.5em; '
		'margin: 1.2em 0; color: #2C3A47; font-size: 1.13em; font-family: ArialMTPro-Regular, Arial, sans-serif; '
		'box-shadow: 0 2px 12px rgba(74,101,114,0.08);">'
		f'{answer}'
		'</div>',
		unsafe_allow_html=True
	)


# --- Enhance Database Section (below main prompt/answer) ---
st.markdown("<hr style='border: 1.5px solid #D1D9DE; margin: 2em 0;'>", unsafe_allow_html=True)
st.markdown(
	'<h2 style="color:#4A6572; font-family:Century Gothic, sans-serif;">Enhance Knowledge Base</h2>',
	unsafe_allow_html=True
)
st.markdown(
	'<span style="color:#4A6572; font-family:ArialMTPro-Regular, Arial, sans-serif;">Upload files or add Q&A pairs to expand the knowledge base.</span>',
	unsafe_allow_html=True
)

uploaded_files = st.file_uploader(
	"",
	type=["txt", "pdf", "mp3", "wav", "ogg", "m4a"],
	accept_multiple_files=True,
	key="file_uploader_enhance"
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
		st.success(f"Added and saved {len(new_chunks)} new chunks.")


# --- LLM-generated suggested questions for Q&A section ---
import json
def get_llm_suggested_questions(persona, n=3):
	prompt = (
		f"You are a helpful assistant. Based on the following persona, suggest {n} personal, diverse, or random introspective, but short and fun questions that a user could add to a knowledge base as Q&A pairs. "
		f"Return only a JSON list of questions.\n\nPersona:\n{persona}"
	)
	headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
	data = {
		"model": "llama-3.3-70b-versatile",
		"messages": [
			{"role": "system", "content": prompt}
		]
	}
	try:
		response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=20)
		if response.status_code == 200:
			content = response.json()['choices'][0]['message']['content']
			# Try to parse as JSON list
			try:
				questions = json.loads(content)
				if isinstance(questions, list):
					return [str(q).strip() for q in questions][:n]
			except Exception:
				# Fallback: try to extract JSON array from text
				import re
				match = re.search(r'\[(.*?)\]', content, re.DOTALL)
				if match:
					items = match.group(1).split(',')
					return [item.strip(' "\n') for item in items if item.strip()][:n]
				# Fallback: split by lines
				return [line.strip('- ').strip() for line in content.split('\n') if line.strip()][:n]
		else:
			return ["What is a good question to add?", "What is a useful fact?", "What is a common FAQ?"]
	except Exception:
		return ["What is a good question to add?", "What is a useful fact?", "What is a common FAQ?"]


# Initialize or refresh suggested questions
if 'suggested_questions' not in st.session_state or st.session_state.get('refresh_suggested', True):
	persona = get_or_create_persona()
	st.session_state.suggested_questions = get_llm_suggested_questions(persona)
	st.session_state.refresh_suggested = False

# Function to update suggested questions after Q&A is added
def update_suggested_questions_qa(latest_answer=None):
	persona = get_or_create_persona()
	st.session_state.suggested_questions = get_llm_suggested_questions(persona)

suggestion_cols = st.columns(len(st.session_state.suggested_questions))
for i, q in enumerate(st.session_state.suggested_questions):
	if suggestion_cols[i].button(q, key=f"qa_suggested_{i}"):
		st.session_state.qa_question = q

if 'qa_question' not in st.session_state:
    st.session_state.qa_question = ""
if 'qa_answer' not in st.session_state:
    st.session_state.qa_answer = ""


with st.form("add_qa_form"):
	user_question = st.text_area("Question", key="qa_question")
	user_answer = st.text_area("Answer", value=st.session_state.qa_answer, key="qa_answer")
	submit_qa = st.form_submit_button("Add Q&A", key="submit_qa_button")
	if submit_qa and user_question.strip() and user_answer.strip():
		with st.spinner("Adding Q&A to knowledge base..."):
			qa_text = f"Q: {user_question.strip()}\nA: {user_answer.strip()}"
			st.session_state.docs.append(qa_text)
			st.session_state.chunk_file_map.append("manual_QA")
			embeddings = embed_texts(st.session_state.docs, st.session_state.model)
			st.session_state.embeddings = embeddings
			st.session_state.index = create_faiss_index(embeddings)
			save_docs_and_embeddings(st.session_state.docs, embeddings, DOCS_EMB_PATH)
			save_faiss_index(st.session_state.index, FAISS_INDEX_PATH)
		st.success("Q&A pair added to knowledge base.")
		update_suggested_questions_qa(user_answer)
		# Rerun to clear form fields safely
		st.rerun()



def retrieve_context(query, k=10):
	if st.session_state.index is None:
		return ""
	query_emb = embed_texts([query], st.session_state.model)
	D, I = st.session_state.index.search(query_emb, k)

	return "\n".join([st.session_state.docs[i] for i in I[0] if i < len(st.session_state.docs)])

