# AI

### About Ilanri(AI) is a context-aware chatbot agent built to answer personal questions about Ilanri van Rossum

It leverages a custom knowledge base of personal documents, powered by a
Retrieval-Augmented Generation (RAG) system to provide authentic,
grounded answers in my voice.

[[Live Demo]] (https://aboutilanri.streamlit.app/)

[[Video Walkthrough]](https://drive.google.com/file/d/1dmHc96PUd5LyiCUZnwmiRRv9LApgGxf3/view?usp=sharing)

## System Setup & Design Choices

### Features

- **Interactive Chat Interface:** A clean Streamlit UI for natural Q&A.
- **Dynamic Knowledge Base:** Ingest and query PDFs, TXT files, and even audio transcripts (mp3, wav, ogg, m4a) via AssemblyAI.
- **Smart Document Processing:** Automated chunking, embedding, and storage in a FAISS vector store for efficient semantic search.
- **Tone Switcher:** Switch between different conversational modes (e.g., Professional, Storytelling, Concise) to hear answers in different styles.
- **AI-Powered Suggestions:** The LLM suggests relevant questions for the user to ask and answer, helping to enrich the knowledge base more effectively.

### Setup & Configuration

1. **Install Dependencies:** `pip install -r requirements.txt`
2. **Set API Keys:** Ensure your environment variables are set:

- `GROQ_API_KEY`: Your Groq API Key.
- `ASSEMBLYAI_API_KEY`: Your AssemblyAI key for audio transcription.

3. **Run the app:** `streamlit run app.py`
4. **Build Knowledge Base:** Use the sidebar to drag-and-drop your PDF, TXT, or audio files. The system will process them automatically.

- **Note:** If you update the core `intro.txt` file, delete the cached `persona_prompt.txt` to force the persona to be regenerated on the next run.

### Debugging

To enable logging of essential actions to use for understanding the system:

1. Set the environment variable before running Streamlit:

   ```bash
   export DEBUG=1
   streamlit run app.py
   ```

   To disable logging, use:

   ```bash
   export DEBUG=0
   streamlit run app.py
   ```

2. All logs are written to `debug.log` in the project directory. This file contains only:
   - User queries
   - Number of context chunks retrieved
   - Success or failure of major actions (e.g., file saves, persona construction, LLM calls)
   - Errors (if any)

3. To monitor logs in real time, use:

   ```bash
   tail -f debug.log
   ```

### Architecture & Technology Choices

The system is a classic RAG pipeline designed for rapid, AI-native development:

1. **Data Ingestion:** Files are uploaded, transcribed (if audio), cleaned and split into chunks.
2. **Embedding & Indexing:** Text chunks are embedded using the `all-MiniLM-L6-v2` model and stored in a FAISS index for fast similarity search.
3. **Retrieval & Generation:** For a query, relevant chunks are retrieved from FAISS. These, along with a system prompt defining the agent's persona and tone, are sent to the Groq LLM to generate a contextual answer.

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend/Backend** | Streamlit | Streamlit was chosen for its rapid prototyping capabilities, allowing me to build a deployable AI-native UI in hours and focus my effort on the innovative RAG and agentic logic.|
| **Vector Database** | FAISS | Selected over Chroma/Weaviate for its simplicity, speed, and because it runs locally.  |
| **Embeddings** | SentenceTransformers | The 'all-MiniLM-L6-v2' model offers a great speed/quality balance. |
| **LLM API** | Groq | Chosen for its incredibly fast inference speeds, providing a snappy user experience. |
| **Audio Processing** | AssemblyAI | Used a robust cloud API (instead of local libs like pydub) to avoid complex system dependencies and ensure accurate transcription. |
| **PDF Processing** | PyPDF | PyPDF was selected for its simplicity and reliability as a pure-Python library, providing robust PDF text extraction without external system dependencies, which ensured a smooth and consistent setup process across different environments.|

I chose to use external APIs for speed in this prototype, consciously accepting the trade-off since I'm only sending my own data for this specific demonstration.

## Sample questions and expected answers

| Question | Expected Answer Style (in Interview Mode) | Actual Output |
| :--- | :--- | :--- |
| "What kind of engineer are you?" | A concise summary based on my CV and intro, mentioning my focus on AI/ML, software development, and problem-solving. It would cite specific projects or skills. | I'm not an engineer by traditional definition, but rather a soon-to-be data science graduate. With a strong foundation in mathematics and a passion for coding, I've developed skills in data analysis, machine learning, and programming languages such as Python, R, and SQL. My experience in data science has equipped me with the ability to collect, analyze, and interpret complex data, making me a driven and enthusiastic data science student. |
| "What are your strongest technical skills?" | A list or paragraph highlighting key technologies (e.g., Python, RAG, LLMs, TensorFlow) mentioned in my CV and project docs, with context on how I've used them. | My strongest technical skills include proficiency in programming languages such as Python, with expertise in libraries like NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn. I'm also skilled in R, SQL, Java, JavaScript, and C. Additionally, I have experience with data science and machine learning tools, including data cleaning and preprocessing, data analysis, data visualization, and machine learning algorithms. I'm familiar with TensorFlow and PyTorch, and have a solid understanding of statistical analysis and hypothesis testing. Furthermore, I have experience with database and web development, including PostgreSQL, React, and REST APIs. |
| "Tell me about a project you're proud of." | A more narrative answer pulling from a README or project description, explaining the goal, what I built, and the outcome, told in a reflective tone. | I'm really proud of the work I did on the SharePoint Restructuring Project at Finovate, Somerset West. From December 2023 to February 2024, I utilized my analytical skills to conduct an in-depth analysis of existing data structures and devised a comprehensive roadmap for cleanup and reorganization. I solely managed the execution of the restructuring process, ensuring accuracy and efficiency, with periodic reviews from the COO. This project was a great opportunity for me to apply my skills in a real-world setting and drive meaningful change. I'm excited to leverage my experience and skills to take on new challenges and continue growing as a data science professional. |
| "What do you value in a team culture?" | Not explicitely mentioned in the context, but something along these lines: collaborative, supportive, and value-driven. | As a data science student, I value a team culture that prioritizes collaboration, open communication, and mutual respect. From my experience working on projects like the Data School Hackathon, where my team won the Best Women's Team award, I've seen firsthand how a supportive and inclusive environment can foster creativity, innovation, and collective growth. In such a culture, individuals feel empowered to share their ideas, learn from each other's strengths and weaknesses, and work together towards a common goal. I believe this kind of environment enables teams to tackle complex problems more effectively, drive meaningful impact, and celebrate each other's successes. Ultimately, a positive and collaborative team culture is essential for achieving exceptional results and creating a rewarding experience for all members involved. |

## Improvements with more time

With more time, I would focus on enhancing the system in the following key areas:

- **Implement Conversational Memory:** Add short-term memory to track dialogue context within a session, allowing for more natural multi-turn conversations and follow-up questions.

- **Advanced RAG Framework Optimization:**
  - **Fine-tune Retrieval Granularity:** Experiment with dynamic chunk sizing and overlap to better match query types.
  - **Refine Chunking Strategy:** Develop a more sophisticated strategy that adapts based on document type (e.g., code, prose, bullet points) and the evolving knowledge base.
  - **Hybrid Search in combination with re-ranking:** Implement a combination of semantic and keyword-based search to improve recall for specific names or technical terms.

- **Enhance Prompting Strategies:**
  - **Few-Shot Prompting:** Provide example question-answer pairs for each tone/mode to guide the LLM towards more consistent and higher-quality outputs.

- **Knowledge Base Enrichment & Management:**
  - **Metadata Tagging:** Label documents and chunks with descriptive attributes (e.g., `project:project_name`, `topic:debugging`, `type:code_sample`) to enable filtered and more precise retrieval.
  - **Ingest Richer Personal Data:**
    - Integrate Spotify API to analyze and include favorite songs or playlists as cultural reference points.
    - Ingest YouTube watch history or playlists to understand content consumption and interests.
    - Process anonymized chains of WhatsApp messages or emails (with personal data redacted) to better model language use and tone in different social contexts.
    - Incorporate any other form of personal writing (journals, design docs, brainstorming notes) to better capture my authentic thought process.

- **Expand Knowledge Sourcing:**
  - **Controlled Web Search:** Add a safe, privacy-conscious functionality to perform web searches (e.g., via a SERP API) to gather and incorporate recent, public information from outside the provided dataset.

- **Polish User Experience:**
  - **Refine UI/UX:** Improve the visual design by refining the color scheme for better accessibility and a less jarring, more professional appearance.

- **Rigorous Testing & Evaluation:**
  - Conduct extensive A/B testing on context window usage, optimal chunking parameters, and alternative embedding models to maximize answer quality and relevance for this specific use-case.

## Prompt Engineering & System Prompts

This project uses several types of prompts to guide the LLM and control the chatbot's behavior:

- **Persona Prompt:**
  - Generated from `intro.txt` and cached in `persona_prompt.txt`.
  - Defines the chatbot's personality, background, and answer style to ensure responses are grounded in Ilanri's authentic voice.
  - Example: *"Respond as Ilanri, a curious and determined 22-year-old engineer..."*

- **Tone/Mode Prompt:**
  - Selected by the user in the sidebar (e.g., Professional, Mentor, Playful, Concise).
  - Appends a short instruction to the system prompt to control the style and format of answers.
  - Example: *"Answer concisely, professionally, and highlight achievements as if in a job interview."*

- **Retrieval-Augmented Prompt:**
  - For each user query, the system retrieves relevant context chunks from the knowledge base and includes them in the prompt.
  - The LLM is instructed to answer strictly using the provided context, in first person.
  - Example: *"Use the following context to answer the question in first person. Strictly stay within the provided context. [context]"*

- **LLM-Suggested Questions Prompt:**
  - The LLM is prompted to generate a list of diverse, fun, or introspective questions that could be added to the knowledge base as Q&A pairs.
  - Example: *"Based on the following persona, suggest 3 personal, diverse, or random introspective, but short and fun questions..."*

- **Q&A Addition Prompt:**
  - When a user adds a new Q&A pair, the question and answer are stored as a chunk in the knowledge base for future retrieval.

These prompt strategies ensure the chatbot is context-aware, consistent in tone, and able to both answer and suggest new knowledge base content effectively.

---

## “Show Your Thinking” artifacts

Please refer to my `db/rationale.pdf` document to follow my thinking. I have commited to git everytime after succesfully implementing a subsection of this project and in the commit message I detailed what I prompted to co-pilot and what I manually edited. This help to track the prompt to the exact code in my project on a high-level. I have the full Co-pilot prompt history at the end of `rationale.pdf`. The manual coding mostly consisted of bug fixing and tweaking.
I hope you enjoy my "thinking"!
