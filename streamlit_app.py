import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(
    page_title="PDF.RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main, .main > div {
    background-color: #0a0a0a !important;
    color: #ffffff !important;
}
.block-container {
    padding: 16px 16px 120px !important;
    max-width: 100% !important;
}

/* ── Hide EVERY Streamlit sidebar toggle button ── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
.st-emotion-cache-czk5ss,
.st-emotion-cache-1q1n0ol,
.st-emotion-cache-pkbazv,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-header"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* ── Sidebar — always open, never collapses ── */
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 0.5px solid #1e1e1e !important;
    width: 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;
    flex-shrink: 0 !important;
    transform: none !important;
    position: relative !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.25rem 1rem 2rem !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 0.5px solid #2a2a2a !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #222 !important;
    border-color: #6366f1 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #111 !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Only PDF";
    font-size: 11px;
    color: #555;
}
[data-testid="stMetric"] {
    background: #111 !important;
    border: 0.5px solid #1e1e1e !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
[data-testid="stMetricLabel"] p { color: #555 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 18px !important; }

/* ── Mobile: sidebar becomes top bar ── */
@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        border-right: none !important;
        border-bottom: 0.5px solid #1e1e1e !important;
        position: relative !important;
        transform: none !important;
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1rem !important;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 10px !important;
        align-items: center !important;
    }
    [data-testid="stAppViewContainer"] {
        flex-direction: column !important;
    }
    .block-container {
        padding: 12px 12px 140px !important;
    }
    .bubble-user { max-width: 88% !important; }
    .bubble-bot { max-width: 92% !important; }
    .session-title { font-size: 15px !important; }
    .feat-grid {
        grid-template-columns: 1fr 1fr !important;
        max-width: 100% !important;
    }
    .hero-title { font-size: 20px !important; }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* ── Chat messages ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0 0 6px 0;
}
.bubble-user {
    background: #6366f1;
    color: #fff;
    border-radius: 14px 14px 3px 14px;
    padding: 9px 14px;
    font-size: 13px;
    max-width: 72%;
    line-height: 1.5;
    word-break: break-word;
}
.msg-bot {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    margin: 0 0 6px 0;
}
.bot-avatar {
    width: 28px; height: 28px;
    background: #1a1a1a;
    border: 0.5px solid #2a2a2a;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
}
.bubble-bot {
    background: #111;
    border: 0.5px solid #1e1e1e;
    border-radius: 3px 14px 14px 14px;
    padding: 9px 14px;
    font-size: 13px; color: #ccc;
    max-width: 80%; line-height: 1.6;
    word-break: break-word;
}

/* ── Thinking dots ── */
.thinking-dots {
    display: inline-flex; gap: 4px;
    align-items: center; padding: 4px 0;
}
.thinking-dots span {
    width: 6px; height: 6px;
    background: #6366f1; border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-5px); opacity: 1; }
}

/* ── Source pills ── */
.source-row { display: flex; gap: 5px; flex-wrap: wrap; margin-top: 6px; }
.source-pill {
    display: inline-block;
    background: #1a1533; color: #a5b4fc;
    border: 0.5px solid #312e81;
    border-radius: 4px; font-size: 10px;
    padding: 2px 8px; font-family: monospace;
}

/* ── Hero ── */
.hero-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 65vh; text-align: center;
    padding: 0 16px;
}
.hero-title { font-size: 22px; font-weight: 500; color: #fff; letter-spacing: -0.5px; margin-bottom: 8px; }
.hero-sub { font-size: 13px; color: #555; line-height: 1.7; margin-bottom: 24px; }
.feat-grid {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 8px; max-width: 500px; margin: 0 auto;
}
.feat-card { background: #111; border: 0.5px solid #1e1e1e; border-radius: 10px; padding: 12px; text-align: left; }
.feat-title { font-size: 12px; font-weight: 500; color: #fff; margin-bottom: 3px; }
.feat-desc { font-size: 11px; color: #555; line-height: 1.4; }

/* ── Session header ── */
.session-label { font-size: 10px; font-weight: 500; color: #444; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.session-title { font-size: 18px; font-weight: 500; color: #fff; margin-bottom: 4px; word-break: break-all; }
.session-meta { font-size: 11px; color: #444; margin-bottom: 12px; }
.divider { height: 0.5px; background: #1e1e1e; margin-bottom: 14px; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stBottom"] {
    background: #0a0a0a !important;
    padding-bottom: 16px !important;
}
[data-testid="stBottom"] > div { background: #0a0a0a !important; }
[data-testid="stChatInput"] > div {
    background: #111 !important;
    border-radius: 14px !important;
    border: 1.5px solid #6366f1 !important;
    box-shadow: 0 0 14px #6366f128 !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 22px #6366f140 !important;
}
[data-testid="stChatInput"] textarea {
    background: #111 !important;
    color: #e0e0e0 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: 14px !important;
    font-size: 13px !important;
    padding: 12px 14px !important;
    caret-color: #6366f1 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #444 !important; }
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border-radius: 9px !important;
    margin-right: 5px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Robot SVG ──────────────────────────────────────────────────────────────────

ROBOT_SVG = '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="#6366f1" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="2" y="10" width="20" height="11" rx="2"/>
  <circle cx="12" cy="4" r="2"/>
  <line x1="12" y1="6" x2="12" y2="10"/>
  <circle cx="8" cy="15" r="1.2" fill="#6366f1" stroke="none"/>
  <circle cx="16" cy="15" r="1.2" fill="#6366f1" stroke="none"/>
  <line x1="9" y1="19" x2="15" y2="19" stroke-width="1.5"/>
</svg>'''

THINKING_HTML = f'''<div class="msg-bot">
  <div class="bot-avatar">{ROBOT_SVG}</div>
  <div><div class="bubble-bot">
    <div class="thinking-dots"><span></span><span></span><span></span></div>
  </div></div>
</div>'''

# ── Session state ──────────────────────────────────────────────────────────────

for key, default in {
    "messages": [],
    "vectorstore": None,
    "pdf_name": None,
    "pdf_pages": 0,
    "collection_id": None,
    "uploader_key": 0,
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Core functions ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner=False)
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found.")
        st.stop()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0,
        max_tokens=1024,
    )

def process_pdf(uploaded_file, embeddings):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    col = f"pdf_{uuid.uuid4().hex[:12]}"
    vs = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=col,
    )
    os.unlink(tmp_path)
    return vs, len(pages), col

def get_answer(question, vectorstore):
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)
    filtered = [(d, s) for d, s in docs_with_scores if s <= 2.0]
    if not filtered:
        filtered = docs_with_scores[:3]
    parts = []
    for doc, _ in filtered:
        page = doc.metadata.get("page", 0) + 1
        parts.append(f"[Page {page}]\n{doc.page_content.strip()}")
    context = "\n\n".join(parts)

    prompt = f"""You are a helpful assistant answering questions based on a document.

RULES:
- Answer directly and clearly
- NEVER mention "chunks", "document chunks", or internal processing
- NEVER say "based on the provided..." phrases
- Use bullet points only when listing multiple items
- If not in document say exactly: "❌ Not found in this document."

DOCUMENT:
{context}

QUESTION: {question}

ANSWER:"""

    llm = load_llm()
    try:
        return llm.invoke(prompt).content.strip(), filtered
    except Exception as e:
        return f"⚠️ Error: {str(e)}", []

def format_sources(docs_with_scores):
    seen = set()
    pills = []
    for doc, score in docs_with_scores:
        page = doc.metadata.get("page", 0) + 1
        key = f"p.{page}"
        if key not in seen:
            seen.add(key)
            sim = max(0.0, round(1 - (score / 2), 4))
            pills.append(f'<span class="source-pill">[{key}] sim {sim:.4f}</span>')
    return '<div class="source-row">' + "".join(pills) + "</div>"

def render_user(text):
    return f'<div class="msg-user"><div class="bubble-user">{text}</div></div>'

def render_bot(text, sources_html=""):
    return f'''<div class="msg-bot">
  <div class="bot-avatar">{ROBOT_SVG}</div>
  <div><div class="bubble-bot">{text}</div>{sources_html}</div>
</div>'''

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;padding-top:4px">
      <div style="width:32px;height:32px;border-radius:8px;
        background:linear-gradient(135deg,#6366f1,#8b5cf6);
        display:flex;align-items:center;justify-content:center;font-size:16px">🤖</div>
      <span style="font-size:16px;font-weight:500;color:#fff;letter-spacing:-0.3px">PDF.RAG</span>
    </div>
    <div style="height:0.5px;background:#1e1e1e;margin:12px 0"></div>
    """, unsafe_allow_html=True)

    if st.button("＋  New session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.pdf_name = None
        st.session_state.pdf_pages = 0
        st.session_state.collection_id = None
        st.session_state.pending_question = None
        st.session_state.uploader_key += 1
        st.rerun()

    st.markdown("""
    <p style="font-size:10px;color:#444;text-transform:uppercase;
    letter-spacing:0.08em;margin:16px 0 8px">Upload PDF</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop PDF here", type="pdf",
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF…"):
            emb = load_embeddings()
            vs, n_pages, col_id = process_pdf(uploaded_file, emb)
            st.session_state.vectorstore = vs
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_pages = n_pages
            st.session_state.collection_id = col_id
            st.session_state.messages = []
        st.success("✅ Ready to chat!")

    if st.session_state.pdf_name:
        st.markdown(
            f'<p style="font-size:12px;color:#888;margin:10px 0 8px;'
            f'word-break:break-all;line-height:1.4">'
            f'📄 {st.session_state.pdf_name}</p>',
            unsafe_allow_html=True)
        st.metric("Pages", st.session_state.pdf_pages)

    st.markdown("""
    <div style="height:0.5px;background:#1e1e1e;margin:20px 0 10px"></div>
    <p style="font-size:10px;color:#333;letter-spacing:0.04em;text-align:center">
    GROQ · LLAMA 3.1 · MINILM · CHROMADB</p>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────

if not st.session_state.pdf_name:
    st.markdown("""
    <div class="hero-wrap">
      <div style="font-size:44px;margin-bottom:14px">🤖</div>
      <div class="hero-title">Chat with any PDF</div>
      <div class="hero-sub">Upload a document and ask anything.<br>
      Get answers with exact page citations.</div>
      <div class="feat-grid">
        <div class="feat-card">
          <div style="font-size:18px;margin-bottom:8px">⬆️</div>
          <div class="feat-title">Upload PDF</div>
          <div class="feat-desc">Drop any PDF from the sidebar</div>
        </div>
        <div class="feat-card">
          <div style="font-size:18px;margin-bottom:8px">💬</div>
          <div class="feat-title">Ask anything</div>
          <div class="feat-desc">Natural language queries</div>
        </div>
        <div class="feat-card">
          <div style="font-size:18px;margin-bottom:8px">🔖</div>
          <div class="feat-title">Page sources</div>
          <div class="feat-desc">Exact page citations</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Session header ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="session-label">Session</div>
<div class="session-title">{st.session_state.pdf_name}</div>
<div class="session-meta">
  📄 {st.session_state.pdf_pages} pages &nbsp;·&nbsp; Answers grounded in your PDF
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(render_user(msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(
            render_bot(msg["content"], msg.get("sources_html", "")),
            unsafe_allow_html=True,
        )

# ── Pending question ───────────────────────────────────────────────────────────

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    placeholder = st.empty()
    placeholder.markdown(THINKING_HTML, unsafe_allow_html=True)
    answer, docs = get_answer(question, st.session_state.vectorstore)
    sources_html = format_sources(docs) if docs else ""
    placeholder.markdown(render_bot(answer, sources_html), unsafe_allow_html=True)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources_html": sources_html,
    })

# ── Chat input ─────────────────────────────────────────────────────────────────

question = st.chat_input("Type your question and press Enter…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(render_user(question), unsafe_allow_html=True)
    st.session_state.pending_question = question
    st.rerun()
