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
/* ══════════════════════════════════════════
   GLOBAL
══════════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main, .main > div {
    background-color: #0a0a0a !important;
    color: #ffffff !important;
}
.block-container {
    padding: 16px 24px 120px !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════════
   HIDE ALL STREAMLIT SIDEBAR TOGGLE BUTTONS
   — we never want them, on any screen size
══════════════════════════════════════════ */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
.st-emotion-cache-czk5ss,
.st-emotion-cache-1q1n0ol,
.st-emotion-cache-pkbazv {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* ══════════════════════════════════════════
   SIDEBAR — DESKTOP
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 0.5px solid #1e1e1e !important;
    min-width: 260px !important;
    max-width: 280px !important;
    width: 260px !important;
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
[data-testid="stFileUploaderDropzoneInstructions"] div span { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { display:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Only PDF";
    font-size: 11px; color: #555;
}
[data-testid="stMetric"] {
    background: #111 !important;
    border: 0.5px solid #1e1e1e !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
[data-testid="stMetricLabel"] p { color: #555 !important; font-size: 11px !important; }
[data-testid="stMetricValue"] { color: #fff !important; font-size: 18px !important; }

/* ══════════════════════════════════════════
   HIDE / SHOW STREAMLIT chrome
══════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* ══════════════════════════════════════════
   MOBILE NAV BAR (hidden on desktop)
══════════════════════════════════════════ */
.mobile-nav {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 52px;
    background: #0f0f0f;
    border-bottom: 0.5px solid #1e1e1e;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    z-index: 10000;
}
.mobile-nav-logo {
    display: flex; align-items: center; gap: 8px;
    font-size: 15px; font-weight: 500; color: #fff;
}
.mobile-nav-logo-icon {
    width: 28px; height: 28px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius: 7px;
    display: flex; align-items: center;
    justify-content: center; font-size: 14px;
}
.mobile-hamburger {
    width: 36px; height: 36px;
    background: #1a1a1a;
    border: 1px solid #6366f1;
    border-radius: 8px;
    display: flex; align-items: center;
    justify-content: center; cursor: pointer;
    font-size: 17px; color: #6366f1;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
}

/* ══════════════════════════════════════════
   MOBILE DRAWER
══════════════════════════════════════════ */
.mob-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 10001;
    backdrop-filter: blur(2px);
}
.mob-drawer {
    position: fixed;
    top: 0; left: 0;
    width: min(80vw, 300px);
    height: 100vh;
    background: #0f0f0f;
    border-right: 0.5px solid #1e1e1e;
    z-index: 10002;
    padding: 56px 16px 24px;
    overflow-y: auto;
    transform: translateX(-110%);
    transition: transform 0.28s cubic-bezier(0.4,0,0.2,1);
}
.mob-drawer.is-open {
    transform: translateX(0);
}
.mob-overlay.is-open {
    display: block;
}
.mob-drawer-close {
    position: absolute;
    top: 12px; right: 12px;
    width: 30px; height: 30px;
    background: #1e1e1e;
    border: 0.5px solid #333;
    border-radius: 6px;
    display: flex; align-items: center;
    justify-content: center; cursor: pointer;
    color: #888; font-size: 15px;
    -webkit-tap-highlight-color: transparent;
}
.mob-drawer-logo {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px;
}
.mob-drawer-logo-icon {
    width: 32px; height: 32px;
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border-radius: 8px;
    display: flex; align-items: center;
    justify-content: center; font-size: 16px;
}
.mob-drawer-logo-text {
    font-size: 16px; font-weight: 500; color: #fff;
}
.mob-divider { height: 0.5px; background: #1e1e1e; margin: 10px 0; }

/* ══════════════════════════════════════════
   CHAT MESSAGES
══════════════════════════════════════════ */
.msg-user {
    display: flex; justify-content: flex-end;
    margin: 0 0 6px 0;
}
.bubble-user {
    background: #6366f1; color: #fff;
    border-radius: 14px 14px 3px 14px;
    padding: 9px 14px; font-size: 13px;
    max-width: 72%; line-height: 1.5;
    word-break: break-word;
}
.msg-bot {
    display: flex; gap: 9px;
    align-items: flex-start; margin: 0 0 6px 0;
}
.bot-avatar {
    width: 28px; height: 28px;
    background: #1a1a1a;
    border: 0.5px solid #2a2a2a;
    border-radius: 8px;
    display: flex; align-items: center;
    justify-content: center;
    flex-shrink: 0; margin-top: 2px;
}
.bubble-bot {
    background: #111;
    border: 0.5px solid #1e1e1e;
    border-radius: 3px 14px 14px 14px;
    padding: 9px 14px; font-size: 13px;
    color: #ccc; max-width: 80%;
    line-height: 1.6; word-break: break-word;
}

/* ══════════════════════════════════════════
   THINKING DOTS
══════════════════════════════════════════ */
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
    0%,80%,100% { transform:translateY(0); opacity:0.4; }
    40% { transform:translateY(-5px); opacity:1; }
}

/* ══════════════════════════════════════════
   SOURCE PILLS
══════════════════════════════════════════ */
.source-row { display:flex; gap:5px; flex-wrap:wrap; margin-top:6px; }
.source-pill {
    display:inline-block;
    background:#1a1533; color:#a5b4fc;
    border:0.5px solid #312e81;
    border-radius:4px; font-size:10px;
    padding:2px 8px; font-family:monospace;
}

/* ══════════════════════════════════════════
   HERO SCREEN
══════════════════════════════════════════ */
.hero-wrap {
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    min-height:65vh; text-align:center; padding:0 16px;
}
.hero-title {
    font-size:22px; font-weight:500;
    color:#fff; letter-spacing:-0.5px; margin-bottom:8px;
}
.hero-sub { font-size:13px; color:#555; line-height:1.7; margin-bottom:24px; }
.feat-grid {
    display:grid; grid-template-columns:repeat(3,1fr);
    gap:8px; max-width:500px; margin:0 auto;
}
.feat-card {
    background:#111; border:0.5px solid #1e1e1e;
    border-radius:10px; padding:12px; text-align:left;
}
.feat-title { font-size:12px; font-weight:500; color:#fff; margin-bottom:3px; }
.feat-desc { font-size:11px; color:#555; line-height:1.4; }

/* ══════════════════════════════════════════
   SESSION HEADER
══════════════════════════════════════════ */
.session-label {
    font-size:10px; font-weight:500; color:#444;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;
}
.session-title {
    font-size:18px; font-weight:500;
    color:#fff; margin-bottom:4px; word-break:break-all;
}
.session-meta { font-size:11px; color:#444; margin-bottom:12px; }
.sdivider { height:0.5px; background:#1e1e1e; margin-bottom:14px; }

/* ══════════════════════════════════════════
   CHAT INPUT
══════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stBottom"] {
    background: #0a0a0a !important;
    padding-bottom: 16px !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
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

/* ══════════════════════════════════════════
   MOBILE MEDIA QUERY
   Only layout/sizing changes — zero design changes
══════════════════════════════════════════ */
@media screen and (max-width: 768px) {

    /* Show mobile nav bar */
    .mobile-nav { display: flex !important; }

    /* Push content below fixed nav */
    .block-container {
        padding: 64px 12px 130px !important;
    }

    /* Hide Streamlit sidebar entirely — drawer replaces it */
    [data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
    }

    /* Messages wider on mobile */
    .bubble-user { max-width: 88% !important; font-size: 13px !important; }
    .bubble-bot  { max-width: 92% !important; font-size: 13px !important; }

    /* Session title smaller */
    .session-title { font-size: 15px !important; }

    /* Hero cards 2-col on mobile */
    .feat-grid {
        grid-template-columns: 1fr 1fr !important;
        max-width: 100% !important;
        gap: 8px !important;
    }
    .hero-title { font-size: 20px !important; }
    .hero-sub   { font-size: 12px !important; }

    /* Chat input full width */
    [data-testid="stBottom"] {
        padding-left: 12px !important;
        padding-right: 12px !important;
        padding-bottom: 20px !important;
    }
    [data-testid="stChatInput"] > div {
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 14px !important;
        padding: 11px 12px !important;
    }
}
</style>

<!-- ══ MOBILE NAV BAR ══ -->
<div class="mobile-nav" id="mobileNav">
  <div class="mobile-nav-logo">
    <div class="mobile-nav-logo-icon">🤖</div>
    PDF.RAG
  </div>
  <div class="mobile-hamburger" id="hamburgerBtn" onclick="openDrawer()">☰</div>
</div>

<!-- ══ DRAWER OVERLAY ══ -->
<div class="mob-overlay" id="mobOverlay" onclick="closeDrawer()"></div>

<!-- ══ MOBILE DRAWER ══ -->
<div class="mob-drawer" id="mobDrawer">
  <div class="mob-drawer-close" onclick="closeDrawer()">✕</div>
  <div class="mob-drawer-logo">
    <div class="mob-drawer-logo-icon">🤖</div>
    <span class="mob-drawer-logo-text">PDF.RAG</span>
  </div>
  <div class="mob-divider"></div>
  <p style="font-size:12px;color:#888;line-height:1.6;margin:0">
    Use the <strong style="color:#a5b4fc">sidebar controls</strong> above the chat area to upload your PDF and start a new session.<br><br>
    Scroll up past the chat to find the upload section.
  </p>
  <div class="mob-divider" style="margin-top:16px"></div>
  <p style="font-size:10px;color:#333;text-align:center;letter-spacing:0.04em;margin-top:8px">
    ABHI
  </p>
</div>

<script>
function openDrawer() {
    document.getElementById('mobDrawer').classList.add('is-open');
    document.getElementById('mobOverlay').classList.add('is-open');
    document.body.style.overflow = 'hidden';
}
function closeDrawer() {
    document.getElementById('mobDrawer').classList.remove('is-open');
    document.getElementById('mobOverlay').classList.remove('is-open');
    document.body.style.overflow = '';
}
</script>
""", unsafe_allow_html=True)

# ── Robot SVG ──────────────────────────────────────────────────────────────────

ROBOT_SVG = '''<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="#6366f1" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
  <rect x="2" y="10" width="20" height="11" rx="2"/>
  <circle cx="12" cy="4" r="2"/>
  <line x1="12" y1="6" x2="12" y2="10"/>
  <circle cx="8"  cy="15" r="1.2" fill="#6366f1" stroke="none"/>
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
    "messages":         [],
    "vectorstore":      None,
    "pdf_name":         None,
    "pdf_pages":        0,
    "collection_id":    None,
    "uploader_key":     0,
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
        st.error("GROQ_API_KEY not found in .env / Streamlit secrets.")
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
    loader  = PyPDFLoader(tmp_path)
    pages   = loader.load()
    chunks  = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n","\n","."," ",""],
    ).split_documents(pages)
    col = f"pdf_{uuid.uuid4().hex[:12]}"
    vs  = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name=col,
    )
    os.unlink(tmp_path)
    return vs, len(pages), col

def get_answer(question, vectorstore):
    raw      = vectorstore.similarity_search_with_score(question, k=5)
    filtered = [(d,s) for d,s in raw if s <= 2.0] or raw[:3]
    context  = "\n\n".join(
        f"[Page {d.metadata.get('page',0)+1}]\n{d.page_content.strip()}"
        for d,_ in filtered
    )
    prompt = f"""You are a helpful assistant answering questions from a document.
RULES:
- Answer directly and clearly
- NEVER mention "chunks", "document chunks", or internal terms
- Use bullet points only for lists
- If not in document: "❌ Not found in this document."

DOCUMENT:
{context}

QUESTION: {question}
ANSWER:"""
    try:
        return load_llm().invoke(prompt).content.strip(), filtered
    except Exception as e:
        return f"⚠️ {e}", []

def format_sources(docs_with_scores):
    seen, pills = set(), []
    for doc, score in docs_with_scores:
        page = doc.metadata.get("page", 0) + 1
        key  = f"p.{page}"
        if key not in seen:
            seen.add(key)
            sim = max(0.0, round(1-(score/2), 4))
            pills.append(
                f'<span class="source-pill">[{key}] sim {sim:.4f}</span>'
            )
    return '<div class="source-row">' + "".join(pills) + "</div>"

def render_user(text):
    return f'<div class="msg-user"><div class="bubble-user">{text}</div></div>'

def render_bot(text, sources=""):
    return (f'<div class="msg-bot">'
            f'<div class="bot-avatar">{ROBOT_SVG}</div>'
            f'<div><div class="bubble-bot">{text}</div>{sources}</div>'
            f'</div>')

# ── Sidebar (desktop) ──────────────────────────────────────────────────────────

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
        st.session_state.update({
            "messages": [], "vectorstore": None,
            "pdf_name": None, "pdf_pages": 0,
            "collection_id": None, "pending_question": None,
            "uploader_key": st.session_state.uploader_key + 1,
        })
        st.rerun()

    st.markdown(
        '<p style="font-size:10px;color:#444;text-transform:uppercase;'
        'letter-spacing:0.08em;margin:16px 0 8px">Upload PDF</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop PDF here", type="pdf",
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded and uploaded.name != st.session_state.pdf_name:
        with st.spinner("Processing…"):
            vs, n, col = process_pdf(uploaded, load_embeddings())
            st.session_state.update({
                "vectorstore": vs, "pdf_name": uploaded.name,
                "pdf_pages": n, "collection_id": col, "messages": [],
            })
        st.success("✅ Ready to chat!")

    if st.session_state.pdf_name:
        st.markdown(
            f'<p style="font-size:12px;color:#888;margin:10px 0 8px;'
            f'word-break:break-all;line-height:1.4">'
            f'📄 {st.session_state.pdf_name}</p>',
            unsafe_allow_html=True,
        )
        st.metric("Pages", st.session_state.pdf_pages)

    st.markdown(
        '<div style="height:0.5px;background:#1e1e1e;margin:20px 0 10px"></div>'
        '<p style="font-size:10px;color:#333;letter-spacing:0.04em;text-align:center">'
        'ABHI</p>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MOBILE — show controls inline (above chat) so users can upload on small screen
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="mob-inline-controls" style="display:none">_placeholder_</div>
<style>
@media screen and (max-width:768px){
  .mob-inline-controls{ display:block !important; }
}
</style>
""", unsafe_allow_html=True)

# Detect mobile via query param trick — render inline upload section
# We use st.expander so it collapses nicely on mobile after upload
with st.expander("⚙️ Upload PDF / New Session", expanded=not bool(st.session_state.pdf_name)):
    col1, col2 = st.columns([2, 1])
    with col1:
        mob_uploaded = st.file_uploader(
            "Upload your PDF",
            type="pdf",
            key=f"mob_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
        )
    with col2:
        if st.button("New session", key="mob_new_session"):
            st.session_state.update({
                "messages": [], "vectorstore": None,
                "pdf_name": None, "pdf_pages": 0,
                "collection_id": None, "pending_question": None,
                "uploader_key": st.session_state.uploader_key + 1,
            })
            st.rerun()

    if mob_uploaded and mob_uploaded.name != st.session_state.pdf_name:
        with st.spinner("Processing PDF…"):
            vs, n, col_id = process_pdf(mob_uploaded, load_embeddings())
            st.session_state.update({
                "vectorstore": vs, "pdf_name": mob_uploaded.name,
                "pdf_pages": n, "collection_id": col_id, "messages": [],
            })
        st.success(f"✅ Ready! {n} pages loaded.")

    if st.session_state.pdf_name:
        st.caption(f"📄 {st.session_state.pdf_name} — {st.session_state.pdf_pages} pages")

# Style the expander to match dark theme and hide on desktop
st.markdown("""
<style>
/* Dark-style expander */
[data-testid="stExpander"] {
    background: #111 !important;
    border: 0.5px solid #1e1e1e !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #888 !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
}
[data-testid="stExpander"] summary:hover { color: #a5b4fc !important; }

/* Hide this expander on DESKTOP — sidebar handles it there */
@media screen and (min-width: 769px) {
    [data-testid="stExpander"] { display: none !important; }
}
</style>
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
<div class="sdivider"></div>
""", unsafe_allow_html=True)

# ── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(render_user(msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(
            render_bot(msg["content"], msg.get("sources_html","")),
            unsafe_allow_html=True,
        )

# ── Pending question ───────────────────────────────────────────────────────────

if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    ph = st.empty()
    ph.markdown(THINKING_HTML, unsafe_allow_html=True)
    answer, docs = get_answer(q, st.session_state.vectorstore)
    sources_html = format_sources(docs) if docs else ""
    ph.markdown(render_bot(answer, sources_html), unsafe_allow_html=True)
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
