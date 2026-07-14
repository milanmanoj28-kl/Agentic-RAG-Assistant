"""
Streamlit Chat UI — Agentic RAG Assistant
==========================================

This is a thin frontend that talks to your existing FastAPI backend
(LangChain + ChromaDB + Groq LLaMA 3.1 + Tavily). It does NOT reimplement
any RAG/agent logic — it just calls your API endpoints.

>>> ADJUST THESE TO MATCH YOUR ACTUAL FASTAPI ROUTES <<<
The endpoint paths/payloads below are reasonable assumptions. If your
FastAPI app uses different route names or request/response schemas,
update the three functions in the "API CLIENT" section — everything
else (UI, chat history, session state) will keep working unchanged.

Run with:
    streamlit run streamlit_app.py

Requires:
    pip install streamlit requests
"""

import os
import uuid
import requests
import streamlit as st

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DEFAULT_BACKEND_URL = os.environ.get("RAG_BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------------------
# STYLING — gives the app a more standard, polished chat-product feel
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Center and cap the main content width like most chat apps */
        .block-container {
            max-width: 820px;
            padding-top: 2rem;
            padding-bottom: 6rem;
        }

        /* Header bar */
        .app-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.2rem;
        }
        .app-header .title {
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0;
        }
        .status-dot {
            height: 8px;
            width: 8px;
            border-radius: 50%;
            background-color: #2ecc71;
            display: inline-block;
        }
        .app-subtitle {
            color: rgba(255,255,255,0.55);
            font-size: 0.85rem;
            margin-top: 0;
            margin-bottom: 1.5rem;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            padding: 0.9rem 1.1rem;
            border-radius: 14px;
            margin-bottom: 0.6rem;
        }

        /* Empty-state suggestion chips */
        .suggestion-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1rem; }

        /* Sidebar section headers */
        section[data-testid="stSidebar"] h3 { font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# API CLIENT — adjust these to match your actual FastAPI routes
# ----------------------------------------------------------------------

def send_chat_message(backend_url: str, session_id: str, message: str, use_web_search: bool):
    """
    Actual endpoint: POST {backend_url}/chat
    Actual payload:  {"question": str}
    Actual response: {"answer": str}

    Note: this backend has no session_id or web_search toggle — it's a single
    global knowledge base with no per-session isolation. session_id and
    use_web_search are accepted here for UI compatibility but not sent.
    """
    resp = requests.post(
        f"{backend_url}/chat",
        json={"question": message},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    answer = data.get("answer", "")
    sources = []  # this backend doesn't return sources
    return answer, sources


def upload_document(backend_url: str, session_id: str, file):
    """
    Actual endpoint: POST {backend_url}/upload
    Actual payload:  multipart form with just "file" (no session_id field exists)
    Actual response: {"filename": str, "chunks_added": int, "message": str}
    """
    files = {"file": (file.name, file.getvalue(), file.type)}
    resp = requests.post(f"{backend_url}/upload", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def reset_conversation(backend_url: str, session_id: str):
    """
    This backend has no /reset endpoint (confirmed via /docs — only /, /chat,
    /upload exist) and no session concept at all. This call will always fail
    silently and just clears local chat history in the UI; the backend's
    underlying knowledge base is not affected by this button.
    """
    try:
        requests.post(f"{backend_url}/reset", json={"session_id": session_id}, timeout=10)
    except requests.RequestException:
        pass


# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str, "sources": [...]}

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🤖 Agentic RAG Assistant")
    st.caption("LangChain · ChromaDB · Groq LLaMA 3.1 · Tavily")

    st.session_state.backend_url = st.text_input(
        "Backend URL", value=st.session_state.backend_url, help="Your FastAPI server address"
    )

    use_web_search = st.toggle("Enable web search (Tavily)", value=True)

    st.divider()
    st.subheader("📄 Upload a document")
    uploaded_file = st.file_uploader(
        "Add a file to this session's knowledge base",
        type=["pdf", "txt", "docx", "md", "csv"],
    )
    if uploaded_file is not None:
        if st.button("Upload", use_container_width=True):
            with st.spinner("Uploading and indexing..."):
                try:
                    result = upload_document(
                        st.session_state.backend_url, st.session_state.session_id, uploaded_file
                    )
                    st.success(f"Uploaded: {result.get('filename', uploaded_file.name)}")
                except requests.RequestException as e:
                    st.error(f"Upload failed: {e}")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        reset_conversation(st.session_state.backend_url, st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session ID: `{st.session_state.session_id[:8]}`")


# ----------------------------------------------------------------------
# MAIN CHAT AREA
# ----------------------------------------------------------------------
# Quick, non-blocking backend health check so the status dot reflects reality
def _backend_is_reachable(url: str) -> bool:
    try:
        requests.get(url, timeout=2)
        return True
    except requests.RequestException:
        return False


backend_up = _backend_is_reachable(st.session_state.backend_url)
status_color = "#2ecc71" if backend_up else "#e74c3c"
status_text = "Connected" if backend_up else "Backend unreachable"

st.markdown(
    f"""
    <div class="app-header">
        <span class="title">💬 Agentic RAG Assistant</span>
    </div>
    <div class="app-subtitle">
        <span class="status-dot" style="background-color:{status_color};"></span>
        &nbsp;{status_text} · session <code>{st.session_state.session_id[:8]}</code>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render chat history
if not st.session_state.messages:
    st.markdown(
        "Ask a question about an uploaded document, or toggle on web search "
        "in the sidebar to pull in live results."
    )
    example_prompts = [
        "Summarize the document I just uploaded",
        "What are the key takeaways from this file?",
        "Search the web for the latest on this topic",
    ]
    cols = st.columns(len(example_prompts))
    for col, example in zip(cols, example_prompts):
        if col.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Chat input — also accepts a prompt pre-filled by clicking an example chip
prompt = st.chat_input("Ask something about your documents or the web...")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = send_chat_message(
                    st.session_state.backend_url,
                    st.session_state.session_id,
                    prompt,
                    use_web_search,
                )
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"- {src}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except requests.RequestException as e:
                error_msg = f"⚠️ Could not reach backend at `{st.session_state.backend_url}`: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})