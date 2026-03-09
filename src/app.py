"""
Healthcare RAG — Streamlit Frontend
Professional chat UI for querying clinical documents via the FastAPI backend.

Run with:
    streamlit run src/app.py
Requires the FastAPI backend running on localhost:8000.
"""

from __future__ import annotations

import sys

# ── stdout encoding (Windows cp1252 → UTF-8) ─────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"
APP_TITLE = "Healthcare RAG Assistant"
APP_ICON = "🏥"
REQUEST_TIMEOUT = 90  # seconds

WELCOME_MESSAGE = (
    "👋 **Welcome to the Healthcare RAG Assistant.**\n\n"
    "I answer clinical questions **exclusively** from indexed WHO, CDC, and NIH "
    "guidelines. Every claim includes a paragraph-level citation.\n\n"
    "Try asking:\n"
    "- *What are the WHO diagnostic criteria for diabetes mellitus?*\n"
    "- *What is the first-line treatment for hypertension?*\n"
    "- *What are the recommended childhood immunization schedules?*\n\n"
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "Always consult a qualified healthcare professional."
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/your-org/healthcare-rag",
        "Report a bug": "https://github.com/your-org/healthcare-rag/issues",
        "About": "Healthcare RAG — Evidence-based clinical Q&A",
    },
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — Healthcare blue theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Global ──────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background-color: #F0F4F8;
}
[data-testid="stSidebar"] {
    background-color: #1B4F72;
    color: white;
}
[data-testid="stSidebar"] * {
    color: white !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.2);
}
[data-testid="stSidebar"] .stButton > button {
    background-color: rgba(255,255,255,0.15);
    color: white !important;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 8px;
    transition: background-color 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(255,255,255,0.25);
}

/* ── Hero Banner ─────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1B4F72 0%, #2980B9 100%);
    color: white;
    padding: 1.4rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 20px rgba(27, 79, 114, 0.25);
}
.hero-banner h1 {
    font-size: 1.65rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.4px;
}
.hero-banner p {
    font-size: 0.88rem;
    opacity: 0.88;
    margin: 0;
}

/* ── Status Badges ───────────────────────────────────── */
.badge-online {
    display: inline-block;
    background: #27AE60;
    color: white !important;
    padding: 0.2rem 0.65rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-offline {
    display: inline-block;
    background: #E74C3C;
    color: white !important;
    padding: 0.2rem 0.65rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ── Metric Cards ────────────────────────────────────── */
.metric-card {
    background: rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin: 0.4rem 0;
    border-left: 3px solid rgba(255,255,255,0.4);
}
.metric-label { font-size: 0.72rem; opacity: 0.75; margin: 0; }
.metric-value { font-size: 1.1rem; font-weight: 700; margin: 0; }

/* ── Citation Cards ──────────────────────────────────── */
.citation-card {
    background: #EBF5FB;
    border-left: 4px solid #2980B9;
    border-radius: 6px;
    padding: 0.55rem 0.9rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #1B4F72;
    line-height: 1.5;
}
.citation-number {
    font-weight: 700;
    color: #2980B9;
    margin-right: 0.4rem;
}
.citation-source { font-weight: 600; }
.citation-location { color: #5D6D7E; font-size: 0.78rem; }

/* ── Declined Warning ────────────────────────────────── */
.declined-card {
    background: #FEF9E7;
    border-left: 4px solid #F39C12;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 0.5rem;
    color: #784212;
    font-size: 0.88rem;
}
.declined-card b { color: #D35400; }

/* ── Timing Badge ────────────────────────────────────── */
.timing-badge {
    display: inline-block;
    background: #F8F9FA;
    border: 1px solid #DEE2E6;
    border-radius: 10px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    color: #6C757D;
    margin-top: 0.4rem;
}

/* ── Chat input ──────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
}

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #D6EAF8;
    border-radius: 8px;
}

/* ── Scrollable chat area ────────────────────────────── */
.main-chat {
    max-height: calc(100vh - 220px);
    overflow-y: auto;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ─────────────────────────────────────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def fetch_health() -> dict | None:
    """Fetch /health from the FastAPI backend (cached 15 s)."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def call_query_api(question: str) -> dict:
    """POST /query and return the JSON response dict."""
    r = requests.post(
        f"{API_BASE}/query",
        json={"question": question},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
#  Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_citations(citations: list[dict]) -> None:
    """Render a styled reference block inside a chat message."""
    if not citations:
        return
    with st.expander(f"📚 {len(citations)} source{'s' if len(citations) != 1 else ''}", expanded=True):
        for c in citations:
            st.markdown(
                f"""
<div class="citation-card">
  <span class="citation-number">[{c['number']}]</span>
  <span class="citation-source">{c['source']}</span>
  <span class="citation-location"> — Page {c['page']}</span>
</div>
""",
                unsafe_allow_html=True,
            )


def render_declined_notice() -> None:
    st.markdown(
        """
<div class="declined-card">
  <b>⚠️ Insufficient Evidence</b><br>
  The indexed clinical documents do not contain enough information to
  answer this question reliably.<br><br>
  For accurate medical information, please consult a qualified healthcare
  professional or refer directly to primary sources:
  <a href="https://www.who.int" style="color:#2980B9;">WHO</a> ·
  <a href="https://www.cdc.gov" style="color:#2980B9;">CDC</a> ·
  <a href="https://www.nih.gov" style="color:#2980B9;">NIH</a>
</div>
""",
        unsafe_allow_html=True,
    )


def render_timing(ms: float) -> None:
    st.markdown(
        f'<span class="timing-badge">⏱ {ms:.0f} ms</span>',
        unsafe_allow_html=True,
    )


def render_message(msg: dict) -> None:
    """Render a single message (user or assistant) from session state."""
    role = msg["role"]
    avatar = "👤" if role == "user" else "🏥"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        if role == "assistant":
            if msg.get("citations"):
                render_citations(msg["citations"])
            if msg.get("declined"):
                render_declined_notice()
            if msg.get("time_ms"):
                render_timing(msg["time_ms"])


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 Healthcare RAG")
    st.markdown("*Ask My Docs — Clinical Edition*")
    st.divider()

    # ── System Status ────────────────────────────────────
    st.markdown("### System Status")
    health = fetch_health()

    if health and health.get("pipeline_ready"):
        st.markdown(
            '<span class="badge-online">● Online</span>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-label">Documents</p>'
                f'<p class="metric-value">{health["documents_indexed"]}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-card">'
                f'<p class="metric-label">Queries</p>'
                f'<p class="metric-value">{st.session_state.total_queries}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )
    elif health:
        st.markdown(
            '<span class="badge-offline">● Degraded</span>',
            unsafe_allow_html=True,
        )
        st.caption("Pipeline not ready. Check server logs.")
    else:
        st.markdown(
            '<span class="badge-offline">● API Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption("Start the backend:")
        st.code("uvicorn src.api:app --port 8000", language="bash")

    st.divider()

    # ── Model Info ───────────────────────────────────────
    st.markdown("### Model Info")
    if health:
        model_short = health.get("model", "N/A").replace("claude-", "")
        emb_short = health.get("embedding_model", "N/A")
        st.markdown(
            f'<div class="metric-card">'
            f'<p class="metric-label">LLM</p>'
            f'<p class="metric-value" style="font-size:0.82rem;">{model_short}</p>'
            f"</div>"
            f'<div class="metric-card">'
            f'<p class="metric-label">Embeddings</p>'
            f'<p class="metric-value" style="font-size:0.75rem;">{emb_short}</p>'
            f"</div>"
            f'<div class="metric-card">'
            f'<p class="metric-label">Retrieval</p>'
            f'<p class="metric-value" style="font-size:0.75rem;">BM25 + Vector + CE</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Connect to API to see model info")

    st.divider()

    # ── Actions ──────────────────────────────────────────
    st.markdown("### Actions")
    if st.button("🗑️  Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        fetch_health.clear()
        st.rerun()

    if st.button("🔄  Refresh Status", use_container_width=True):
        fetch_health.clear()
        st.rerun()

    st.divider()

    # ── About ────────────────────────────────────────────
    st.markdown("### About")
    st.caption(
        "Answers are grounded **exclusively** in indexed clinical documents. "
        "The system will decline questions when evidence is insufficient.\n\n"
        "Sources: WHO, CDC, NIH guidelines.\n\n"
        "**Not a substitute for professional medical advice.**"
    )
    if health:
        st.caption(f"API v{health.get('version', '?')}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main — Hero + Chat
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="hero-banner">
  <h1>🏥 Healthcare RAG Assistant</h1>
  <p>Evidence-based answers from WHO · CDC · NIH clinical guidelines — with paragraph-level citations</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Render conversation history ───────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🏥"):
        st.markdown(WELCOME_MESSAGE)
else:
    for msg in st.session_state.messages:
        render_message(msg)

# ── Chat input (pinned to bottom by Streamlit) ────────────────────────────────
if prompt := st.chat_input("Ask a clinical question...", key="chat_input"):
    # ── Render user turn ──────────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # ── Call API & render assistant turn ─────────────────────────────────────
    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Searching clinical documents…"):
            try:
                data = call_query_api(prompt)

                answer = data["answer"]
                citations = data.get("citations", [])
                declined = data.get("declined", False)
                time_ms = data.get("processing_time_ms", 0)

                # Render answer
                st.markdown(answer)

                # Render citations
                if citations:
                    render_citations(citations)

                # Render decline notice
                if declined:
                    render_declined_notice()

                # Render timing
                render_timing(time_ms)

                # Store in session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "declined": declined,
                        "time_ms": time_ms,
                    }
                )
                st.session_state.total_queries += 1
                fetch_health.clear()  # refresh doc count

            except requests.ConnectionError:
                err = (
                    "**Cannot reach the API server.**\n\n"
                    "Start it with:\n```bash\nuvicorn src.api:app --port 8000\n```"
                )
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "citations": [], "declined": False}
                )

            except requests.Timeout:
                err = "**Request timed out.** The model may be processing a complex query. Please try again."
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "citations": [], "declined": False}
                )

            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response else "?"
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                err = f"**API Error {code}:** {detail}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "citations": [], "declined": False}
                )

            except Exception as exc:
                err = f"**Unexpected error:** {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "citations": [], "declined": False}
                )
