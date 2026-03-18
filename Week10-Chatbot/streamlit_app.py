"""
streamlit_app.py
================
Week 10 Assignment – Deploying and Testing the Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-18

Purpose:
    Streamlit deployment of the Hypotify Clinical Chatbot.

    Streamlit is chosen as an alternative (or primary) deployment target
    because:
        – Zero HTML/CSS/JS required: the entire UI is described in Python
        – Built-in state management (st.session_state) replaces the Flask
          session cookie
        – One command to run:  streamlit run streamlit_app.py
        – Easy cloud hosting via Streamlit Community Cloud (free tier)

    This module reuses the HypotifyChatbot engine and db_manager which are
    now co-located in this directory (consolidated from Weeks 8 and 9).

Usage:
    streamlit run streamlit_app.py

    The app opens automatically in the browser at http://localhost:8501
"""

import os
import sys
import time

import streamlit as st

# All modules (chatbot_w9, db_manager, preprocess, etc.) are co-located in
# this directory — no sys.path manipulation needed.

# ── Streamlit page config (MUST be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Hypotify Clinical Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: dark clinical aesthetic ───────────────────────────────────────
st.markdown("""
<style>
/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Dark background for whole app ── */
.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Chat message bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
    color: #fff;
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 4px 0 4px 15%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(59,130,246,0.25);
}

.bot-bubble {
    background: rgba(22, 27, 34, 0.95);
    border: 1px solid rgba(56,68,85,0.6);
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px;
    margin: 4px 15% 4px 0;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12.5px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    color: #e6edf3;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.msg-meta {
    font-size: 10px;
    color: #7d8590;
    margin: 2px 4px 8px 4px;
}

.meta-user { text-align: right; }
.meta-bot  { text-align: left;  }

/* ── Welcome card ── */
.welcome-card {
    background: rgba(22,27,34,0.85);
    border: 1px solid rgba(56,68,85,0.6);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 20px;
}

/* ── Status pill ── */
.status-ok   { color: #22c55e; font-weight: 600; }
.status-warn { color: #f59e0b; font-weight: 600; }
.status-err  { color: #ef4444; font-weight: 600; }

/* ── Sidebar stat rows ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 0;
    border-bottom: 1px solid rgba(56,68,85,0.3);
}
.stat-label { color: #7d8590; font-family: monospace; }
.stat-value { color: #93c5fd; font-weight: 600; font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# ── Load the chatbot (cached so it only loads once across reruns) ──────────────

@st.cache_resource(show_spinner="Loading Hypotify Clinical Chatbot…")
def load_chatbot():
    """
    Import and initialise HypotifyChatbot.

    Wrapped in st.cache_resource so Streamlit only loads the Keras model
    and connects to the database once per server session, not on every
    page interaction.

    Returns:
        HypotifyChatbot instance, or None if initialisation fails.
    """
    try:
        from chatbot_w9 import HypotifyChatbot
        bot = HypotifyChatbot()
        return bot
    except ImportError as exc:
        st.error(f"Could not import chatbot engine: {exc}")
        return None
    except SystemExit:
        st.error(
            "Chatbot failed to initialise -- model artefacts are missing. "
            "Run `python train.py` in Week10-Chatbot first."
        )
        return None
    except Exception as exc:
        st.error(f"Unexpected error loading chatbot: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def load_db_manager():
    """
    Import db_manager (cached, like the chatbot).

    Returns:
        db_manager module or None.
    """
    try:
        import db_manager as dbm
        dbm.connect().close()   # ping test
        return dbm
    except Exception:
        return None


# ── Session state init ─────────────────────────────────────────────────────────

def _init_session():
    """
    Initialise Streamlit session state keys on first run.

    st.session_state persists across reruns within the same browser tab,
    replacing the Flask session cookie mechanism.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []          # list of {"role": "user"|"bot", "text": str, "time": str}
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True


# ── Helper: timestamp string ───────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%H:%M")


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(chatbot, dbm):
    """Build the left sidebar with status, DB stats, and quick commands."""

    st.sidebar.markdown("## 🏥 Hypotify")
    st.sidebar.markdown("*Clinical Chatbot — Week 10*")
    st.sidebar.divider()

    # ── Status ────────────────────────────────────────────────────────────────
    model_ok = chatbot is not None
    db_ok    = dbm is not None

    if model_ok and db_ok:
        st.sidebar.markdown('<span class="status-ok">● Online</span>', unsafe_allow_html=True)
    elif model_ok:
        st.sidebar.markdown('<span class="status-warn">● DB unavailable</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-err">● Chatbot offline</span>', unsafe_allow_html=True)

    st.sidebar.divider()

    # ── Database stats ─────────────────────────────────────────────────────────
    st.sidebar.markdown("**Database Stats**")

    TABLE_LABELS = {
        "intents":          "Intents",
        "patients_summary": "Pt Summary",
        "conversation_log": "Conv. Log",
        "user_feedback":    "Feedback",
        "patients":         "Patients",
        "admissions":       "Admissions",
        "diagnoses":        "Diagnoses",
        "labs":             "Labs (sample)",
    }

    if dbm is not None:
        try:
            stats = dbm.get_db_stats()
            rows_html = "".join(
                f'<div class="stat-row">'
                f'<span class="stat-label">{TABLE_LABELS.get(k, k)}</span>'
                f'<span class="stat-value">{v:,}</span>'
                f'</div>'
                for k, v in stats.items()
            )
            st.sidebar.markdown(rows_html, unsafe_allow_html=True)
        except Exception:
            st.sidebar.caption("Could not load DB stats.")
    else:
        st.sidebar.caption("Database not connected.")

    st.sidebar.divider()

    # ── Quick commands ─────────────────────────────────────────────────────────
    st.sidebar.markdown("**Quick Commands**")

    quick = [
        ("❓ help",               "help"),
        ("🗄️ db info",            "db info"),
        ("♂♀ insights gender",   "insights gender"),
        ("📅 insights age",       "insights age"),
        ("🌍 insights race",      "insights race"),
        ("🗣️ insights language",  "insights language"),
        ("💰 insights poverty",   "insights poverty"),
        ("❤️ insights marital",   "insights marital"),
        ("😊 sentiment example",  "sentiment I feel great about this tool!"),
        ("🏷️ ner example",        "ner Dr. Patel treated on 2025-03-10"),
        ("🔍 search glucose",     "search glucose"),
    ]

    for label, cmd in quick:
        if st.sidebar.button(label, key=f"quick_{cmd}", use_container_width=True):
            _send_message(cmd, chatbot)
            st.rerun()

    st.sidebar.divider()
    
    # ── Top 100 Q&A Viewer ─────────────────────────────────────────────────────
    if st.sidebar.button("💯 View Top 100 Q&A", use_container_width=True):
        st.session_state.show_top100 = not st.session_state.get("show_top100", False)
        st.rerun()
        
    st.sidebar.divider()
    st.sidebar.caption(
        "Author: Shruti Malik · ITEC5025\n"
        "Week 10 · March 2026"
    )


# ── Message rendering ──────────────────────────────────────────────────────────

def render_messages():
    """Render the full conversation history as styled HTML bubbles."""

    # Welcome card (shown until user sends first message)
    if st.session_state.show_welcome:
        st.markdown("""
        <div class="welcome-card">
        <h3 style="color:#93c5fd; margin-bottom:8px">Welcome to Hypotify 🏥</h3>
        <p style="color:#7d8590; font-size:13px; line-height:1.7">
        I'm your AI clinical assistant.<br>
        I can look up patient records, analyze population statistics,
        perform sentiment analysis, extract named entities, and more.<br><br>
        Try typing <code>help</code> to see all available commands,
        or use the quick-command buttons in the sidebar.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # Message bubbles
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["text"]}</div>'
                f'<div class="msg-meta meta-user">You · {msg["time"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Escape any HTML in bot responses before inserting into HTML
            safe_text = (msg["text"]
                         .replace("&", "&amp;")
                         .replace("<", "&lt;")
                         .replace(">", "&gt;"))
            st.markdown(
                f'<div class="bot-bubble">{safe_text}</div>'
                f'<div class="msg-meta meta-bot">🏥 Hypotify · {msg["time"]}</div>',
                unsafe_allow_html=True,
            )


# ── Core send logic ────────────────────────────────────────────────────────────

def _send_message(user_text: str, chatbot) -> None:
    """
    Process a user message and append both the user turn and bot response
    to st.session_state.messages.

    Args:
        user_text (str): The raw text the user typed or a quick command.
        chatbot:         HypotifyChatbot instance (or None).
    """
    user_text = user_text.strip()
    if not user_text:
        return

    # Hide welcome card as soon as any message is sent
    st.session_state.show_welcome = False

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "text": user_text,
        "time": _now(),
    })

    # Get bot response
    if chatbot is None:
        bot_text = (
            "Chatbot engine is not available.\n"
            "Make sure model artefacts exist and db_setup.py has been run."
        )
    else:
        try:
            bot_text = chatbot.respond(user_text)
        except Exception as exc:
            bot_text = f"⚠️ An error occurred: {exc}"

    st.session_state.messages.append({
        "role": "bot",
        "text": bot_text,
        "time": _now(),
    })


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    """Entry point — builds the full Streamlit UI."""

    _init_session()

    # Load shared resources
    chatbot = load_chatbot()
    dbm     = load_db_manager()

    # Sidebar
    render_sidebar(chatbot, dbm)

    # Main column header
    st.markdown(
        '<h2 style="color:#e6edf3; margin-bottom:0">Hypotify Clinical Chatbot</h2>'
        '<p style="color:#7d8590; font-size:13px; margin-top:4px">'
        'ITEC5025 · Week 10 Deployment · Author: Shruti Malik</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # View Top 100 Tracker
    if st.session_state.get("show_top100", False):
        st.markdown("### 📚 Top 100 Probable Questions and Answers")
        try:
            qa_path = os.path.join(os.path.dirname(__file__), "top100_qa.txt")
            with open(qa_path, "r", encoding="utf-8") as f:
                content = f.read()
            with st.expander("Click to read the full document", expanded=True):
                st.code(content, language="text")
        except Exception as e:
            st.error(f"Could not load top100_qa.txt: {e}")
        st.divider()

    # Render conversation history
    render_messages()

    # ── Input area ─────────────────────────────────────────────────────────────
    # Streamlit re-runs the whole script on every interaction, so we use a
    # form with clear_on_submit=True to clear the text box after sending.
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])

        with col1:
            user_input = st.text_input(
                label="Message",
                label_visibility="collapsed",
                placeholder=(
                    "Type a message or command… "
                    "e.g. 'insights gender', 'patient <uuid>', 'help'"
                ),
                key="input_field",
            )

        with col2:
            submitted = st.form_submit_button("➤ Send", use_container_width=True)

    if submitted and user_input.strip():
        _send_message(user_input.strip(), chatbot)
        st.rerun()   # refresh to display the new messages

    # ── Context-aware hint bar ─────────────────────────────────────────────────
    if st.session_state.messages:
        last_intent = ""
        if chatbot and chatbot.context_history:
            last_intent = chatbot.context_history[-1]

        hints = {
            "patient_lookup":    "Try: `admissions <id>` · `diagnoses <id>` · `labs <id>`",
            "insights_gender":   "Try: `insights age` · `insights race` · `insights poverty`",
            "insights_age":      "Try: `insights language` · `insights marital`",
            "sentiment":         "Try: `ner <text>` for named entity recognition",
            "ner":               "Try: `sentiment <text>` for sentiment analysis",
        }
        hint = hints.get(last_intent, "")
        if hint:
            st.caption(f"💡 {hint}")
    else:
        st.caption(
            "💡 Try: `help` · `insights gender` · `db info` · `search glucose` · `feedback 5 great!`"
        )

    # ── Clear conversation button ──────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            st.rerun()


if __name__ == "__main__":
    main()
