"""
app.py
======
Week 10 Assignment – Deploying and Testing the Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-18

Purpose:
    Flask-based web deployment of the Hypotify Clinical Chatbot.

    This module is the Week 10 deployment layer. It wraps the Week 9
    chatbot engine (chatbot_w9.py / HypotifyChatbot) in a browser-
    accessible REST API and serves a polished HTML/CSS/JS user interface.

    Endpoints:
        GET  /              – Serve the chat UI (index.html)
        POST /chat          – Accept user message, return bot response (JSON)
        GET  /health        – Database & model health check (JSON)
        GET  /stats         – Live database statistics (JSON)
        GET  /history       – Conversation history for the current session (JSON)

    Deployment:
        python app.py                  # development server on port 5000
        python app.py --port 8080      # custom port

    The Flask development server is suitable for local demo and coursework.
    A production deployment would use Gunicorn or uWSGI behind Nginx,
    but for a single-user academic demonstration this is the appropriate
    and simplest choice.
"""

import argparse
import json
import logging
import os
import sys

# ── Force UTF-8 output on Windows ──────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

# ── Flask import ───────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, render_template, session
    from flask_cors import CORS
except ImportError:
    print(
        "ERROR: Flask or flask-cors not installed.\n"
        "Run:  pip install flask flask-cors"
    )
    sys.exit(1)

# ── Resolve template path ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # .../Week10-Chatbot
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")

# All modules (chatbot_w9, db_manager, preprocess, etc.) are now co-located
# in this directory after consolidation from Week 8 and Week 9.

# ── Import the chatbot engine ─────────────────────────────────────────────────
try:
    from chatbot_w9 import HypotifyChatbot
    import db_manager
    DB_OK = True
    log.info("Imported HypotifyChatbot and db_manager.")
except ImportError as exc:
    log.error(f"Could not import chatbot engine: {exc}")
    HypotifyChatbot = None
    db_manager = None
    DB_OK = False

# ── Flask app setup ────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = os.urandom(24)   # used for Flask session (stores session_id per browser tab)
CORS(app)                         # allow cross-origin requests for local development

# ── Initialise the chatbot (once at startup) ───────────────────────────────────
chatbot: HypotifyChatbot | None = None

def _init_chatbot() -> None:
    """Load the chatbot model and warm up the database connection."""
    global chatbot
    if HypotifyChatbot is None:
        log.error("HypotifyChatbot class unavailable. Check Week9-Chatbot imports.")
        return
    try:
        log.info("Initialising HypotifyChatbot ...")
        chatbot = HypotifyChatbot()
        log.info("HypotifyChatbot ready.")
    except SystemExit:
        log.error(
            "Chatbot failed to initialise. "
            "Run  python train.py  then  python db_setup.py  in this directory."
        )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Serve the main chat interface.

    Stores a session ID in the browser session cookie so conversation
    history can be retrieved per tab.
    """
    # Give each browser session a unique ID mirroring the chatbot's session_id
    if chatbot is not None and "session_id" not in session:
        session["session_id"] = chatbot.session_id
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle a user message and return the chatbot response.

    Request body (JSON):
        { "message": "<user text>" }

    Response (JSON):
        {
          "response":  "<bot text>",
          "intent":    "<resolved intent tag>",
          "session":   "<session id>"
        }

    HTTP 400 is returned if the message field is missing or empty.
    HTTP 503 is returned if the chatbot engine is not ready.
    """
    if chatbot is None:
        return jsonify({
            "error": "Chatbot engine is not available. Check server logs.",
            "suggestion": "Run python train.py then python db_setup.py in Week10-Chatbot."
        }), 503

    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        bot_response = chatbot.respond(user_message)
        # Extract intent from context history (last entry after respond())
        resolved_intent = (
            chatbot.context_history[-1] if chatbot.context_history else "unknown"
        )
        return jsonify({
            "response": bot_response,
            "intent":   resolved_intent,
            "session":  chatbot.session_id,
        })
    except Exception as exc:
        log.error(f"Error in /chat: {exc}", exc_info=True)
        return jsonify({"error": "Internal server error.", "detail": str(exc)}), 500


@app.route("/health")
def health():
    """
    Return a JSON health-check report covering the model and database.

    This endpoint is useful for automated monitoring and for the
    'Status' indicator in the UI header.

    Response keys:
        status      – overall "ok" or "degraded"
        model       – whether the Keras model is loaded
        database    – whether the SQLite DB is reachable
        tables      – row counts for all DB tables (if DB is available)
    """
    model_ok = chatbot is not None
    db_status = {}
    db_reachable = False

    if DB_OK and db_manager is not None:
        try:
            db_status = db_manager.get_db_stats()
            db_reachable = True
        except Exception as exc:
            log.warning(f"/health DB check failed: {exc}")

    overall = "ok" if (model_ok and db_reachable) else "degraded"
    return jsonify({
        "status":   overall,
        "model":    "loaded" if model_ok else "unavailable",
        "database": "connected" if db_reachable else "unavailable",
        "tables":   db_status,
    })


@app.route("/stats")
def stats():
    """
    Return live database table statistics as JSON.

    The UI uses this endpoint to populate the 'Database Stats' panel
    on first load and whenever the user triggers a db info command.
    """
    if not DB_OK or db_manager is None:
        return jsonify({"error": "Database module unavailable."}), 503
    try:
        return jsonify(db_manager.get_db_stats())
    except Exception as exc:
        log.error(f"/stats failed: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/history")
def history():
    """
    Return the last 20 conversation turns for the current session.

    The session ID is read from the chatbot instance (all turns in
    the current Python process belong to the same session).
    """
    if chatbot is None or not DB_OK or db_manager is None:
        return jsonify([])
    try:
        turns = db_manager.get_conversation_history(chatbot.session_id, limit=20)
        return jsonify(turns)
    except Exception as exc:
        log.error(f"/history failed: {exc}")
        return jsonify({"error": str(exc)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    """Parse command-line arguments, initialise the chatbot, and start Flask."""
    parser = argparse.ArgumentParser(description="Hypotify Clinical Chatbot – Web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    args = parser.parse_args()

    # Initialise the chatbot before the server starts accepting connections
    _init_chatbot()

    print(f"\n  ╔═══════════════════════════════════════════╗")
    print(f"  ║   Hypotify Clinical Chatbot – Week 10     ║")
    print(f"  ║   Web UI → http://{args.host}:{args.port}/         ║")
    print(f"  ╚═══════════════════════════════════════════╝\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
