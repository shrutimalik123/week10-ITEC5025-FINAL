"""
chatbot_w9.py
=============
Week 9 Assignment – Database Integration for Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-09

Purpose:
    Week 9 extension of the Hypotify Clinical Chatbot.

    New features over Week 8:
        1. SQLite database integration (hypotify.db)
           – Intent responses are fetched from the DB via db_manager.py
           – Every conversation turn is logged to the conversation_log table
        2. User-feedback command
           – "feedback <rating> <comment>" stores a 1-5 star rating in the DB
        3. DB-info command
           – "db info" (or --db-info flag) prints table row counts
        4. Search command
           – "search <keyword>" queries the intents table for matching patterns
        5. JSON-driven fallback
           – Loads intents.json (exported from DB) as a secondary fallback
             so the chatbot still works even if the DB query returns nothing

    All Week 8 advanced NLP features are preserved:
        – BiLSTM intent classification, context tracking, VADER sentiment,
          spaCy NER, regex pattern overrides, graceful error handling.

Usage:
    python chatbot_w9.py              # interactive mode
    python chatbot_w9.py --test       # smoke tests
    python chatbot_w9.py --db-info    # print DB statistics and exit
"""

import json
import logging
import os
import pickle
import random
import re
import sys
import uuid

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

# ── Third-party imports ───────────────────────────────────────────────────────
try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    for _pkg, _path in [
        ("punkt",         "tokenizers/punkt"),
        ("punkt_tab",     "tokenizers/punkt_tab"),
        ("wordnet",       "corpora/wordnet"),
        ("vader_lexicon", "sentiment/vader_lexicon"),
    ]:
        try:
            nltk.data.find(_path)
        except LookupError:
            nltk.download(_pkg, quiet=True)

    _lemmatizer = WordNetLemmatizer()
    _vader      = SentimentIntensityAnalyzer()
    NLTK_OK     = True
except ImportError:
    NLTK_OK = False
    print("WARNING: nltk not installed – sentiment disabled. Run: pip install nltk")

# spaCy NER (optional)
try:
    import spacy
    try:
        _nlp_spacy = spacy.load("en_core_web_sm")
        SPACY_OK   = True
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        try:
            _nlp_spacy = spacy.load("en_core_web_sm")
            SPACY_OK   = True
        except OSError:
            SPACY_OK = False
except ImportError:
    SPACY_OK = False

# ── Local DB module ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

try:
    import db_manager
    DB_OK = True
except ImportError:
    DB_OK = False
    print("WARNING: db_manager.py not found – running without DB integration.")

# ── Paths ──────────────────────────────────────────────────────────────────────
# All model artefacts are co-located in this directory (consolidated from Week 8).
# If they are missing, run:  python train.py  in this directory.
ROOT_DIR  = os.path.dirname(SCRIPT_DIR)          # still used for data CSVs

MODEL_PATH     = os.path.join(SCRIPT_DIR, "chatbot_model_w8.keras")
TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "tokenizer_w8.pkl")
ENCODER_PATH   = os.path.join(SCRIPT_DIR, "label_encoder_w8.pkl")
PATIENTS_CSV   = os.path.join(ROOT_DIR,  "data", "patients_cleaned.csv")
ADMISSIONS_CSV = os.path.join(ROOT_DIR,  "data", "admissions_cleaned.csv")
DIAGNOSES_CSV  = os.path.join(ROOT_DIR,  "data", "diagnoses_cleaned.csv")

INTENTS_JSON   = os.path.join(SCRIPT_DIR, "intents.json")   # exported from DB

MAX_LEN              = 20
CONFIDENCE_THRESHOLD = 0.45
CONTEXT_WINDOW       = 3

# ── JSON fallback responses (loaded from intents.json if available) ────────────
def _load_json_responses() -> dict:
    """
    Load intents.json (exported from the DB) and return a dict mapping
    tag → list of responses. Used as a secondary fallback if the DB returns
    nothing for a tag.
    """
    if not os.path.exists(INTENTS_JSON):
        return {}
    try:
        with open(INTENTS_JSON, encoding="utf-8") as f:
            data = json.load(f)
        result = {}
        for intent in data.get("intents", []):
            tag = intent.get("tag")
            responses = intent.get("responses", [])
            if tag and responses:
                result[tag] = responses
        return result
    except Exception as exc:
        log.warning(f"Could not load intents.json: {exc}")
        return {}

# Hardcoded ultimate-fallback responses (used only if DB and JSON both fail)
HARDCODED_FALLBACK = {
    "greeting": [
        "Hello! Welcome to the Hypotify Clinical Chatbot.",
        "Hi there! How can I assist you today?",
    ],
    "goodbye": [
        "Goodbye! Stay healthy!",
        "Take care! Come back anytime.",
    ],
    "help": [
        ("Available commands:\n"
         "  patient <id>           – lookup a specific patient\n"
         "  insights language/age/gender/race/poverty/marital\n"
         "  sentiment <text>       – analyse sentiment\n"
         "  ner <text>             – named entity recognition\n"
         "  search <keyword>       – search chatbot knowledge base\n"
         "  feedback <1-5> <text>  – submit feedback\n"
         "  db info                – show database statistics\n"
         "  Type 'exit' to quit."),
    ],
    "unknown": [
        "I'm not sure what you mean. Try rephrasing, or type 'help'.",
        "Could you rephrase that? Type 'help' to see what I can do.",
    ],
}

# Context elaboration prompts (unchanged from Week 8)
CONTEXT_ELABORATIONS = {
    "patient_lookup":    "Would you like to also see their lab results or diagnoses? Provide the patient ID.",
    "list_patients":     "I can also filter patients by gender, race, language, or age.",
    "insights_language": "I can also show age, gender, poverty, or race statistics.",
    "insights_age":      "Would you also like gender or language distribution data?",
    "insights_gender":   "I can also show race distribution or poverty statistics if needed.",
    "insights_poverty":  "Would you like to see this broken down by race, gender, or language?",
    "insights_race":     "I can also show language distribution or poverty statistics alongside race data.",
    "sentiment":         "I can also run Named Entity Recognition. Use: ner <text>",
    "ner":               "I can also analyse the sentiment. Use: sentiment <text>",
    "lab_results":       "Would you like a specific lab panel (CBC, metabolic) or all results?",
    "diagnosis_lookup":  "Would you also like to see the patient's admission history or lab results?",
}


# ── NLP helpers ───────────────────────────────────────────────────────────────

def preprocess_input(text: str) -> str:
    """
    Lowercase → tokenise → lemmatise → rejoin.
    Mirrors the pipeline used in preprocess.py so inference matches training.
    """
    if not NLTK_OK:
        return text.lower().strip()
    try:
        tokens = word_tokenize(text.lower())
        lemmas = [_lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]
        return " ".join(lemmas)
    except Exception:
        return text.lower().strip()


def analyse_sentiment(text: str) -> dict:
    """
    Use VADER to compute sentiment scores.
    Returns dict with keys: compound, label ('positive'/'negative'/'neutral').
    """
    if not NLTK_OK:
        return {"compound": 0.0, "label": "neutral"}
    try:
        scores   = _vader.polarity_scores(text)
        compound = scores["compound"]
        label    = ("positive" if compound >= 0.05
                    else "negative" if compound <= -0.05
                    else "neutral")
        return {"compound": round(compound, 3), "label": label, "scores": scores}
    except Exception as exc:
        log.warning(f"Sentiment failed: {exc}")
        return {"compound": 0.0, "label": "neutral"}


def extract_entities_spacy(text: str) -> list:
    """
    Extract named entities using spaCy (en_core_web_sm) with regex fallback.
    """
    if not SPACY_OK:
        return _extract_entities_fallback(text)
    try:
        doc = _nlp_spacy(text)
        return [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "MONEY",
                              "NORP", "FAC", "LOC", "PRODUCT"}
        ]
    except Exception as exc:
        log.warning(f"spaCy NER failed: {exc}")
        return _extract_entities_fallback(text)


def _extract_entities_fallback(text: str) -> list:
    """Rule-based NER fallback using regex patterns."""
    entities = []
    for m in re.findall(r"\bP\d{6}\b", text, re.IGNORECASE):
        entities.append({"text": m.upper(), "label": "PATIENT_ID"})
    date_pat = (r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
                r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b")
    for m in re.finditer(date_pat, text, re.IGNORECASE):
        entities.append({"text": m.group(), "label": "DATE"})
    for pn in re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text):
        if pn not in {e["text"] for e in entities}:
            entities.append({"text": pn, "label": "PROPER_NOUN"})
    return entities


# ── Patient data helpers (CSV-based, same as Week 8) ─────────────────────────

def _load_csv_safe(path: str, nrows=None):
    """Load a CSV with safe error handling."""
    try:
        import pandas as pd
        if nrows:
            return pd.read_csv(path, nrows=nrows, encoding="utf-8", on_bad_lines="skip")
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:
        log.error(f"Failed to load {path}: {exc}")
        return None


def get_patient_by_id(patient_id: str) -> str:
    """Look up a patient record by ID from the patients CSV (CSV-based fallback)."""
    df = _load_csv_safe(PATIENTS_CSV)
    if df is None:
        return "Patient data is currently unavailable."
    patient_id = patient_id.upper()
    match = df[df["patient_id"].str.upper() == patient_id]
    if match.empty:
        return f"No patient found with ID: {patient_id}"
    row = match.iloc[0]
    lines = [f"\n  Patient Record: {patient_id}", "  " + "-" * 40]
    for col, label in [("gender", "Gender"), ("date_of_birth", "Date of Birth"),
                        ("race", "Race"), ("marital_status", "Marital Status"),
                        ("language", "Language"), ("poverty_pct", "Poverty %")]:
        if col in row.index:
            lines.append(f"  {label:<20}: {row[col]}")
    return "\n".join(lines)


def get_fallback_patient(patient_id: str, data_type: str) -> str:
    """
    CSV-based fallback for admissions, diagnoses, and labs queries when the
    clinical DB tables are not yet populated (import_patients.py not run yet).

    Args:
        patient_id (str): Patient identifier.
        data_type  (str): One of 'admissions', 'diagnoses', 'labs'.
    Returns:
        str: Formatted results or a prompt to run import_patients.py.
    """
    try:
        import pandas as pd
        if data_type == "admissions":
            df = _load_csv_safe(ADMISSIONS_CSV)
            if df is None:
                return (f"Admissions data unavailable. Run  python import_patients.py  "
                        f"to load all data into the database.")
            uid = patient_id.upper()
            rows = df[df["patient_id"].str.upper() == uid]
            if rows.empty:
                return f"No admissions found for patient {patient_id}."
            lines = [f"\n  Admissions for {patient_id} ({len(rows)} record(s)):"]
            lines.append("  " + "-" * 52)
            for _, r in rows.iterrows():
                lines.append(
                    f"  {r.get('admission_id','?')[:16]}…  "
                    f"{r.get('admission_start','?')}  →  {r.get('admission_end','?')}"
                )
            return "\n".join(lines)

        elif data_type == "diagnoses":
            df = _load_csv_safe(DIAGNOSES_CSV)
            if df is None:
                return (f"Diagnoses data unavailable. Run  python import_patients.py  "
                        f"to load all data into the database.")
            uid  = patient_id.upper()
            rows = df[df["patient_id"].str.upper() == uid].head(20)
            if rows.empty:
                return f"No diagnoses found for patient {patient_id}."
            lines = [f"\n  Diagnoses for {patient_id} (up to 20):"]
            lines.append("  " + "-" * 52)
            for _, r in rows.iterrows():
                desc = str(r.get("diagnosis_description", "Unknown"))[:55]
                code = str(r.get("diagnosis_code", "?"))
                lines.append(f"  {code}: {desc}")
            return "\n".join(lines)

        else:  # labs — CSV is too large to scan; always redirect to DB
            return (
                f"Lab results from CSV require 'python import_patients.py' to be run first.\n"
                f"Once the database is populated, use:  labs {patient_id}"
            )
    except Exception as exc:
        log.warning(f"get_fallback_patient failed: {exc}")
        return f"Unable to retrieve {data_type} for {patient_id}. Run import_patients.py first."



def get_patient_summary_stats(category: str) -> str:
    """Generate population-level statistics for a category from the patients CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(PATIENTS_CSV, encoding="utf-8", on_bad_lines="skip")
    except Exception as exc:
        return f"Unable to load patient data: {exc}"

    category = category.lower()
    if category == "gender" and "gender" in df.columns:
        counts = df["gender"].value_counts()
        lines  = ["\n  Gender Distribution:"]
        for gen, cnt in counts.items():
            lines.append(f"    {gen:<15}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
        return "\n".join(lines)
    elif category == "language" and "language" in df.columns:
        counts = df["language"].value_counts().head(10)
        lines  = ["\n  Language Distribution (Top 10):"]
        for lang, cnt in counts.items():
            lines.append(f"    {lang:<20}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
        return "\n".join(lines)
    elif category == "race" and "race" in df.columns:
        counts = df["race"].value_counts().head(10)
        lines  = ["\n  Race Distribution (Top 10):"]
        for race, cnt in counts.items():
            lines.append(f"    {race:<25}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
        return "\n".join(lines)
    elif category == "age" and "date_of_birth" in df.columns:
        import pandas as pd
        df["dob"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
        df["age"] = ((pd.Timestamp("today") - df["dob"]).dt.days / 365.25).round(1)
        a = df["age"].dropna()
        return (f"\n  Age Statistics:\n"
                f"    Mean age   : {a.mean():.1f} years\n"
                f"    Median age : {a.median():.1f} years\n"
                f"    Youngest   : {a.min():.1f} years\n"
                f"    Oldest     : {a.max():.1f} years")
    elif category == "poverty" and "poverty_pct" in df.columns:
        pov = df["poverty_pct"].dropna()
        return (f"\n  Poverty Statistics:\n"
                f"    Mean poverty %   : {pov.mean():.1f}%\n"
                f"    Median poverty % : {pov.median():.1f}%\n"
                f"    Patients w/ data : {(df['poverty_pct'] > 0).sum():,}")
    elif category == "marital" and "marital_status" in df.columns:
        counts = df["marital_status"].value_counts()
        lines  = ["\n  Marital Status Distribution:"]
        for status, cnt in counts.items():
            lines.append(f"    {status:<20}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")
        return "\n".join(lines)
    return f"No data for category '{category}'. Try: gender, language, race, age, poverty, marital."


# ── Model inference ───────────────────────────────────────────────────────────

def load_artifacts() -> tuple:
    """
    Load the trained Keras model, tokenizer, and label encoder.
    Also verifies the SQLite database is present.

    Returns:
        tuple: (model, tokenizer, label_encoder)
    """
    # Check model artifacts
    for path, name in [
        (MODEL_PATH,     "chatbot_model_w8.keras"),
        (TOKENIZER_PATH, "tokenizer_w8.pkl"),
        (ENCODER_PATH,   "label_encoder_w8.pkl"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact: {name}\n"
                f"Expected at: {path}\n"
                "Run  python train.py  in Week10-Chatbot first."
            )

    # Check / warn about database
    db_path = os.path.join(SCRIPT_DIR, "hypotify.db")
    if not os.path.exists(db_path):
        print("  WARNING: hypotify.db not found. Run  python db_setup.py  first.")
        print("           Chatbot will use JSON fallback responses only.\n")

    model = tf_load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    return model, tokenizer, le


def predict_intent(text: str, model, tokenizer, le) -> tuple:
    """
    Run BiLSTM inference to predict the intent of a user message.

    Returns:
        tuple: (intent_tag: str, confidence: float)
    """
    cleaned     = preprocess_input(text)
    sequences   = tokenizer.texts_to_sequences([cleaned])
    padded      = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
    predictions = model.predict(padded, verbose=0)[0]
    class_idx   = int(np.argmax(predictions))
    confidence  = float(predictions[class_idx])
    intent_tag  = le.inverse_transform([class_idx])[0]
    return intent_tag, confidence


# ── Regex command patterns ────────────────────────────────────────────────────
_PATIENT_ID_RE  = re.compile(r"\bP\d{6}\b", re.IGNORECASE)
_SENTIMENT_RE   = re.compile(r"^sentiment\s+(.+)$", re.IGNORECASE)
_NER_RE         = re.compile(r"^ner\s+(.+)$", re.IGNORECASE)
_INSIGHTS_RE    = re.compile(r"\b(language|age|gender|race|poverty|marital)\b", re.IGNORECASE)
_FEEDBACK_RE    = re.compile(r"^feedback\s+([1-5])\s*(.*)?$", re.IGNORECASE)
_SEARCH_RE      = re.compile(r"^search\s+(.+)$", re.IGNORECASE)
_DBINFO_RE      = re.compile(r"^db\s+info$", re.IGNORECASE)
# New: sub-commands for clinical DB tables
_ADMISSIONS_RE  = re.compile(r"^admissions?\s+(\S+)", re.IGNORECASE)
_DIAGNOSES_RE   = re.compile(r"^diagnos[ei]s?\s+(\S+)", re.IGNORECASE)
_LABS_RE        = re.compile(r"^labs?\s+(\S+)", re.IGNORECASE)


# ── Context-aware chatbot ─────────────────────────────────────────────────────

class HypotifyChatbot:
    """
    Context-aware clinical chatbot with SQLite database integration.

    Database interactions:
        - Every conversation turn is logged to conversation_log via db_manager.
        - Intent responses are fetched from the intents table first; only if the
          DB returns nothing does the chatbot fall back to intents.json or
          hardcoded responses.
        - User feedback is stored via the 'feedback <rating> <comment>' command.
    """

    def __init__(self):
        """Load model artefacts, initialise context window and session ID."""
        print("  Loading model artefacts …", end=" ", flush=True)
        try:
            self.model, self.tokenizer, self.le = load_artifacts()
            print("done.")
        except FileNotFoundError as exc:
            print(f"\n  ERROR: {exc}")
            sys.exit(1)

        # Load JSON fallback responses (exported from DB)
        self._json_responses = _load_json_responses()
        if self._json_responses:
            print(f"  Loaded intents.json ({len(self._json_responses)} tags as fallback).")

        # DB availability check
        if DB_OK:
            try:
                stats = db_manager.get_db_stats()
                print(f"  Connected to hypotify.db  "
                      f"({stats.get('intents', 0)} intents, "
                      f"{stats.get('conversation_log', 0)} log entries).")
            except Exception as exc:
                log.warning(f"DB connectivity check failed: {exc}")

        print()

        # Unique session ID for this conversation run
        self.session_id: str = str(uuid.uuid4())[:8]

        # Context window (last CONTEXT_WINDOW intent tags)
        self.context_history: list[str] = []

    # ── Context helpers ────────────────────────────────────────────────────────

    def _update_context(self, intent: str) -> None:
        """Append an intent to the rolling context window."""
        self.context_history.append(intent)
        if len(self.context_history) > CONTEXT_WINDOW:
            self.context_history.pop(0)

    def _is_followup(self, intent: str) -> bool:
        """Return True if the current intent follows the same (or a followup) intent."""
        if not self.context_history:
            return False
        last = self.context_history[-1]
        return last == intent or last == "context_followup"

    # ── Response selection ─────────────────────────────────────────────────────

    def _get_response_for_tag(self, tag: str) -> str | None:
        """
        Try three sources in priority order and return the first hit:
            1. SQLite DB  (db_manager.get_intent_response)
            2. intents.json (loaded at startup)
            3. Hardcoded HARDCODED_FALLBACK dict
        Returns None only if all three sources fail.
        """
        # 1 – Database (primary source)
        if DB_OK:
            resp = db_manager.get_intent_response(tag)
            if resp:
                return resp

        # 2 – JSON (exported from DB; acts as a read-only cache)
        if tag in self._json_responses:
            return random.choice(self._json_responses[tag])

        # 3 – Hardcoded fallback
        if tag in HARDCODED_FALLBACK:
            return random.choice(HARDCODED_FALLBACK[tag])

        return None

    def _build_response(self, intent: str, confidence: float,
                        raw_input: str) -> tuple[str, str]:
        """
        Build the chatbot response for a predicted intent.

        Returns:
            tuple (response_text, resolved_intent_tag)
        """
        # ── Low confidence ─────────────────────────────────────────────────────
        if confidence < CONFIDENCE_THRESHOLD:
            msg = ("I'm not quite sure what you mean. Could you rephrase? "
                   "Type 'help' to see available commands.")
            return msg, "unknown"

        # ── feedback <rating> <comment> ────────────────────────────────────────
        m = _FEEDBACK_RE.match(raw_input)
        if m:
            rating  = int(m.group(1))
            comment = (m.group(2) or "").strip()
            if DB_OK:
                success = db_manager.add_user_feedback(self.session_id, rating, comment)
                if success:
                    return (f"  Thank you for your {rating}/5 rating! "
                            f"{'Your comment has been saved.' if comment else ''}"
                            ).strip(), "user_feedback"
            return f"  Feedback received ({rating}/5). Thank you!", "user_feedback"

        # ── db info ────────────────────────────────────────────────────────────
        if _DBINFO_RE.match(raw_input.strip()):
            return _format_db_stats(), "db_info"

        # ── search <keyword> ──────────────────────────────────────────────────
        m = _SEARCH_RE.match(raw_input)
        if m:
            keyword = m.group(1).strip()
            return _format_search_results(keyword), "search"

        # ── sentiment analysis ─────────────────────────────────────────────────
        m = _SENTIMENT_RE.match(raw_input)
        if m or intent == "sentiment":
            text_to_analyse = m.group(1) if m else raw_input
            result = analyse_sentiment(text_to_analyse)
            emoji  = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(
                result["label"], "")
            return (
                f"  Sentiment Analysis:\n"
                f"    Text     : \"{text_to_analyse}\"\n"
                f"    Sentiment: {result['label'].upper()} {emoji}\n"
                f"    Score    : {result['compound']:+.3f}  "
                f"(+1.0 = very positive, -1.0 = very negative)"
            ), "sentiment"

        # ── Named Entity Recognition ───────────────────────────────────────────
        m = _NER_RE.match(raw_input)
        if m or intent == "ner":
            text_for_ner = m.group(1) if m else raw_input.replace("ner ", "", 1)
            entities     = extract_entities_spacy(text_for_ner)
            if entities:
                lines = [f"  Named Entities in: \"{text_for_ner}\""]
                for ent in entities:
                    lines.append(f"    [{ent['label']:>15}]  {ent['text']}")
                return "\n".join(lines), "ner"
            return f"  No named entities detected in: \"{text_for_ner}\"", "ner"

        # ── admissions <id> ────────────────────────────────────────────────────
        m = _ADMISSIONS_RE.match(raw_input)
        if m:
            pid = m.group(1).strip()
            if DB_OK:
                return db_manager.get_patient_admissions(pid), "admission_lookup"
            return get_fallback_patient(pid, "admissions"), "admission_lookup"

        # ── diagnoses <id> ─────────────────────────────────────────────────────
        m = _DIAGNOSES_RE.match(raw_input)
        if m:
            pid = m.group(1).strip()
            if DB_OK:
                return db_manager.get_patient_diagnoses(pid), "diagnosis_lookup"
            return get_fallback_patient(pid, "diagnoses"), "diagnosis_lookup"

        # ── labs <id> ──────────────────────────────────────────────────────────
        m = _LABS_RE.match(raw_input)
        if m:
            pid = m.group(1).strip()
            if DB_OK:
                return db_manager.get_patient_labs(pid), "lab_results"
            return get_fallback_patient(pid, "labs"), "lab_results"

        # ── Patient ID direct lookup ───────────────────────────────────────────
        pid_match = _PATIENT_ID_RE.search(raw_input)
        if pid_match:
            pid = pid_match.group()
            if DB_OK:
                return db_manager.get_patient_from_db(pid), "patient_lookup"
            return get_patient_by_id(pid), "patient_lookup"

        # ── Clinical insights — prefer DB, fall back to CSV ────────────────────
        def _get_insights(category: str) -> str:
            """Try DB-computed stats first, then CSV fallback."""
            if DB_OK:
                db_result = db_manager.get_population_stats_from_db(category)
                if db_result:
                    return db_result
            return get_patient_summary_stats(category)

        if intent.startswith("insights_"):
            category = intent.replace("insights_", "")
            return _get_insights(category), intent

        kw_match = _INSIGHTS_RE.search(raw_input)
        if kw_match and intent not in ("translate", "medical_terms"):
            category = kw_match.group(1)
            return _get_insights(category), f"insights_{category}"

        # ── Context elaboration ────────────────────────────────────────────────
        if self._is_followup(intent) and intent in CONTEXT_ELABORATIONS:
            return CONTEXT_ELABORATIONS[intent], intent

        # ── DB / JSON / hardcoded response ─────────────────────────────────────
        resp = self._get_response_for_tag(intent)
        if resp:
            return resp, intent

        # Ultimate fallback
        fallback = (self._get_response_for_tag("unknown")
                    or "I'm not sure what you mean. Type 'help' for assistance.")
        return fallback, "unknown"

    # ── Public respond method ─────────────────────────────────────────────────

    def respond(self, user_input: str) -> str:
        """
        Generate a response to a user message.

        Pipeline:
            1. Predict intent with BiLSTM model
            2. Detect sentiment for empathy adjustment
            3. Build response (DB-driven or rule-based)
            4. Log the conversation turn to the database
            5. Update context window

        Args:
            user_input (str): Raw message from the user.

        Returns:
            str: Chatbot response.
        """
        user_input = user_input.strip()
        if not user_input:
            return "Please type a message. Type 'help' to see available commands."

        try:
            # Predict intent
            intent, confidence = predict_intent(
                user_input, self.model, self.tokenizer, self.le
            )

            # Sentiment for empathy
            sentiment = analyse_sentiment(user_input)

            # Build response
            response, resolved_intent = self._build_response(intent, confidence, user_input)

            # Empathy prefix for strongly negative messages
            if (sentiment["compound"] < -0.5
                    and resolved_intent not in ("sentiment", "ner", "goodbye",
                                                "user_feedback", "db_info")):
                response = "I'm sorry to hear that. I'm here to help.\n" + response

            # ── Log this turn to the database ──────────────────────────────────
            if DB_OK:
                db_manager.log_conversation(
                    session_id   = self.session_id,
                    user_message = user_input,
                    bot_response = response,
                    intent_tag   = resolved_intent,
                )

            # Update context window
            self._update_context(resolved_intent)

            return response

        except Exception as exc:
            log.error(f"Error generating response: {exc}", exc_info=True)
            return "An error occurred while processing your request. Please try again."


# ── Formatting helpers ────────────────────────────────────────────────────────

def _format_db_stats() -> str:
    """Format DB row counts into a readable string."""
    if not DB_OK:
        return "  Database module (db_manager) is not available."
    stats = db_manager.get_db_stats()
    if not stats:
        return "  Could not retrieve database statistics."
    lines = ["\n  Database Statistics (hypotify.db):"]
    lines.append("  " + "-" * 42)
    for table, cnt in stats.items():
        lines.append(f"    {table:<22}: {cnt:>6,} rows")
    lines.append("  " + "-" * 42)
    return "\n".join(lines)


def _format_search_results(keyword: str) -> str:
    """Search the intents DB and format results."""
    if not DB_OK:
        return "  Database search is unavailable."
    results = db_manager.search_intents(keyword)
    if not results:
        return f"  No results found for '{keyword}' in the knowledge base."
    lines = [f"\n  Search results for '{keyword}':"]
    lines.append("  " + "-" * 50)
    for r in results[:10]:
        lines.append(f"    [{r['tag']:<18}] {r['pattern']}")
    if len(results) > 10:
        lines.append(f"    … and {len(results) - 10} more results.")
    return "\n".join(lines)


# ── Interactive loop ──────────────────────────────────────────────────────────

def run_interactive(chatbot: HypotifyChatbot) -> None:
    """Run the interactive terminal chat session."""
    print("=" * 65)
    print("  Hypotify Clinical Chatbot  –  Week 9 (DB Integration)")
    print(f"  Session ID : {chatbot.session_id}")
    print("  Type 'help' for commands | 'exit' to quit")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
            print("  Bot: Goodbye! Thank you for using Hypotify. Stay healthy!\n")
            break

        response = chatbot.respond(user_input)
        print(f"  Bot: {response}\n")


# ── Smoke test mode ───────────────────────────────────────────────────────────

def run_smoke_tests(chatbot: HypotifyChatbot) -> None:
    """Run a set of predefined queries and print responses for verification."""
    print("\n" + "=" * 65)
    print("  Smoke Test: 10 sample queries (Week 9)")
    print("=" * 65 + "\n")

    test_cases = [
        ("hello",                                  "greeting"),
        ("list all patients",                      "list_patients"),
        ("what is the gender distribution",        "insights_gender"),
        ("show age statistics",                    "insights_age"),
        ("sentiment I am feeling very sad today",  "sentiment"),
        ("ner Shruti Malik visited on 2024-01-15", "ner"),
        ("search glucose",                         "search"),
        ("db info",                                "db_info"),
        ("feedback 5 Very helpful chatbot!",       "user_feedback"),
        ("goodbye",                                "goodbye"),
    ]

    passed = 0
    for user_input, _ in test_cases:
        response = chatbot.respond(user_input)
        short    = response[:100] + ("…" if len(response) > 100 else "")
        print(f"  Query   : {user_input}")
        print(f"  Response: {short}")
        print()
        passed += 1

    print("=" * 65)
    print(f"  Smoke tests complete: {passed}/{len(test_cases)} queries executed.")
    print("=" * 65 + "\n")


# ── DB info flag ──────────────────────────────────────────────────────────────

def print_db_info() -> None:
    """Print database statistics and exit (used with --db-info flag)."""
    print("\n  Hypotify – Week 9 Database Info")
    print("  " + "=" * 42)
    print(_format_db_stats())
    if DB_OK:
        intents = db_manager.get_all_intents()
        print(f"\n  Intent Tags ({len(intents)} total):")
        print("  " + "-" * 42)
        for item in intents:
            print(f"    {item['tag']:<25}: {item['count']:>3} patterns")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        if "--db-info" in sys.argv:
            print_db_info()
            sys.exit(0)

        chatbot = HypotifyChatbot()

        if "--test" in sys.argv:
            run_smoke_tests(chatbot)
        else:
            run_interactive(chatbot)

    except KeyboardInterrupt:
        print("\n  Session ended.\n")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  FATAL ERROR: {exc}")
        log.exception(exc)
        sys.exit(1)
