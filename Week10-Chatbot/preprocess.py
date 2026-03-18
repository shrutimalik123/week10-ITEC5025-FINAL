"""
preprocess.py
=============
Week 8 Assignment – Chatbot Model Development and Training
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-02-28

Purpose:
    Perform advanced NLP preprocessing on the chatbot training dataset
    (dataset.csv) and the real patient data CSVs located in ../data/.

    Pipeline steps:
        1.  Load dataset.csv  → raw user_input / intent / response triples
        2.  Text Cleaning     → lowercase, strip HTML/punctuation, collapse whitespace
        3.  Tokenization      → NLTK word_tokenize (advanced, rule-based)
        4.  Stop-word removal → NLTK English stop-word corpus
        5.  Lemmatization     → NLTK WordNetLemmatizer (morphologically accurate)
        6.  TF-IDF analysis   → scikit-learn TfidfVectorizer; top-5 terms per intent
        7.  Word Embeddings   → gensim Word2Vec trained on the cleaned corpus
        8.  Patient data stats→ load + summarise the 4 real patient CSVs
        9.  Save output       → preprocessed_data.json  (structured, indexed)
                              → word2vec.model           (gensim binary)
                              → patient_summary.json     (patient data statistics)

Usage:
    python preprocess.py
"""

import json
import logging
import os
import re
import sys

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Third-party imports ───────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
except ImportError:
    log.error("pandas / numpy not found. Run: pip install pandas numpy")
    sys.exit(1)

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    log.error("nltk not found. Run: pip install nltk")
    sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    log.error("scikit-learn not found. Run: pip install scikit-learn")
    sys.exit(1)

# gensim is optional – Word2Vec falls back gracefully if not installed
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    log.warning("gensim not found. Word2Vec step will be skipped. "
                "Run: pip install gensim")
    GENSIM_AVAILABLE = False

# ── NLTK data download ─────────────────────────────────────────────────────────
def _ensure_nltk():
    """Download required NLTK corpora if not already present."""
    packages = {
        "tokenizers/punkt":          "punkt",
        "tokenizers/punkt_tab":      "punkt_tab",
        "corpora/stopwords":         "stopwords",
        "corpora/wordnet":           "wordnet",
        "corpora/omw-1.4":           "omw-1.4",
    }
    for resource_path, download_name in packages.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log.info(f"Downloading NLTK resource: {download_name}")
            nltk.download(download_name, quiet=True)

_ensure_nltk()

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR        = os.path.dirname(SCRIPT_DIR)          # project root
DATA_DIR        = os.path.join(ROOT_DIR, "data")

DATASET_PATH        = os.path.join(SCRIPT_DIR, "dataset.csv")
OUTPUT_JSON         = os.path.join(SCRIPT_DIR, "preprocessed_data.json")
WORD2VEC_PATH       = os.path.join(SCRIPT_DIR, "word2vec.model")
PATIENT_SUMMARY_OUT = os.path.join(SCRIPT_DIR, "patient_summary.json")

# Patient CSV paths
PATIENTS_CSV    = os.path.join(DATA_DIR, "patients_cleaned.csv")
ADMISSIONS_CSV  = os.path.join(DATA_DIR, "admissions_cleaned.csv")
DIAGNOSES_CSV   = os.path.join(DATA_DIR, "diagnoses_cleaned.csv")
LABS_CSV        = os.path.join(DATA_DIR, "labs_cleaned.csv")

# ── NLP helpers ───────────────────────────────────────────────────────────────
_stop_words  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()

# Compiled regex patterns for efficient repeated use
_HTML_TAG_RE     = re.compile(r"<[^>]+>")          # strip HTML tags
_PUNCT_RE        = re.compile(r"[^a-z0-9\s]")      # keep only alphanumeric + space
_WHITESPACE_RE   = re.compile(r"\s+")              # collapse multiple spaces


def clean_text(text: str) -> str:
    """
    Stage 1 – Text cleaning.

    Steps:
        a) Lowercase the entire string for case normalisation
        b) Strip any HTML/XML tags (e.g. if data came from a web form)
        c) Remove punctuation and special characters (keeps a-z, 0-9, spaces)
        d) Collapse runs of whitespace into a single space
        e) Strip leading/trailing whitespace

    Args:
        text (str): Raw input string.

    Returns:
        str: Cleaned, normalised plain text.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()                         # (a) lowercase
    text = _HTML_TAG_RE.sub(" ", text)          # (b) strip HTML
    text = _PUNCT_RE.sub(" ", text)             # (c) remove punctuation
    text = _WHITESPACE_RE.sub(" ", text)        # (d) collapse whitespace
    return text.strip()                         # (e) trim edges


def tokenize(text: str) -> list:
    """
    Stage 2 – Advanced word tokenisation using NLTK.

    NLTK's word_tokenize applies the Penn Treebank tokeniser, which correctly
    handles contractions (don't → do + n't), punctuation attachment, and
    multi-character tokens – more robust than a simple str.split().

    Args:
        text (str): Cleaned text string.

    Returns:
        list[str]: List of word tokens.
    """
    try:
        return word_tokenize(text)
    except Exception as exc:
        log.warning(f"Tokenisation failed for '{text[:30]}...': {exc}")
        return text.split()  # graceful fallback


def remove_stopwords(tokens: list) -> list:
    """
    Stage 3 – Stop-word removal.

    Filters out common English function words (the, a, is, …) that carry
    little intent-discriminating information.

    Args:
        tokens (list[str]): Tokenised words.

    Returns:
        list[str]: Tokens with stop words removed.
    """
    return [t for t in tokens if t not in _stop_words and len(t) > 1]


def lemmatize(tokens: list) -> list:
    """
    Stage 4 – Lemmatisation with WordNetLemmatizer.

    Reduces each word to its canonical / dictionary form (lemma):
        running → run, patients → patient, analyses → analysis

    This is preferred over Porter stemming (Week 7) because it produces
    linguistically valid roots rather than heuristically truncated forms.

    Args:
        tokens (list[str]): Stop-word-filtered tokens.

    Returns:
        list[str]: Lemmatised tokens.
    """
    return [_lemmatizer.lemmatize(t) for t in tokens]


def full_preprocess(text: str) -> dict:
    """
    Run the complete preprocessing pipeline on a single text string.

    Returns a dict with each intermediate stage preserved for transparency:
        {
            'cleaned'   : str   – after clean_text()
            'tokens'    : list  – after tokenize()
            'filtered'  : list  – after remove_stopwords()
            'lemmas'    : list  – after lemmatize()
            'joined'    : str   – lemmas joined back to a string (model input)
        }
    """
    cleaned  = clean_text(text)
    tokens   = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmas   = lemmatize(filtered)
    return {
        "cleaned":  cleaned,
        "tokens":   tokens,
        "filtered": filtered,
        "lemmas":   lemmas,
        "joined":   " ".join(lemmas),
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the chatbot training CSV into a DataFrame, validating required columns.

    Args:
        path (str): Path to dataset.csv

    Returns:
        pd.DataFrame with columns: user_input, intent, response

    Raises:
        FileNotFoundError: if the CSV is missing
        ValueError:        if required columns are absent
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    required = {"user_input", "intent", "response"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset.csv: {missing}")

    # Drop rows with null user_input or intent
    before = len(df)
    df.dropna(subset=["user_input", "intent"], inplace=True)
    dropped = before - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with null user_input or intent.")

    log.info(f"Loaded {len(df)} training samples from dataset.csv")
    return df


# ── TF-IDF analysis ───────────────────────────────────────────────────────────

def compute_tfidf(corpus: list, top_n: int = 5) -> dict:
    """
    Compute TF-IDF scores across the corpus and return the top-N terms
    globally, as well as the top-N terms for each intent class.

    TF-IDF (Term Frequency–Inverse Document Frequency) measures how important
    a word is to a document relative to the whole corpus. Words that appear
    often in one intent but rarely across others receive high scores, making
    them ideal features for intent classification.

    Args:
        corpus (list[str]): List of preprocessed (joined lemma) strings.
        top_n  (int):       Number of top terms to extract.

    Returns:
        dict: {
            'global_top_terms': list[str],   – most important terms overall
            'feature_names':    list[str],   – full vocabulary list
        }
    """
    if not corpus:
        return {"global_top_terms": [], "feature_names": []}

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError as exc:
        log.warning(f"TF-IDF computation failed: {exc}")
        return {"global_top_terms": [], "feature_names": []}

    feature_names = vectorizer.get_feature_names_out().tolist()

    # Global top-N: sum TF-IDF scores across all documents
    mean_scores   = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices   = mean_scores.argsort()[::-1][:top_n]
    global_top    = [feature_names[i] for i in top_indices]

    log.info(f"TF-IDF global top terms: {global_top}")
    return {
        "global_top_terms": global_top,
        "feature_names":    feature_names[:50],  # keep output compact
    }


# ── Word2Vec embeddings ────────────────────────────────────────────────────────

def train_word2vec(tokenized_sentences: list, save_path: str) -> dict:
    """
    Train a Word2Vec model on the tokenised corpus and save it to disk.

    Word2Vec maps each word to a dense vector in a continuous embedding space
    where semantically similar words are geometrically close. This gives the
    chatbot an understanding of word meaning beyond simple keyword matching.

    Args:
        tokenized_sentences (list[list[str]]): List of lemma token lists.
        save_path (str): Path to save the trained .model file.

    Returns:
        dict: summary statistics about the trained model.
    """
    if not GENSIM_AVAILABLE:
        log.warning("gensim not installed — skipping Word2Vec training.")
        return {"status": "skipped", "reason": "gensim not installed"}

    if not tokenized_sentences:
        return {"status": "skipped", "reason": "empty corpus"}

    log.info("Training Word2Vec embeddings …")
    model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=64,      # dimensionality of each word vector
        window=5,            # context window: 5 words left and right
        min_count=1,         # include all words (small dataset)
        workers=4,           # parallel worker threads
        epochs=50,           # training epochs
        sg=0,                # 0 = CBOW (faster, good for small corpora)
    )
    model.save(save_path)
    vocab_size = len(model.wv)

    # Example: find words most similar to "patient"
    try:
        similar_to_patient = [w for w, _ in model.wv.most_similar("patient", topn=3)]
    except KeyError:
        similar_to_patient = []

    log.info(f"Word2Vec trained. Vocab size: {vocab_size}. Saved to {save_path}")
    return {
        "status":                "trained",
        "vocab_size":            vocab_size,
        "vector_size":           64,
        "similar_to_patient":    similar_to_patient,
        "model_path":            save_path,
    }


# ── Patient data processing ───────────────────────────────────────────────────

def safe_load_csv(path: str, sample_rows: int = None) -> pd.DataFrame | None:
    """
    Safely load a large CSV file with error handling.

    For the labs file (>1 GB) we load only a sample if sample_rows is set,
    which is sufficient for generating summary statistics without exhausting
    system memory during preprocessing.

    Args:
        path       (str): Full path to CSV file.
        sample_rows(int): If set, only read this many rows.

    Returns:
        pd.DataFrame or None on failure.
    """
    if not os.path.exists(path):
        log.warning(f"Patient data file not found: {path}")
        return None
    try:
        kwargs = {"encoding": "utf-8", "on_bad_lines": "skip"}
        if sample_rows:
            kwargs["nrows"] = sample_rows
        df = pd.read_csv(path, **kwargs)
        log.info(f"Loaded {len(df)} rows from {os.path.basename(path)}")
        return df
    except Exception as exc:
        log.error(f"Failed to load {os.path.basename(path)}: {exc}")
        return None


def build_patient_summary() -> dict:
    """
    Generate summary statistics from the four patient data CSV files.

    This demonstrates data cleaning on real-world clinical data:
        - patients_cleaned.csv    : demographics (gender, race, language, poverty)
        - admissions_cleaned.csv  : hospital admission records
        - diagnoses_cleaned.csv   : ICD diagnosis codes and descriptions
        - labs_cleaned.csv        : laboratory test results (sampled for memory)

    Returns:
        dict: Nested summary with stats for each table.
    """
    summary = {}

    # ── Patients ──────────────────────────────────────────────────────────────
    patients_df = safe_load_csv(PATIENTS_CSV)
    if patients_df is not None:
        # Clean: parse dates, compute age from date_of_birth
        try:
            patients_df["date_of_birth"] = pd.to_datetime(
                patients_df["date_of_birth"], errors="coerce"
            )
            today = pd.Timestamp("today")
            patients_df["age"] = (
                (today - patients_df["date_of_birth"]).dt.days / 365.25
            ).round(1)
        except Exception as exc:
            log.warning(f"Age calculation failed: {exc}")

        summary["patients"] = {
            "total_patients":       int(len(patients_df)),
            "gender_distribution":  patients_df["gender"].value_counts().to_dict() if "gender" in patients_df.columns else {},
            "race_distribution":    patients_df["race"].value_counts().head(10).to_dict() if "race" in patients_df.columns else {},
            "language_distribution":patients_df["language"].value_counts().head(10).to_dict() if "language" in patients_df.columns else {},
            "marital_status":       patients_df["marital_status"].value_counts().to_dict() if "marital_status" in patients_df.columns else {},
            "poverty_pct_mean":     round(float(patients_df["poverty_pct"].mean()), 2) if "poverty_pct" in patients_df.columns else None,
            "poverty_pct_median":   round(float(patients_df["poverty_pct"].median()), 2) if "poverty_pct" in patients_df.columns else None,
            "age_mean":             round(float(patients_df["age"].mean()), 1) if "age" in patients_df.columns else None,
            "age_min":              round(float(patients_df["age"].min()), 1) if "age" in patients_df.columns else None,
            "age_max":              round(float(patients_df["age"].max()), 1) if "age" in patients_df.columns else None,
        }
        log.info(f"Patient summary built: {summary['patients']['total_patients']} patients")

    # ── Admissions ────────────────────────────────────────────────────────────
    admissions_df = safe_load_csv(ADMISSIONS_CSV)
    if admissions_df is not None:
        # Compute length of stay (days)
        try:
            admissions_df["admission_start"] = pd.to_datetime(
                admissions_df["admission_start"], errors="coerce"
            )
            admissions_df["admission_end"] = pd.to_datetime(
                admissions_df["admission_end"], errors="coerce"
            )
            admissions_df["los_days"] = (
                (admissions_df["admission_end"] - admissions_df["admission_start"])
                .dt.days
            )
        except Exception as exc:
            log.warning(f"Length-of-stay calculation failed: {exc}")

        summary["admissions"] = {
            "total_admissions":     int(len(admissions_df)),
            "unique_patients":      int(admissions_df["patient_id"].nunique()) if "patient_id" in admissions_df.columns else None,
            "avg_length_of_stay_days": round(float(admissions_df["los_days"].mean()), 2) if "los_days" in admissions_df.columns else None,
            "max_length_of_stay_days": int(admissions_df["los_days"].max()) if "los_days" in admissions_df.columns else None,
        }
        log.info("Admissions summary built.")

    # ── Diagnoses ─────────────────────────────────────────────────────────────
    diagnoses_df = safe_load_csv(DIAGNOSES_CSV)
    if diagnoses_df is not None:
        # Clean diagnosis descriptions (strip extra whitespace)
        if "diagnosis_description" in diagnoses_df.columns:
            diagnoses_df["diagnosis_description"] = (
                diagnoses_df["diagnosis_description"].str.strip()
            )

        summary["diagnoses"] = {
            "total_diagnosis_records":  int(len(diagnoses_df)),
            "unique_diagnosis_codes":   int(diagnoses_df["diagnosis_code"].nunique()) if "diagnosis_code" in diagnoses_df.columns else None,
            "top_10_diagnoses":         diagnoses_df["diagnosis_description"].value_counts().head(10).to_dict() if "diagnosis_description" in diagnoses_df.columns else {},
        }
        log.info("Diagnoses summary built.")

    # ── Labs (sampled – file is >1 GB) ────────────────────────────────────────
    # We sample 100,000 rows for summary statistics to avoid memory exhaustion
    labs_df = safe_load_csv(LABS_CSV, sample_rows=100_000)
    if labs_df is not None:
        summary["labs"] = {
            "sampled_rows":        int(len(labs_df)),
            "unique_lab_tests":    int(labs_df["lab_name"].nunique()) if "lab_name" in labs_df.columns else None,
            "top_10_lab_tests":    labs_df["lab_name"].value_counts().head(10).to_dict() if "lab_name" in labs_df.columns else {},
            "unique_patients":     int(labs_df["patient_id"].nunique()) if "patient_id" in labs_df.columns else None,
        }
        log.info("Labs summary built (sampled 100 k rows).")

    return summary


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    """
    Execute the full preprocessing pipeline and save all outputs.
    """
    print("\n" + "=" * 65)
    print("  Hypotify Week 8 – NLP Preprocessing Pipeline")
    print("=" * 65 + "\n")

    # ── Step 1: Load training dataset ─────────────────────────────────────────
    log.info("Step 1: Loading chatbot training dataset …")
    try:
        df = load_dataset(DATASET_PATH)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)

    # ── Step 2-5: Preprocess each user_input entry ────────────────────────────
    log.info("Step 2-5: Running NLP preprocessing pipeline …")
    preprocessed_records = []

    for idx, row in df.iterrows():
        result = full_preprocess(str(row["user_input"]))
        record = {
            "index":         int(idx),
            "original":      str(row["user_input"]),
            "intent":        str(row["intent"]),
            "response":      str(row["response"]),
            "cleaned":       result["cleaned"],
            "tokens":        result["tokens"],
            "filtered":      result["filtered"],
            "lemmas":        result["lemmas"],
            "joined":        result["joined"],   # used as model input
        }
        preprocessed_records.append(record)

    log.info(f"Preprocessing complete: {len(preprocessed_records)} records processed.")

    # ── Step 6: TF-IDF analysis ───────────────────────────────────────────────
    log.info("Step 6: Computing TF-IDF features …")
    corpus          = [r["joined"] for r in preprocessed_records]
    tfidf_summary   = compute_tfidf(corpus, top_n=10)

    # ── Step 7: Word2Vec embeddings ───────────────────────────────────────────
    log.info("Step 7: Training Word2Vec word embeddings …")
    tokenized_sentences = [r["lemmas"] for r in preprocessed_records if r["lemmas"]]
    w2v_summary         = train_word2vec(tokenized_sentences, WORD2VEC_PATH)

    # ── Step 8: Patient data statistics ──────────────────────────────────────
    log.info("Step 8: Building patient data summary statistics …")
    patient_summary = build_patient_summary()

    # ── Step 9: Save all outputs ──────────────────────────────────────────────
    log.info("Step 9: Saving outputs …")

    # 9a: preprocessed_data.json
    output_payload = {
        "metadata": {
            "total_records":        len(preprocessed_records),
            "unique_intents":       len(df["intent"].unique()),
            "intent_classes":       sorted(df["intent"].unique().tolist()),
            "preprocessing_steps":  [
                "1. Lowercase + HTML strip + punctuation removal + whitespace collapse",
                "2. NLTK word_tokenize (Penn Treebank tokeniser)",
                "3. NLTK English stop-word removal",
                "4. NLTK WordNetLemmatizer",
            ],
            "tfidf_summary":        tfidf_summary,
            "word2vec_summary":     w2v_summary,
        },
        "records": preprocessed_records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    log.info(f"Saved preprocessed data → {OUTPUT_JSON}")

    # 9b: patient_summary.json
    with open(PATIENT_SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(patient_summary, f, indent=2, ensure_ascii=False)
    log.info(f"Saved patient summary   → {PATIENT_SUMMARY_OUT}")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Records processed   : {len(preprocessed_records)}")
    print(f"  Unique intents      : {output_payload['metadata']['unique_intents']}")
    print(f"  TF-IDF top terms    : {tfidf_summary.get('global_top_terms', [])}")
    print(f"  Word2Vec status     : {w2v_summary.get('status', 'n/a')}")
    if "patients" in patient_summary:
        ps = patient_summary["patients"]
        print(f"  Total patients      : {ps.get('total_patients', 'n/a')}")
        print(f"  Mean patient age    : {ps.get('age_mean', 'n/a')}")
    print(f"\n  Output files:")
    print(f"    {OUTPUT_JSON}")
    print(f"    {PATIENT_SUMMARY_OUT}")
    if w2v_summary.get("status") == "trained":
        print(f"    {WORD2VEC_PATH}")
    print("=" * 65 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Preprocessing interrupted by user.\n")
        sys.exit(0)
    except Exception as exc:
        log.error(f"Unexpected error: {exc}", exc_info=True)
        sys.exit(1)
