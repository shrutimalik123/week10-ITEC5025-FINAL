"""
db_setup.py
===========
Week 9 Assignment – Database Integration for Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-09

Purpose:
    Create and seed the SQLite database (hypotify.db) used by the Week 9 chatbot.

    Tables created:
        intents          – chatbot intent patterns and responses (from dataset.csv)
        patients_summary – aggregated population statistics (from patient_summary.json)
        conversation_log – every chat turn recorded by the chatbot at runtime
        user_feedback    – optional ratings/comments submitted by users

Usage:
    python db_setup.py
    python db_setup.py --reset    # drop and recreate all tables
"""

import csv
import json
import os
import sqlite3
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
WEEK8_DIR    = os.path.join(ROOT_DIR, "Week8-Chatbot")

DB_PATH      = os.path.join(SCRIPT_DIR, "hypotify.db")
DATASET_CSV  = os.path.join(WEEK8_DIR,  "dataset.csv")
SUMMARY_JSON = os.path.join(WEEK8_DIR,  "patient_summary.json")


# ── Schema ─────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- intents: stores all chatbot intent patterns and their responses
CREATE TABLE IF NOT EXISTS intents (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    tag      TEXT    NOT NULL,          -- intent label, e.g. 'greeting'
    pattern  TEXT    NOT NULL,          -- example user phrase
    response TEXT    NOT NULL           -- chatbot response text
);

-- patients_summary: aggregated population statistics
CREATE TABLE IF NOT EXISTS patients_summary (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT    NOT NULL,          -- e.g. 'gender', 'language'
    key      TEXT    NOT NULL,          -- e.g. 'Female', 'English'
    value    TEXT    NOT NULL           -- numeric or text value
);

-- conversation_log: every chat turn recorded at runtime
CREATE TABLE IF NOT EXISTS conversation_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    user_message TEXT    NOT NULL,
    bot_response TEXT    NOT NULL,
    intent_tag   TEXT
);

-- user_feedback: optional ratings submitted by users
CREATE TABLE IF NOT EXISTS user_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    timestamp  TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    rating     INTEGER CHECK(rating BETWEEN 1 AND 5),
    comment    TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with foreign-key support enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    """Execute the schema SQL to create all tables if they do not exist."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    print("  [✓] Tables created (or already exist).")


def drop_tables(conn: sqlite3.Connection) -> None:
    """Drop all tables for a clean reset."""
    for table in ("intents", "patients_summary", "conversation_log", "user_feedback"):
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    print("  [✓] All tables dropped.")


def seed_intents(conn: sqlite3.Connection) -> int:
    """
    Seed the intents table from dataset.csv.

    The CSV has three columns: user_input, intent, response.
    Each row becomes one (tag, pattern, response) record.

    Returns:
        int: Number of rows inserted.
    """
    if not os.path.exists(DATASET_CSV):
        print(f"  [!] dataset.csv not found at {DATASET_CSV} – skipping intents seed.")
        return 0

    rows_inserted = 0
    with open(DATASET_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pattern  = (row.get("user_input") or "").strip()
            tag      = (row.get("intent")     or "").strip()
            response = (row.get("response")   or "").strip()

            # Skip blank or header-like rows
            if not pattern or not tag or not response:
                continue

            conn.execute(
                "INSERT INTO intents (tag, pattern, response) VALUES (?, ?, ?)",
                (tag, pattern, response)
            )
            rows_inserted += 1

    conn.commit()
    print(f"  [✓] Seeded {rows_inserted} intent rows from dataset.csv.")
    return rows_inserted


def seed_patients_summary(conn: sqlite3.Connection) -> int:
    """
    Seed the patients_summary table from patient_summary.json.

    Flattens the nested JSON structure into (category, key, value) rows.

    Returns:
        int: Number of rows inserted.
    """
    if not os.path.exists(SUMMARY_JSON):
        print(f"  [!] patient_summary.json not found at {SUMMARY_JSON} – skipping.")
        return 0

    with open(SUMMARY_JSON, encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    # ── patients section ──────────────────────────────────────────────────────
    patients = data.get("patients", {})

    # Scalar values
    for key in ("total_patients", "poverty_pct_mean", "poverty_pct_median",
                "age_mean", "age_min", "age_max"):
        if key in patients:
            rows.append(("patients", key, str(patients[key])))

    # Distribution dicts
    for category_key in ("gender_distribution", "race_distribution",
                          "language_distribution", "marital_status"):
        dist = patients.get(category_key, {})
        pretty = category_key.replace("_distribution", "").replace("_", " ")
        for sub_key, val in dist.items():
            rows.append((pretty, sub_key, str(val)))

    # ── admissions section ────────────────────────────────────────────────────
    admissions = data.get("admissions", {})
    for key, val in admissions.items():
        rows.append(("admissions", key, str(val)))

    # ── diagnoses section ─────────────────────────────────────────────────────
    diagnoses = data.get("diagnoses", {})
    for key in ("total_diagnosis_records", "unique_diagnosis_codes"):
        if key in diagnoses:
            rows.append(("diagnoses", key, str(diagnoses[key])))
    # Top diagnoses
    for diag, cnt in diagnoses.get("top_10_diagnoses", {}).items():
        rows.append(("top_diagnosis", diag, str(cnt)))

    # ── labs section ──────────────────────────────────────────────────────────
    labs = data.get("labs", {})
    for key in ("sampled_rows", "unique_lab_tests", "unique_patients"):
        if key in labs:
            rows.append(("labs", key, str(labs[key])))
    for test, cnt in labs.get("top_10_lab_tests", {}).items():
        rows.append(("top_lab_test", test, str(cnt)))

    conn.executemany(
        "INSERT INTO patients_summary (category, key, value) VALUES (?, ?, ?)",
        rows
    )
    conn.commit()
    print(f"  [✓] Seeded {len(rows)} patient summary rows from patient_summary.json.")
    return len(rows)


def print_db_stats(conn: sqlite3.Connection) -> None:
    """Print row counts for every table."""
    print("\n  Database Statistics:")
    print("  " + "-" * 40)
    for table in ("intents", "patients_summary", "conversation_log", "user_feedback"):
        row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
        print(f"    {table:<22}: {row['cnt']:>6} rows")
    print("  " + "-" * 40)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    reset = "--reset" in sys.argv

    print("\n  Hypotify Clinical Chatbot – Week 9 Database Setup")
    print("  " + "=" * 48)
    print(f"  Database path : {DB_PATH}")
    print()

    conn = get_connection()

    try:
        if reset:
            print("  [!] --reset flag detected – dropping existing tables …")
            drop_tables(conn)

        create_tables(conn)
        seed_intents(conn)
        seed_patients_summary(conn)
        print_db_stats(conn)
        print("\n  [✓] Database setup complete.\n")

    except sqlite3.Error as exc:
        print(f"\n  [ERROR] SQLite error: {exc}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
