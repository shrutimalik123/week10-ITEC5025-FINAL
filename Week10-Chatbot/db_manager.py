"""
db_manager.py
=============
Week 9 Assignment – Database Integration for Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-09

Purpose:
    Central CRUD module for all database interactions.
    Imported by chatbot_w9.py so the chatbot never writes raw SQL itself.

    Functions provided:
        connect()                              – open a SQLite connection
        get_intent_response(tag)               – fetch a random response for an intent
        get_all_intents()                      – list every unique intent tag + count
        log_conversation(...)                  – INSERT a chat turn into conversation_log
        add_user_feedback(...)                 – INSERT user rating/comment
        get_conversation_history(session_id)   – SELECT the last N turns for a session
        search_intents(keyword)                – search patterns + responses by keyword
        get_db_stats()                         – row counts for every table
        get_summary_stat(category, key)        – look up a value from patients_summary
"""

import os
import random
import sqlite3
import logging

log = logging.getLogger(__name__)

# ── Path resolution ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(SCRIPT_DIR, "hypotify.db")

# ── Connection factory ─────────────────────────────────────────────────────────

def _reassemble_db():
    """Reassemble the database from parts if necessary."""
    part1 = DB_PATH + ".part1"
    part2 = DB_PATH + ".part2"
    
    if not os.path.exists(DB_PATH):
        if os.path.exists(part1) and os.path.exists(part2):
            log.info("Reassembling hypotify.db from parts...")
            with open(DB_PATH, "wb") as f_out:
                for p in [part1, part2]:
                    with open(p, "rb") as f_in:
                        f_out.write(f_in.read())
            log.info("Reassembly complete.")
        else:
            # Check for legacy paths or other issues
            pass

def connect() -> sqlite3.Connection:
    """
    Open and return a new SQLite connection to hypotify.db.
    Reassembles the DB from parts if missing but parts are present.
    """
    _reassemble_db()
    
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run  python db_setup.py  first to create and seed the database."
        )
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# ── Read operations ────────────────────────────────────────────────────────────

def get_intent_response(tag: str) -> str | None:
    """
    Retrieve a random chatbot response for the given intent tag.

    Queries the intents table for all responses matching the tag,
    then returns one at random. Returns None if no match is found.

    Args:
        tag (str): Intent label, e.g. 'greeting', 'insights_gender'.

    Returns:
        str | None: A response string, or None if the tag is not in the DB.
    """
    try:
        conn = connect()
        rows = conn.execute(
            "SELECT response FROM intents WHERE tag = ?", (tag,)
        ).fetchall()
        conn.close()
        if not rows:
            return None
        return random.choice(rows)["response"]
    except Exception as exc:
        log.warning(f"get_intent_response({tag!r}) failed: {exc}")
        return None


def get_all_intents() -> list[dict]:
    """
    Return a list of every unique intent tag with its response count.

    Returns:
        list of dicts with keys: tag, count
    """
    try:
        conn = connect()
        rows = conn.execute(
            "SELECT tag, COUNT(*) AS cnt FROM intents GROUP BY tag ORDER BY tag"
        ).fetchall()
        conn.close()
        return [{"tag": row["tag"], "count": row["cnt"]} for row in rows]
    except Exception as exc:
        log.warning(f"get_all_intents() failed: {exc}")
        return []


def search_intents(keyword: str) -> list[dict]:
    """
    Search the intents table for patterns or responses containing a keyword.

    Performs a case-insensitive LIKE search on both the pattern and response fields.

    Args:
        keyword (str): Search term.

    Returns:
        list of dicts with keys: tag, pattern, response
    """
    try:
        conn   = connect()
        like   = f"%{keyword}%"
        rows   = conn.execute(
            """SELECT tag, pattern, response
               FROM   intents
               WHERE  pattern  LIKE ? OR response LIKE ?
               LIMIT  20""",
            (like, like)
        ).fetchall()
        conn.close()
        return [
            {"tag": r["tag"], "pattern": r["pattern"], "response": r["response"]}
            for r in rows
        ]
    except Exception as exc:
        log.warning(f"search_intents({keyword!r}) failed: {exc}")
        return []


def get_summary_stat(category: str, key: str) -> str | None:
    """
    Look up a single value from the patients_summary table.

    Args:
        category (str): Category column value, e.g. 'gender'.
        key      (str): Key column value, e.g. 'Female'.

    Returns:
        str | None: The stored value, or None if not found.
    """
    try:
        conn = connect()
        row  = conn.execute(
            "SELECT value FROM patients_summary WHERE category = ? AND key = ?",
            (category, key)
        ).fetchone()
        conn.close()
        return row["value"] if row else None
    except Exception as exc:
        log.warning(f"get_summary_stat() failed: {exc}")
        return None


def get_conversation_history(session_id: str, limit: int = 10) -> list[dict]:
    """
    Retrieve the most recent conversation turns for a given session.

    Args:
        session_id (str): Unique identifier for the chat session.
        limit      (int): Maximum number of turns to return (default 10).

    Returns:
        list of dicts with keys: timestamp, user_message, bot_response, intent_tag
    """
    try:
        conn = connect()
        rows = conn.execute(
            """SELECT timestamp, user_message, bot_response, intent_tag
               FROM   conversation_log
               WHERE  session_id = ?
               ORDER  BY id DESC
               LIMIT  ?""",
            (session_id, limit)
        ).fetchall()
        conn.close()
        return [
            {
                "timestamp":    r["timestamp"],
                "user_message": r["user_message"],
                "bot_response": r["bot_response"],
                "intent_tag":   r["intent_tag"],
            }
            for r in reversed(rows)   # return in chronological order
        ]
    except Exception as exc:
        log.warning(f"get_conversation_history() failed: {exc}")
        return []


def get_db_stats() -> dict:
    """
    Return row counts for every table in the database.

    Returns:
        dict mapping table name → row count
    """
    tables = ("intents", "patients_summary", "conversation_log", "user_feedback")
    stats  = {}
    try:
        conn = connect()
        for table in tables:
            row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
            stats[table] = row["cnt"]
        conn.close()
    except Exception as exc:
        log.warning(f"get_db_stats() failed: {exc}")
    return stats


# ── Write operations ───────────────────────────────────────────────────────────

def log_conversation(session_id: str,
                     user_message: str,
                     bot_response: str,
                     intent_tag: str | None = None) -> bool:
    """
    INSERT a single conversation turn into the conversation_log table.

    Args:
        session_id   (str)      : Unique session identifier.
        user_message (str)      : The raw message typed by the user.
        bot_response (str)      : The chatbot's response string.
        intent_tag   (str|None) : Predicted intent label (optional).

    Returns:
        bool: True if the insert succeeded, False on error.
    """
    try:
        conn = connect()
        conn.execute(
            """INSERT INTO conversation_log
               (session_id, user_message, bot_response, intent_tag)
               VALUES (?, ?, ?, ?)""",
            (session_id, user_message, bot_response, intent_tag)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        log.warning(f"log_conversation() failed: {exc}")
        return False


def add_user_feedback(session_id: str,
                      rating: int,
                      comment: str = "") -> bool:
    """
    INSERT a user feedback record into the user_feedback table.

    Args:
        session_id (str) : Session this feedback belongs to.
        rating     (int) : Integer score 1–5.
        comment    (str) : Optional free-text comment.

    Returns:
        bool: True if the insert succeeded, False on error.
    """
    if not (1 <= rating <= 5):
        log.warning(f"add_user_feedback: rating {rating} out of range 1-5.")
        return False
    try:
        conn = connect()
        conn.execute(
            """INSERT INTO user_feedback (session_id, rating, comment)
               VALUES (?, ?, ?)""",
            (session_id, rating, comment)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        log.warning(f"add_user_feedback() failed: {exc}")
        return False


def update_intent_response(tag: str, pattern: str, new_response: str) -> bool:
    """
    UPDATE the response for a specific (tag, pattern) pair in the intents table.

    This allows the chatbot to store updated knowledge back into the database.

    Args:
        tag          (str): Intent label.
        pattern      (str): The pattern whose response will be updated.
        new_response (str): The new response text.

    Returns:
        bool: True if a row was updated, False otherwise.
    """
    try:
        conn = connect()
        cursor = conn.execute(
            "UPDATE intents SET response = ? WHERE tag = ? AND pattern = ?",
            (new_response, tag, pattern)
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated
    except Exception as exc:
        log.warning(f"update_intent_response() failed: {exc}")
        return False


def insert_intent(tag: str, pattern: str, response: str) -> bool:
    """
    INSERT a new (tag, pattern, response) row into the intents table.

    Enables the chatbot to learn new patterns at runtime.

    Args:
        tag      (str): Intent label.
        pattern  (str): New user phrase pattern.
        response (str): Chatbot response for this phrase.

    Returns:
        bool: True if the insert succeeded.
    """
    try:
        conn = connect()
        conn.execute(
            "INSERT INTO intents (tag, pattern, response) VALUES (?, ?, ?)",
            (tag, pattern, response)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as exc:
        log.warning(f"insert_intent() failed: {exc}")
        return False


# ── Clinical patient DB queries (populated by import_patients.py) ──────────────

def _clinical_tables_exist(conn: sqlite3.Connection) -> bool:
    """Return True if the patients table exists and has data."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM sqlite_master "
            "WHERE type='table' AND name='patients'"
        ).fetchone()
        if row["cnt"] == 0:
            return False
        # Also check it has data
        row2 = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()
        return row2["cnt"] > 0
    except Exception:
        return False


def get_patient_from_db(patient_id: str) -> str:
    """
    Look up a patient record directly from the SQLite patients table.

    Falls back to a helpful message if the clinical tables are not yet loaded.

    Args:
        patient_id (str): Patient ID, e.g. 'P000042' or a UUID.

    Returns:
        str: Formatted patient summary, or an appropriate message.
    """
    try:
        conn = connect()
        if not _clinical_tables_exist(conn):
            conn.close()
            return (
                "Patient records are not yet loaded into the database.\n"
                "Run  python import_patients.py  to import all 100,000 patients."
            )

        # Try exact match first; then case-insensitive
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM patients WHERE UPPER(patient_id) = UPPER(?)",
                (patient_id,)
            ).fetchone()
        conn.close()

        if row is None:
            return f"No patient found with ID: {patient_id}"

        lines = [f"\n  Patient Record: {row['patient_id']}", "  " + "-" * 42]
        for col, label in [
            ("gender",         "Gender"),
            ("date_of_birth",  "Date of Birth"),
            ("race",           "Race"),
            ("marital_status", "Marital Status"),
            ("language",       "Language"),
            ("poverty_pct",    "Poverty %"),
        ]:
            val = row[col] if row[col] is not None else "N/A"
            lines.append(f"  {label:<20}: {val}")
        return "\n".join(lines)

    except Exception as exc:
        log.warning(f"get_patient_from_db({patient_id!r}) failed: {exc}")
        return f"Error retrieving patient {patient_id}: {exc}"


def get_patient_admissions(patient_id: str) -> str:
    """
    Retrieve all hospital admissions for a patient from the SQLite admissions table.

    Args:
        patient_id (str): Patient identifier.

    Returns:
        str: Formatted admissions history string.
    """
    try:
        conn = connect()
        if not _clinical_tables_exist(conn):
            conn.close()
            return "Admissions data not yet loaded. Run  python import_patients.py  first."

        rows = conn.execute(
            """SELECT admission_id, admission_start, admission_end
               FROM admissions WHERE patient_id = ?
               ORDER BY admission_start""",
            (patient_id,)
        ).fetchall()
        conn.close()

        if not rows:
            return f"No admission records found for patient {patient_id}."

        lines = [f"\n  Admissions for {patient_id} ({len(rows)} record(s)):"]
        lines.append("  " + "-" * 52)
        for r in rows:
            start = r["admission_start"] or "N/A"
            end   = r["admission_end"]   or "N/A"
            lines.append(f"  Admission {r['admission_id'][:12]}…  {start}  →  {end}")
        return "\n".join(lines)

    except Exception as exc:
        log.warning(f"get_patient_admissions() failed: {exc}")
        return f"Error retrieving admissions: {exc}"


def get_patient_diagnoses(patient_id: str) -> str:
    """
    Retrieve all ICD-10 diagnoses for a patient from the SQLite diagnoses table.

    Args:
        patient_id (str): Patient identifier.

    Returns:
        str: Formatted diagnosis list string.
    """
    try:
        conn = connect()
        if not _clinical_tables_exist(conn):
            conn.close()
            return "Diagnoses data not yet loaded. Run  python import_patients.py  first."

        rows = conn.execute(
            """SELECT diagnosis_code, diagnosis_description
               FROM diagnoses WHERE patient_id = ?
               LIMIT 20""",
            (patient_id,)
        ).fetchall()
        conn.close()

        if not rows:
            return f"No diagnosis records found for patient {patient_id}."

        lines = [f"\n  Diagnoses for {patient_id} (up to 20 shown):"]
        lines.append("  " + "-" * 52)
        seen = set()
        for r in rows:
            desc  = (r["diagnosis_description"] or "Unknown")[:55]
            code  = r["diagnosis_code"] or "?"
            entry = f"{code}: {desc}"
            if entry not in seen:
                lines.append(f"  {entry}")
                seen.add(entry)
        return "\n".join(lines)

    except Exception as exc:
        log.warning(f"get_patient_diagnoses() failed: {exc}")
        return f"Error retrieving diagnoses: {exc}"


def get_patient_labs(patient_id: str) -> str:
    """
    Retrieve lab results for a patient from the SQLite labs table.

    Args:
        patient_id (str): Patient identifier.

    Returns:
        str: Formatted lab results string.
    """
    try:
        conn = connect()
        if not _clinical_tables_exist(conn):
            conn.close()
            return "Lab data not yet loaded. Run  python import_patients.py  first."

        # Check labs table exists
        has_labs = conn.execute(
            "SELECT COUNT(*) AS cnt FROM sqlite_master WHERE type='table' AND name='labs'"
        ).fetchone()["cnt"] > 0

        if not has_labs:
            conn.close()
            return "Labs table not found. Run  python import_patients.py  first."

        rows = conn.execute(
            """SELECT lab_name, lab_value, lab_units, lab_datetime
               FROM labs WHERE patient_id = ?
               ORDER BY lab_datetime DESC
               LIMIT 15""",
            (patient_id,)
        ).fetchall()
        conn.close()

        if not rows:
            return (f"No lab records found for patient {patient_id} "
                    f"in the sampled dataset (100K rows).")

        lines = [f"\n  Lab Results for {patient_id} (most recent 15):"]
        lines.append("  " + "-" * 58)
        for r in rows:
            name  = (r["lab_name"]  or "Unknown")[:30]
            val   = (r["lab_value"] or "N/A")
            units = (r["lab_units"] or "")
            dt    = (r["lab_datetime"] or "")[:16]
            lines.append(f"  {name:<32} {val:>8} {units:<8}  {dt}")
        return "\n".join(lines)

    except Exception as exc:
        log.warning(f"get_patient_labs() failed: {exc}")
        return f"Error retrieving lab results: {exc}"


def get_population_stats_from_db(category: str) -> str | None:
    """
    Compute population-level statistics directly from the patients SQL table.

    This uses the full 100,000-patient dataset (after import_patients.py runs),
    providing more accurate results than the pre-computed patient_summary.json.

    Args:
        category (str): One of 'gender', 'race', 'language', 'marital', 'age', 'poverty'.

    Returns:
        str | None: Formatted statistics string, or None if tables not loaded.
    """
    try:
        conn = connect()
        if not _clinical_tables_exist(conn):
            conn.close()
            return None  # Chatbot will fall back to CSV-based stats

        category = category.lower()

        if category == "gender":
            rows = conn.execute(
                "SELECT gender, COUNT(*) AS cnt FROM patients "
                "GROUP BY gender ORDER BY cnt DESC"
            ).fetchall()
            total = sum(r["cnt"] for r in rows)
            lines = [f"\n  Gender Distribution (DB: {total:,} patients):"]
            for r in rows:
                pct = r["cnt"] / total * 100
                lines.append(f"    {(r['gender'] or 'Unknown'):<15}: {r['cnt']:>7,}  ({pct:.1f}%)")
            conn.close()
            return "\n".join(lines)

        elif category == "race":
            rows = conn.execute(
                "SELECT race, COUNT(*) AS cnt FROM patients "
                "GROUP BY race ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()["cnt"]
            lines = [f"\n  Race Distribution – Top 10 (DB: {total:,} patients):"]
            for r in rows:
                pct = r["cnt"] / total * 100
                lines.append(f"    {(r['race'] or 'Unknown'):<25}: {r['cnt']:>7,}  ({pct:.1f}%)")
            conn.close()
            return "\n".join(lines)

        elif category == "language":
            rows = conn.execute(
                "SELECT language, COUNT(*) AS cnt FROM patients "
                "GROUP BY language ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()["cnt"]
            lines = [f"\n  Language Distribution – Top 10 (DB: {total:,} patients):"]
            for r in rows:
                pct = r["cnt"] / total * 100
                lines.append(f"    {(r['language'] or 'Unknown'):<22}: {r['cnt']:>7,}  ({pct:.1f}%)")
            conn.close()
            return "\n".join(lines)

        elif category == "marital":
            rows = conn.execute(
                "SELECT marital_status, COUNT(*) AS cnt FROM patients "
                "GROUP BY marital_status ORDER BY cnt DESC"
            ).fetchall()
            total = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()["cnt"]
            lines = [f"\n  Marital Status Distribution (DB: {total:,} patients):"]
            for r in rows:
                pct = r["cnt"] / total * 100
                lines.append(f"    {(r['marital_status'] or 'Unknown'):<20}: {r['cnt']:>7,}  ({pct:.1f}%)")
            conn.close()
            return "\n".join(lines)

        elif category == "poverty":
            row = conn.execute(
                "SELECT AVG(poverty_pct) AS avg_p, "
                "       COUNT(CASE WHEN poverty_pct > 0 THEN 1 END) AS with_data "
                "FROM patients"
            ).fetchone()
            total = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()["cnt"]
            conn.close()
            avg_p     = row["avg_p"] or 0
            with_data = row["with_data"] or 0
            return (
                f"\n  Poverty Statistics (DB: {total:,} patients):\n"
                f"    Mean poverty %       : {avg_p:.1f}%\n"
                f"    Patients w/ data     : {with_data:,}"
            )

        elif category == "age":
            # Compute age from date_of_birth using SQLite date functions
            row = conn.execute(
                """SELECT
                     AVG((julianday('now') - julianday(date_of_birth)) / 365.25) AS avg_age,
                     MIN((julianday('now') - julianday(date_of_birth)) / 365.25) AS min_age,
                     MAX((julianday('now') - julianday(date_of_birth)) / 365.25) AS max_age,
                     COUNT(*) AS cnt
                   FROM patients
                   WHERE date_of_birth IS NOT NULL"""
            ).fetchone()
            conn.close()
            return (
                f"\n  Age Statistics (DB: {row['cnt']:,} patients with DOB):\n"
                f"    Mean age   : {row['avg_age']:.1f} years\n"
                f"    Youngest   : {row['min_age']:.1f} years\n"
                f"    Oldest     : {row['max_age']:.1f} years"
            )

        conn.close()
        return None

    except Exception as exc:
        log.warning(f"get_population_stats_from_db({category!r}) failed: {exc}")
        return None


def get_db_stats() -> dict:
    """
    Return row counts for every table in the database (intents + clinical).

    Returns:
        dict mapping table name → row count (0 if table doesn't exist yet)
    """
    all_tables = (
        "intents", "patients_summary", "conversation_log", "user_feedback",
        "patients", "admissions", "diagnoses", "labs",
    )
    stats = {}
    try:
        conn = connect()
        for table in all_tables:
            try:
                row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
                stats[table] = row["cnt"]
            except sqlite3.OperationalError:
                stats[table] = 0   # Table doesn't exist yet
        conn.close()
    except Exception as exc:
        log.warning(f"get_db_stats() failed: {exc}")
    return stats
