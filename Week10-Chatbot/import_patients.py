"""
import_patients.py
=================
Week 9 Assignment – Database Integration for Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-09

Purpose:
    Import all 100,000 patient records plus associated admissions, diagnoses,
    and lab results from the four cleaned CSV files into the hypotify.db
    SQLite database.

    This extends the Week 9 database beyond the intent/summary tables so that
    the chatbot can retrieve full patient data directly from SQL queries instead
    of loading large CSV files at runtime.

    Tables populated by this script:
        patients        – 100,000 patient demographics
        admissions      – 361,760 hospital admission events
        diagnoses       – 361,760 ICD-10 diagnosis records
        labs            – up to LAB_SAMPLE_SIZE lab results (default 100,000)
                          (the full labs CSV is >1 GB; a representative sample
                           is used to keep DB size manageable)

Usage:
    python import_patients.py
    python import_patients.py --labs-sample 50000   # smaller lab sample
    python import_patients.py --no-labs             # skip labs entirely

Runtime estimate:
    patients  : ~30 s     (100K rows, 7 cols)
    admissions: ~2 min    (361K rows, 4 cols)
    diagnoses : ~2 min    (361K rows, 4 cols)
    labs      : ~1 min    (100K sample, 6 cols)
"""

import os
import sqlite3
import sys
import time

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(ROOT_DIR, "data")

DB_PATH      = os.path.join(SCRIPT_DIR, "hypotify.db")
PATIENTS_CSV = os.path.join(DATA_DIR, "patients_cleaned.csv")
ADMISS_CSV   = os.path.join(DATA_DIR, "admissions_cleaned.csv")
DIAGN_CSV    = os.path.join(DATA_DIR, "diagnoses_cleaned.csv")
LABS_CSV     = os.path.join(DATA_DIR, "labs_cleaned.csv")

# Default lab sample size (labs CSV is >1 GB; full load is optional)
LAB_SAMPLE_SIZE = 100_000

# ── Schema for clinical tables ─────────────────────────────────────────────────
CLINICAL_SCHEMA = """
-- Full patient demographics (100,000 rows from patients_cleaned.csv)
CREATE TABLE IF NOT EXISTS patients (
    patient_id     TEXT PRIMARY KEY,
    gender         TEXT,
    date_of_birth  TEXT,
    race           TEXT,
    marital_status TEXT,
    language       TEXT,
    poverty_pct    REAL
);

-- Hospital admission events (361,760 rows from admissions_cleaned.csv)
CREATE TABLE IF NOT EXISTS admissions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id       TEXT NOT NULL,
    admission_id     TEXT NOT NULL,
    admission_start  TEXT,
    admission_end    TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ICD-10 diagnosis records (361,760 rows from diagnoses_cleaned.csv)
CREATE TABLE IF NOT EXISTS diagnoses (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            TEXT NOT NULL,
    admission_id          TEXT NOT NULL,
    diagnosis_code        TEXT,
    diagnosis_description TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Lab results (sampled from labs_cleaned.csv — default 100,000 rows)
CREATE TABLE IF NOT EXISTS labs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id   TEXT NOT NULL,
    admission_id TEXT NOT NULL,
    lab_name     TEXT,
    lab_value    TEXT,
    lab_units    TEXT,
    lab_datetime TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);
"""

# ── Indexes for fast patient-ID lookup ─────────────────────────────────────────
INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_admissions_patient   ON admissions(patient_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_patient    ON diagnoses(patient_id);
CREATE INDEX IF NOT EXISTS idx_labs_patient         ON labs(patient_id);
CREATE INDEX IF NOT EXISTS idx_admissions_id        ON admissions(admission_id);
"""


def get_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection with row factory."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run  python db_setup.py  first."
        )
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode = WAL")   # Write-Ahead Logging for speed
    conn.execute("PRAGMA synchronous  = NORMAL")
    conn.execute("PRAGMA cache_size   = -64000") # 64 MB page cache
    conn.row_factory = sqlite3.Row
    return conn


def create_clinical_tables(conn: sqlite3.Connection) -> None:
    """Create the four clinical tables if they do not already exist."""
    conn.executescript(CLINICAL_SCHEMA)
    conn.commit()
    print("  [✓] Clinical tables created (or already exist).")


def create_indexes(conn: sqlite3.Connection) -> None:
    """Build indexes on patient_id columns for fast lookup."""
    conn.executescript(INDEX_SQL)
    conn.commit()
    print("  [✓] Indexes created.")


def _table_is_populated(conn: sqlite3.Connection, table: str) -> bool:
    """Return True if the table already has at least one row."""
    row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
    return row["cnt"] > 0


def import_patients(conn: sqlite3.Connection) -> int:
    """
    Import all 100,000 patient demographics into the patients table.

    Uses pandas to read the CSV in one shot, then writes to SQLite using
    DataFrame.to_sql() with if_exists='append' for efficiency.

    Returns:
        int: Number of rows inserted.
    """
    if _table_is_populated(conn, "patients"):
        count = conn.execute("SELECT COUNT(*) AS cnt FROM patients").fetchone()["cnt"]
        print(f"  [–] patients table already has {count:,} rows — skipping import.")
        return count

    if not os.path.exists(PATIENTS_CSV):
        print(f"  [!] {PATIENTS_CSV} not found — skipping.")
        return 0

    print(f"  Importing patients from {os.path.basename(PATIENTS_CSV)} …", flush=True)
    t0 = time.time()
    df = pd.read_csv(PATIENTS_CSV, encoding="utf-8", on_bad_lines="skip", dtype=str)

    # Ensure poverty_pct is numeric
    df["poverty_pct"] = pd.to_numeric(df.get("poverty_pct"), errors="coerce")

    # Use to_sql with chunksize for memory efficiency
    df.to_sql("patients", conn, if_exists="append", index=False, chunksize=5_000)
    conn.commit()
    elapsed = time.time() - t0
    print(f"  [✓] Imported {len(df):,} patient rows  ({elapsed:.1f}s)")
    return len(df)


def import_admissions(conn: sqlite3.Connection) -> int:
    """
    Import all hospital admission records into the admissions table.

    Returns:
        int: Number of rows inserted.
    """
    if _table_is_populated(conn, "admissions"):
        count = conn.execute("SELECT COUNT(*) AS cnt FROM admissions").fetchone()["cnt"]
        print(f"  [–] admissions table already has {count:,} rows — skipping.")
        return count

    if not os.path.exists(ADMISS_CSV):
        print(f"  [!] {ADMISS_CSV} not found — skipping.")
        return 0

    print(f"  Importing admissions from {os.path.basename(ADMISS_CSV)} …", flush=True)
    t0    = time.time()
    total = 0
    # Read in chunks to avoid loading 28 MB fully into memory at once
    for chunk in pd.read_csv(ADMISS_CSV, encoding="utf-8",
                              on_bad_lines="skip", dtype=str, chunksize=20_000):
        chunk.to_sql("admissions", conn, if_exists="append", index=False)
        total += len(chunk)
        print(f"    … {total:,} rows", end="\r", flush=True)
    conn.commit()
    elapsed = time.time() - t0
    print(f"  [✓] Imported {total:,} admission rows  ({elapsed:.1f}s)          ")
    return total


def import_diagnoses(conn: sqlite3.Connection) -> int:
    """
    Import all ICD-10 diagnosis records into the diagnoses table.

    Returns:
        int: Number of rows inserted.
    """
    if _table_is_populated(conn, "diagnoses"):
        count = conn.execute("SELECT COUNT(*) AS cnt FROM diagnoses").fetchone()["cnt"]
        print(f"  [–] diagnoses table already has {count:,} rows — skipping.")
        return count

    if not os.path.exists(DIAGN_CSV):
        print(f"  [!] {DIAGN_CSV} not found — skipping.")
        return 0

    print(f"  Importing diagnoses from {os.path.basename(DIAGN_CSV)} …", flush=True)
    t0    = time.time()
    total = 0
    for chunk in pd.read_csv(DIAGN_CSV, encoding="utf-8",
                              on_bad_lines="skip", dtype=str, chunksize=20_000):
        chunk.to_sql("diagnoses", conn, if_exists="append", index=False)
        total += len(chunk)
        print(f"    … {total:,} rows", end="\r", flush=True)
    conn.commit()
    elapsed = time.time() - t0
    print(f"  [✓] Imported {total:,} diagnosis rows  ({elapsed:.1f}s)          ")
    return total


def import_labs(conn: sqlite3.Connection, sample_size: int = LAB_SAMPLE_SIZE) -> int:
    """
    Import a sample of lab results into the labs table.

    The full labs CSV is >1 GB. We read it in chunks, accumulating rows until
    we reach sample_size. This keeps the database at a reasonable size while
    still providing real lab data for chatbot queries.

    Args:
        sample_size (int): Maximum number of lab rows to import.

    Returns:
        int: Number of rows inserted.
    """
    if _table_is_populated(conn, "labs"):
        count = conn.execute("SELECT COUNT(*) AS cnt FROM labs").fetchone()["cnt"]
        print(f"  [–] labs table already has {count:,} rows — skipping.")
        return count

    if not os.path.exists(LABS_CSV):
        print(f"  [!] {LABS_CSV} not found — skipping.")
        return 0

    print(f"  Importing labs sample ({sample_size:,} rows) …", flush=True)
    t0    = time.time()
    total = 0
    for chunk in pd.read_csv(LABS_CSV, encoding="utf-8",
                              on_bad_lines="skip", dtype=str, chunksize=10_000):
        remaining = sample_size - total
        if remaining <= 0:
            break
        sub = chunk.head(remaining)
        sub.to_sql("labs", conn, if_exists="append", index=False)
        total += len(sub)
        print(f"    … {total:,} rows", end="\r", flush=True)
    conn.commit()
    elapsed = time.time() - t0
    print(f"  [✓] Imported {total:,} lab rows  ({elapsed:.1f}s)          ")
    return total


def print_stats(conn: sqlite3.Connection) -> None:
    """Print final row counts for all clinical tables."""
    print("\n  Clinical Table Statistics:")
    print("  " + "-" * 42)
    for table in ("patients", "admissions", "diagnoses", "labs"):
        try:
            row = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
            print(f"    {table:<14}: {row['cnt']:>10,} rows")
        except Exception:
            print(f"    {table:<14}: (not available)")
    print("  " + "-" * 42)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    skip_labs   = "--no-labs"      in sys.argv
    lab_sample  = LAB_SAMPLE_SIZE

    # Allow custom lab sample via --labs-sample <n>
    if "--labs-sample" in sys.argv:
        idx = sys.argv.index("--labs-sample")
        if idx + 1 < len(sys.argv):
            try:
                lab_sample = int(sys.argv[idx + 1])
            except ValueError:
                pass

    print("\n  Hypotify – Week 9 Patient Data Import")
    print("  " + "=" * 48)
    print(f"  Database  : {DB_PATH}")
    print(f"  Data dir  : {DATA_DIR}")
    print(f"  Lab sample: {'skipped' if skip_labs else f'{lab_sample:,} rows'}")
    print()

    conn = get_connection()
    try:
        # Step 1: Create clinical tables
        create_clinical_tables(conn)

        # Step 2: Import patients (must be first — other tables FK to it)
        import_patients(conn)

        # Step 3: Import admissions
        import_admissions(conn)

        # Step 4: Import diagnoses
        import_diagnoses(conn)

        # Step 5: Import labs (sample)
        if not skip_labs:
            import_labs(conn, sample_size=lab_sample)
        else:
            print("  [–] Labs import skipped (--no-labs flag).")

        # Step 6: Build fast-lookup indexes
        print("  Building indexes …", end=" ", flush=True)
        create_indexes(conn)

        # Summary
        print_stats(conn)
        print("\n  [✓] Patient import complete.\n")

    except Exception as exc:
        print(f"\n  [ERROR] Import failed: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
