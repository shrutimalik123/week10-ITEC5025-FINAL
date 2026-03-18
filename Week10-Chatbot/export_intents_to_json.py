"""
export_intents_to_json.py
=========================
Week 9 Assignment – Database Integration for Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-09

Purpose:
    Export the intents table from hypotify.db into intents.json.

    This fulfils the Week 9 requirement:
        "Run the Python script chatbot to see updated responses from new SQL
         data that was imported into the JSON file."

    The exported JSON groups all patterns and responses by intent tag:
    {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["hello", "hi", ...],
                "responses": ["Hello! ...", "Hi there! ...", ...]
            },
            ...
        ]
    }

Usage:
    python export_intents_to_json.py
    python export_intents_to_json.py --output custom_name.json
"""

import json
import os
import sqlite3
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(SCRIPT_DIR, "hypotify.db")
DEFAULT_JSON = os.path.join(SCRIPT_DIR, "intents.json")


def export_intents(output_path: str = DEFAULT_JSON) -> int:
    """
    Read all rows from the intents table and write them to a JSON file.

    Groups rows by tag so each intent object contains unique patterns
    and unique responses (duplicates are preserved – mirrors dataset.csv).

    Args:
        output_path (str): Destination path for the JSON file.

    Returns:
        int: Number of unique intent tags exported.

    Raises:
        FileNotFoundError: if hypotify.db does not exist.
        sqlite3.Error:     on query failure.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run  python db_setup.py  first."
        )

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT tag, pattern, response FROM intents ORDER BY tag, id"
    ).fetchall()
    conn.close()

    # ── Group by tag ───────────────────────────────────────────────────────────
    intent_map: dict[str, dict] = {}
    for row in rows:
        tag = row["tag"]
        if tag not in intent_map:
            intent_map[tag] = {"tag": tag, "patterns": [], "responses": []}
        intent_map[tag]["patterns"].append(row["pattern"])
        intent_map[tag]["responses"].append(row["response"])

    # ── Build output structure ─────────────────────────────────────────────────
    output = {"intents": list(intent_map.values())}

    # ── Write JSON ─────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return len(intent_map)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    # Allow custom output path via --output flag
    output_path = DEFAULT_JSON
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]

    print("\n  Hypotify – Export Intents DB → JSON")
    print("  " + "=" * 40)
    print(f"  Source  : {DB_PATH}")
    print(f"  Output  : {output_path}")
    print()

    try:
        num_tags = export_intents(output_path)
        print(f"  [✓] Exported {num_tags} intent tags to intents.json")
        print(f"  File size: {os.path.getsize(output_path):,} bytes\n")
    except FileNotFoundError as exc:
        print(f"  [ERROR] {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"  [ERROR] Export failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
