"""
test_top100_live.py
===================
Week 10 Assignment – Testing the Live Chatbot with Top 100 Questions
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-19

Purpose:
    Parses all 100 Q&A pairs from top100_qa.txt, sends each question through
    the HypotifyChatbot engine (same engine as the live Streamlit deployment),
    compares the actual output against the expected answer using keyword/fuzzy
    matching, and writes a detailed results report.

    The local chatbot_w9.py engine IS the live bot — Streamlit Cloud simply
    calls HypotifyChatbot.respond() in the same way this script does.

Usage:
    cd Week10-Chatbot
    python test_top100_live.py
    python test_top100_live.py --verbose      # print each Q/A during run
    python test_top100_live.py --report-only  # just print the saved report
"""

import os
import re
import sys
import time
import argparse
import textwrap
from datetime import datetime

# ── Force UTF-8 output on Windows ─────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Source files ───────────────────────────────────────────────────────────────
TOP100_FILE = os.path.join(SCRIPT_DIR, "top100_qa.txt")
REPORT_FILE = os.path.join(SCRIPT_DIR, "top100_test_results.txt")

# ── Thresholds ─────────────────────────────────────────────────────────────────
# A test is PASS if ≥ this fraction of expected keyword tokens appear in actual
KEYWORD_MATCH_THRESHOLD = 0.3   # 30 % of expected keywords must appear


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 – Parse top100_qa.txt into a list of (question, expected_answer) tuples
# ══════════════════════════════════════════════════════════════════════════════

def parse_top100(path: str) -> list[tuple[str, str]]:
    """
    Parse top100_qa.txt into a list of (question, expected_answer) tuples.

    The file format uses lines like:
        Q1:  Hello
        A:   Hello! Welcome to …

    Multi-line answers continue until the next Q or a separator line.
    """
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    pairs: list[tuple[str, str]] = []
    current_q: str | None = None
    current_a_lines: list[str] = []

    q_re = re.compile(r"^Q\d+[:\.]?\s+(.+)", re.IGNORECASE)
    a_re = re.compile(r"^A[:\.]?\s*(.*)", re.IGNORECASE)

    def flush():
        if current_q is not None and current_a_lines:
            answer = " ".join(current_a_lines).strip()
            pairs.append((current_q.strip(), answer))

    state = "seeking_q"

    for raw in lines:
        line = raw.rstrip("\r\n")
        stripped = line.strip()

        # Skip decorative separators and headers
        if re.match(r"^[=─\-]+$", stripped) or not stripped:
            continue
        if stripped.startswith("TOP 100") or stripped.startswith("CATEGORY"):
            continue
        if stripped.startswith("The following") or stripped.startswith("Author"):
            continue
        if stripped.startswith("Hypotify") and "Chatbot" in stripped:
            continue
        if stripped.startswith("END OF"):
            break

        q_match = q_re.match(stripped)
        a_match = a_re.match(stripped)

        if q_match:
            flush()
            current_q = q_match.group(1).strip()
            current_a_lines = []
            state = "seeking_a"

        elif state == "seeking_a" and a_match:
            # First answer line
            first_line = a_match.group(1).strip()
            current_a_lines = [first_line] if first_line else []
            state = "in_answer"

        elif state == "in_answer":
            # Continuation of answer (indented or just text)
            if not q_re.match(stripped) and not stripped.startswith("────"):
                current_a_lines.append(stripped)

    flush()
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 – Load the chatbot engine
# ══════════════════════════════════════════════════════════════════════════════

def load_chatbot():
    """Import and return a HypotifyChatbot instance."""
    try:
        from chatbot_w9 import HypotifyChatbot  # type: ignore
        print("  Loading HypotifyChatbot engine … (may take 10–30 s for model)")
        bot = HypotifyChatbot()
        print("  ✓  Engine loaded.\n")
        return bot
    except ImportError as exc:
        print(f"  ✗  Cannot import chatbot_w9: {exc}")
        sys.exit(1)
    except SystemExit:
        print("  ✗  Chatbot initialisation failed (missing model artefacts).")
        sys.exit(1)
    except Exception as exc:
        print(f"  ✗  Unexpected error: {exc}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 – Keyword-match comparison
# ══════════════════════════════════════════════════════════════════════════════

# Words that are not meaningful for matching
STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "is",
    "are", "it", "on", "with", "this", "i", "my", "me", "you", "be",
    "by", "at", "as", "do", "so", "if", "was", "will", "can", "from",
    "that", "have", "not", "your", "type", "use", "all", "any",
}

def _keywords(text: str) -> set[str]:
    """Extract meaningful lowercase tokens from text."""
    tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    return {t for t in tokens if t not in STOP and len(t) > 1}


def compare(expected: str, actual: str) -> tuple[bool, float, str]:
    """
    Compare expected and actual responses.

    Returns:
        (pass_bool, match_ratio, reason_string)

    Strategy:
        1. If expected contains [Returns …] / [Same as …] bracket notes → SKIP
           (these are meta-references, not literal expected text)
        2. Keyword intersection: count how many expected keywords appear in actual.
        3. PASS if ratio ≥ KEYWORD_MATCH_THRESHOLD or actual is non-empty response
           and expected is a meta-placeholder.
    """
    exp_lower = expected.lower()

    # Meta-placeholders → just check that actual is non-empty
    if re.search(r"\[returns|same as|see above\]", exp_lower):
        passed = bool(actual and len(actual.strip()) > 0)
        return passed, 1.0 if passed else 0.0, "meta-placeholder — checking non-empty response"

    exp_kw = _keywords(expected)
    act_kw = _keywords(actual)

    if not exp_kw:
        passed = bool(actual and len(actual.strip()) > 0)
        return passed, 1.0 if passed else 0.0, "no expected keywords — checking non-empty response"

    matched = exp_kw & act_kw
    ratio = len(matched) / len(exp_kw)
    passed = ratio >= KEYWORD_MATCH_THRESHOLD

    matched_str = ", ".join(sorted(matched)[:8])  # show up to 8 matched keywords
    missing = exp_kw - act_kw
    missing_str = ", ".join(sorted(missing)[:5])

    reason = (
        f"matched {len(matched)}/{len(exp_kw)} keywords ({ratio:.0%}) | "
        f"found: [{matched_str}] | "
        f"missing: [{missing_str}]"
    )
    return passed, ratio, reason


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 – Run the test
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(pairs: list[tuple[str, str]], bot, verbose: bool = False) -> list[dict]:
    """
    Send each question to the chatbot engine, compare with expected, and
    return a list of result dicts.
    """
    results = []

    print(f"  Running {len(pairs)} questions …\n")
    print(f"  {'#':>3}  {'STATUS':<6}  {'MATCH':>5}  QUESTION")
    print("  " + "─" * 70)

    for i, (question, expected) in enumerate(pairs, start=1):
        t0 = time.time()
        try:
            actual = bot.respond(question)
        except Exception as exc:
            actual = f"[ERROR: {exc}]"
        elapsed_ms = (time.time() - t0) * 1000

        passed, ratio, reason = compare(expected, actual)
        status = "PASS ✓" if passed else "FAIL ✗"
        short_q = question[:50] + ("…" if len(question) > 50 else "")

        print(f"  {i:>3}  {status:<6}  {ratio:>4.0%}  {short_q}")

        if verbose:
            print(f"       Q: {question}")
            print(f"       E: {textwrap.shorten(expected, 100)}")
            print(f"       A: {textwrap.shorten(actual,   100)}")
            print(f"       reason: {reason}")
            print()

        results.append({
            "n":         i,
            "question":  question,
            "expected":  expected,
            "actual":    actual,
            "passed":    passed,
            "ratio":     ratio,
            "reason":    reason,
            "ms":        elapsed_ms,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 – Generate the report
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(results: list[dict]) -> str:
    """Build and return the full text report."""
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    failed  = total - passed
    avg_ms  = sum(r["ms"] for r in results) / total if total else 0
    avg_ratio = sum(r["ratio"] for r in results) / total if total else 0

    lines = []
    lines.append("=" * 80)
    lines.append("  HYPOTIFY CLINICAL CHATBOT — TOP 100 QUESTION TEST REPORT")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Live URL: https://week10-itec5025-final-54k7qmgyfdwbwzwwvmod3j.streamlit.app/")
    lines.append(f"  Engine:   HypotifyChatbot (chatbot_w9.py) — same as live deployment")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  Total questions   : {total}")
    lines.append(f"  PASS              : {passed}  ({passed/total:.1%})")
    lines.append(f"  FAIL              : {failed}  ({failed/total:.1%})")
    lines.append(f"  Avg keyword match : {avg_ratio:.1%}")
    lines.append(f"  Avg response time : {avg_ms:.0f} ms")
    lines.append(f"  Match threshold   : {KEYWORD_MATCH_THRESHOLD:.0%} keyword overlap = PASS")
    lines.append("")
    lines.append("─" * 80)
    lines.append("  DETAILED RESULTS")
    lines.append("─" * 80)
    lines.append("")

    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        lines.append(f"Q{r['n']:>3}  [{status}]  Match: {r['ratio']:.0%}  ({r['ms']:.0f} ms)")
        lines.append(f"  Question : {r['question']}")
        lines.append(f"  Expected : {textwrap.shorten(r['expected'], width=120)}")
        lines.append(f"  Actual   : {textwrap.shorten(r['actual'],   width=120)}")
        lines.append(f"  Detail   : {r['reason']}")
        lines.append("")

    lines.append("─" * 80)
    lines.append("  FAILURES SUMMARY")
    lines.append("─" * 80)
    failures = [r for r in results if not r["passed"]]
    if not failures:
        lines.append("  🎉  All 100 questions PASSED!")
    else:
        for r in failures:
            lines.append(f"  Q{r['n']:>3}: {r['question']}")
            lines.append(f"        Expected  : {textwrap.shorten(r['expected'], 100)}")
            lines.append(f"        Actual    : {textwrap.shorten(r['actual'],   100)}")
            lines.append(f"        Match     : {r['ratio']:.0%}  — {r['reason']}")
            lines.append("")

    lines.append("=" * 80)
    lines.append(f"  FINAL RESULT: {'ALL PASS ✓' if failed == 0 else f'{passed}/{total} PASS ({passed/total:.1%})'}")
    lines.append("=" * 80)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test the Hypotify chatbot with top 100 Q&A pairs.")
    parser.add_argument("--verbose",     action="store_true", help="Print Q/A pairs during run")
    parser.add_argument("--report-only", action="store_true", help="Print existing report and exit")
    args = parser.parse_args()

    print("=" * 70)
    print("  Hypotify – Top 100 Question Live Test")
    print("  ITEC5025 | Author: Shruti Malik | 2026-03-19")
    print("=" * 70)
    print()

    if args.report_only:
        if os.path.exists(REPORT_FILE):
            with open(REPORT_FILE, encoding="utf-8") as f:
                print(f.read())
        else:
            print(f"  No report file found at: {REPORT_FILE}")
            print("  Run without --report-only to generate one.")
        return

    # 1. Parse Q&A pairs
    print(f"  Parsing {TOP100_FILE} …")
    pairs = parse_top100(TOP100_FILE)
    print(f"  ✓  Parsed {len(pairs)} question/answer pairs.\n")

    if len(pairs) < 90:
        print(f"  ⚠️  WARNING: Only {len(pairs)} pairs found; expected ~100. Check the parser.")

    # 2. Load chatbot
    bot = load_chatbot()

    # 3. Run tests
    results = run_tests(pairs, bot, verbose=args.verbose)

    # 4. Generate report
    print()
    report = generate_report(results)

    # 5. Save report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    # 6. Print summary
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print()
    print("=" * 70)
    print(f"  Tests run    : {total}")
    print(f"  PASS         : {passed}  ({passed/total:.1%})")
    print(f"  FAIL         : {failed}  ({failed/total:.1%})")
    print(f"  Result       : {'ALL PASS ✓' if failed == 0 else f'{passed}/{total} passed'}")
    print(f"  Report saved : {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
