"""Completely rewrites make_pdf.py to use a reliable Latin-1 sanitizer."""

import re
from pathlib import Path
from fpdf import FPDF

HERE     = Path(__file__).parent
TXT_FILE = HERE / "Week10_Deploy_Paper_ShrutiMalik.txt"
PDF_FILE = HERE / "Week10_Deploy_Paper_ShrutiMalik.pdf"

# ── Colours ────────────────────────────────────────────────────────────────────
NAVY       = (13,  71, 161)
ACCENT     = (30, 136, 229)
MID_GRAY   = (80,  80,  80)
DARK_GRAY  = (50,  50,  50)
LIGHT_GRAY = (130,130, 130)
WHITE      = (255,255, 255)


# ── Unicode sanitizer ──────────────────────────────────────────────────────────
_UNICODE_MAP = [
    ("\u2192", "->"),   ("\u2190", "<-"),
    ("\u2014", "--"),   ("\u2013", "-"),
    ("\u2018", "'"),    ("\u2019", "'"),
    ("\u201c", '"'),    ("\u201d", '"'),
    ("\u2026", "..."),
    ("\u2022", "*"),    ("\u25cf", "*"),
    ("\u2500", "-"),    ("\u2501", "="),    ("\u2502", "|"),   ("\u2550", "="),
    ("\u2212", "-"),
    ("\u2265", ">="),   ("\u2264", "<="),
    ("\u00b0", "deg"),  ("\u00b1", "+/-"),  ("\u00d7", "x"),
    ("\u03b1", "alpha"),("\u03b2", "beta"),
    ("\u2713", "OK"),   ("\u2717", "X"),
    ("\u2610", "[ ]"),  ("\u2611", "[x]"),
    # Emoji
    ("\U0001F60A", "(positive)"),
    ("\U0001F61F", "(negative)"),
    ("\U0001F610", "(neutral)"),
    ("\U0001F3E5", "[Hospital]"),
]

def sanitize(text: str) -> str:
    for src, dst in _UNICODE_MAP:
        text = text.replace(src, dst)
    # Final pass: drop anything not encodable in Latin-1
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# ── Regex helpers ──────────────────────────────────────────────────────────────
_RULE_RE    = re.compile(r"^={10,}\s*$")
_DASH_RE    = re.compile(r"^[-]{10,}\s*$")
_SECTION_RE = re.compile(r"^\d+\.\s+\S")
_CAT_RE     = re.compile(r"^CATEGORY\s+\d+")
_QA_Q_RE    = re.compile(r"^Q\d+:")
_QA_A_RE    = re.compile(r"^A:")
_INDENT_RE  = re.compile(r"^\s{4,}")


class PaperPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(0, 8, "Hypotify Clinical Chatbot - Week 10 | Shruti Malik",
                  align="L", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-16)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*LIGHT_GRAY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def hline(self, color=ACCENT, lw=0.3):
        self.set_draw_color(*color)
        self.set_line_width(lw)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def para(self, text, indent=0):
        if indent:
            self.set_x(self.l_margin + indent)
        self.multi_cell(self.w - self.l_margin - self.r_margin - indent,
                        5, text, align="L", new_x="LMARGIN", new_y="NEXT")


def add_cover(pdf: PaperPDF):
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.set_margins(22, 22, 22)

    # Navy top bar
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 55, style="F")

    pdf.set_y(14)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 10, "ITEC5025 - Week 10",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 9, "Deploying the Hypotify Clinical Chatbot",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "A Web-Based Interface, Automated Testing, and Retrospective",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(65)
    pdf.set_text_color(*DARK_GRAY)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Author: Shruti Malik",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Date: March 18, 2026",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Natural Language Processing in AI Chatbots",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.hline(ACCENT, 0.6)

    # Summary box
    pdf.set_fill_color(240, 245, 255)
    pdf.rect(22, pdf.get_y(), 166, 46, style="F")
    pdf.set_y(pdf.get_y() + 4)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 6, "  Project Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MID_GRAY)
    lines = [
        "  The Hypotify Clinical Chatbot integrates a Bidirectional LSTM intent classifier,",
        "  SQLite relational database (100,000 patients, 361,760 admissions/diagnoses,",
        "  100,000 lab results), VADER sentiment analysis, spaCy Named Entity Recognition,",
        "  and Flask/Streamlit web deployments. This paper documents the complete arc",
        "  from Week 6 through Week 10, with 30 automated test results and 100 Q&As.",
    ]
    for ln in lines:
        pdf.cell(0, 5, ln, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(16)

    # Stats row
    stats = [
        ("100,000", "Patients"),
        ("361,760", "Admissions"),
        ("361,760", "Diagnoses"),
        ("30",      "Tests"),
        ("100",     "Q&A Pairs"),
    ]
    col_w = 166 / len(stats)
    for pass_no in range(2):
        pdf.set_x(22)
        for val, lbl in stats:
            if pass_no == 0:
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(*NAVY)
                pdf.cell(col_w, 8, val, align="C")
            else:
                pdf.set_font("Helvetica", "", 7)
                pdf.set_text_color(*LIGHT_GRAY)
                pdf.cell(col_w, 5, lbl, align="C")
        pdf.ln(8 if pass_no == 0 else 5)


def build_pdf():
    raw = TXT_FILE.read_text(encoding="utf-8", errors="replace")
    raw = sanitize(raw)
    lines = raw.splitlines()

    pdf = PaperPDF()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.set_margins(22, 22, 22)

    add_cover(pdf)
    pdf.add_page()

    skip_header_lines = 16   # skip the plain-text title header block
    i = 0

    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()
        i += 1

        # Skip the decorative plain-text header at top of file
        if i <= skip_header_lines:
            continue

        # === dividers
        if _RULE_RE.match(line):
            pdf.ln(1)
            pdf.hline(NAVY, 0.5)
            continue

        # --- dividers
        if _DASH_RE.match(line):
            pdf.ln(1)
            pdf.hline((180, 200, 230), 0.2)
            continue

        # Long dash-line separators (box-drawing, already converted to ---)
        if re.match(r"^-{10,}\s*$", line):
            pdf.ln(1)
            pdf.hline((180, 200, 230), 0.2)
            continue

        # All-caps decorative banner lines that repeat the title
        if line and line.isupper() and len(line) > 40:
            continue

        # Numbered section headings "1. Deployment Method..."
        if _SECTION_RE.match(line):
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*NAVY)
            pdf.set_fill_color(235, 241, 255)
            pdf.cell(0, 8, "  " + line, align="L", fill=True,
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*DARK_GRAY)
            pdf.ln(2)
            continue

        # CATEGORY headings
        if _CAT_RE.match(line):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*ACCENT)
            pdf.para(line)
            pdf.set_text_color(*DARK_GRAY)
            pdf.ln(1)
            continue

        # Q&A question
        if _QA_Q_RE.match(line):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.para(line)
            pdf.set_text_color(*DARK_GRAY)
            continue

        # Q&A answer
        if _QA_A_RE.match(line):
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*MID_GRAY)
            pdf.para(line, indent=5)
            pdf.set_text_color(*DARK_GRAY)
            continue

        # Indented code-style lines
        if _INDENT_RE.match(raw_line):
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(*MID_GRAY)
            pdf.para(line, indent=8)
            pdf.set_text_color(*DARK_GRAY)
            continue

        # Blank line
        if not line:
            pdf.ln(2)
            continue

        # Default body text
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*DARK_GRAY)
        pdf.para(line)

    pdf.output(str(PDF_FILE))
    print(f"PDF saved: {PDF_FILE}")
    print(f"Total pages: {pdf.page}")


if __name__ == "__main__":
    build_pdf()
