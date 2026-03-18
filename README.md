# Hypotify Clinical Chatbot — ITEC5025 Final Project

**Author:** Shruti Malik  
**Course:** ITEC5025 – Natural Language Processing in AI Chatbots  
**Submitted:** March 2026

---

## About

The **Hypotify Clinical Chatbot** is the capstone project for ITEC5025. It is a
web-deployable AI assistant that can look up patient records, compute population
statistics, perform sentiment analysis, extract named entities, and more — all
backed by a SQLite database of 100,000 synthetic clinical patient records.

The project was developed across Weeks 6–10 of the course, integrating:

| Layer | Technology |
|---|---|
| Intent classification | Bidirectional LSTM (TensorFlow / Keras) |
| NLP preprocessing | NLTK tokenization, WordNet lemmatization |
| Sentiment analysis | VADER (NLTK) |
| Named entity recognition | spaCy `en_core_web_sm` |
| Database | SQLite (`hypotify.db`) — 100K patients, 361K admissions/diagnoses, 100K lab rows |
| Web deployment | Flask REST API + Streamlit UI |
| Testing | 30-test automated suite |

---

## Repository Structure

```
week10-ITEC5025-FINAL/
├── Week10-Chatbot/          ← All source code, model, and database (self-contained)
│   ├── app.py               # Flask web server
│   ├── streamlit_app.py     # Streamlit web UI
│   ├── chatbot_w9.py        # HypotifyChatbot engine (BiLSTM + DB)
│   ├── db_manager.py        # SQLite CRUD layer
│   ├── hypotify.db.part1    # Database part 1 (<100MB)
│   ├── hypotify.db.part2    # Database part 2 (<100MB)
│   ├── chatbot_model_w8.keras  # Trained Bidirectional LSTM model
│   ├── tokenizer_w8.pkl     # Keras tokenizer
│   ├── label_encoder_w8.pkl # Intent label encoder
│   ├── preprocess.py        # NLP preprocessing pipeline
│   ├── train.py             # Model training script
│   ├── dataset.csv          # Training data (370 labeled intent examples)
│   ├── intents.json         # Intent knowledge base (exported from DB)
│   ├── templates/
│   │   └── index.html       # Flask chat UI (dark glassmorphism)
│   ├── test_chatbot_w10.py  # 30-test automated suite
│   ├── top100_qa.txt        # 100 probable Q&A reference pairs
│   ├── Week10_Deploy_Paper_ShrutiMalik.txt  # Academic paper
│   ├── Week10_Deploy_Paper_ShrutiMalik.pdf  # PDF version of paper
│   └── requirements_w10.txt # Python dependencies
├── .gitignore
├── LICENSE
└── README.md                ← This file
```

---

## Quick Start

```bash
cd Week10-Chatbot

# Install dependencies
pip install -r requirements_w10.txt

# Option A – Streamlit (recommended)
streamlit run streamlit_app.py
# Opens automatically at http://localhost:8501

# Option B – Flask
python app.py
# Open http://127.0.0.1:5000

# Run the test suite
python test_chatbot_w10.py -v
```

---

## Key Features

- **Patient records** — `patient <uuid>`, `admissions <uuid>`, `diagnoses <uuid>`, `labs <uuid>`
- **Population insights** — `insights gender/age/race/language/poverty/marital`
- **Sentiment analysis** — `sentiment <text>` (VADER, score -1.0 to +1.0)
- **Named entity recognition** — `ner <text>` (spaCy: PERSON, DATE, ORG, GPE, MONEY)
- **Knowledge base search** — `search <keyword>`
- **Feedback** — `feedback <1-5> <comment>`
- **DB info** — `db info`

---

## Academic Paper

The full development retrospective (Weeks 6–10), including 100 Q&A pairs, is documented in:

- [`Week10_Deploy_Paper_ShrutiMalik.pdf`](Week10-Chatbot/Week10_Deploy_Paper_ShrutiMalik.pdf)
- [`Week10_Deploy_Paper_ShrutiMalik.txt`](Week10-Chatbot/Week10_Deploy_Paper_ShrutiMalik.txt)
