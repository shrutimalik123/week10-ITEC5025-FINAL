# Week 10 – Deploying and Testing the Hypotify Clinical Chatbot

**Author:** Shruti Malik  
**Course:** ITEC5025 – Natural Language Processing in AI Chatbots  
**Date:** March 18, 2026

> **Note:** All Week 8 and Week 9 files have been consolidated into this folder.
> This directory is fully self-contained — no other folders are required to run.

---

## What's in This Folder

| File / Folder | Description |
|---|---|
| `app.py` | Flask web server — `/chat`, `/health`, `/stats`, `/history` endpoints |
| `streamlit_app.py` | Streamlit UI — one-command deployment, dark theme, sidebar commands |
| `chatbot_w9.py` | `HypotifyChatbot` engine — BiLSTM + DB + sentiment + NER |
| `db_manager.py` | SQLite CRUD layer — patient, admission, diagnoses, labs, logs |
| `hypotify.db` | SQLite database (~137 MB) — 100K patients, 361K admissions/diagnoses, 100K labs |
| `chatbot_model_w8.keras` | Trained Bidirectional LSTM model |
| `tokenizer_w8.pkl` | Keras tokenizer (fitted on training data) |
| `label_encoder_w8.pkl` | Scikit-learn LabelEncoder for 20 intent classes |
| `preprocess.py` | NLP preprocessing — tokenise, lemmatise, clean |
| `train.py` | Model training script (re-run to retrain) |
| `model.py` | Keras model architecture definition |
| `dataset.csv` | Training data: 370 labeled intent examples |
| `intents.json` | Intent knowledge base exported from DB |
| `db_setup.py` | Creates and seeds `hypotify.db` |
| `import_patients.py` | Imports patient/admissions/diagnoses/labs CSVs into DB |
| `templates/index.html` | Flask chat UI (dark glassmorphism, sidebar, live DB stats) |
| `test_chatbot_w10.py` | 30-test automated suite |
| `top100_qa.txt` | 100 probable questions and expected chatbot answers |
| `Week10_Deploy_Paper_ShrutiMalik.txt` | Full academic paper |
| `Week10_Deploy_Paper_ShrutiMalik.pdf` | PDF version of the paper |
| `requirements_w10.txt` | Python dependencies |
| `make_pdf.py` | Script to regenerate the PDF from the .txt paper |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements_w10.txt
```

> Core ML/NLP libraries (tensorflow, nltk, spacy, pandas, scikit-learn) must also be installed.
> The full list is in `requirements_w10.txt`.

### 2a. Run with Streamlit (recommended)

```bash
streamlit run streamlit_app.py
```

Opens automatically at **http://localhost:8501**

### 2b. Run with Flask

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 3. Run the test suite

```bash
python test_chatbot_w10.py -v
```

---

## REST API Reference (Flask)

| Endpoint | Method | Description | Response |
|---|---|---|---|
| `/` | GET | Serve the Chat UI | HTML |
| `/chat` | POST | Send a message, get bot response | `{"response": "…", "intent": "…", "session": "…"}` |
| `/health` | GET | Model + DB health check | `{"status": "ok", "model": "loaded", …}` |
| `/stats` | GET | Live DB table row counts | `{"intents": 386, "patients": 100000, …}` |
| `/history` | GET | Last 20 conversation turns | `[{"user_message": …, "bot_response": …}, …]` |

---

## Chatbot Commands

| Command | Example | Description |
|---|---|---|
| `patient <uuid>` | `patient F7CF0FE9-AFCD-49EF-BFB3-E42302FFA0D3` | Look up a patient record |
| `admissions <uuid>` | `admissions F7CF0FE9-…` | Patient hospital admission history |
| `diagnoses <uuid>` | `diagnoses F7CF0FE9-…` | ICD-10 diagnoses |
| `labs <uuid>` | `labs F7CF0FE9-…` | Lab test results |
| `insights <category>` | `insights gender` | Population statistics (gender/age/race/language/poverty/marital) |
| `sentiment <text>` | `sentiment I feel great` | VADER sentiment analysis |
| `ner <text>` | `ner Dr. Smith 2025-01-15` | Named entity recognition (spaCy) |
| `search <keyword>` | `search glucose` | Search knowledge base |
| `feedback <1-5> <text>` | `feedback 5 excellent` | Submit star rating |
| `db info` | `db info` | Show DB table row counts |
| `help` | `help` | List all commands |
| `exit` / `bye` | `goodbye` | End the conversation |

---

## Test Suite

```
TestNLP               –  8 tests  : preprocess, sentiment, NER
TestDatabase          –  9 tests  : DB connection, CRUD, feedback, log
TestChatbotEngine     – 14 tests  : greeting, help, commands, context window, E2E
TestFlaskAPI          – 10 tests  : /health, /stats, /, /chat, /history, error paths
TestE2EConversation   –  1 test   : 10-turn clinical conversation with timing
─────────────────────────────────────────────────────
Total                 : 30 tests
```

---

## Database

`hypotify.db` contains 8 tables:

| Table | Rows | Description |
|---|---|---|
| `intents` | 386 | Chatbot knowledge base (tags, patterns, responses) |
| `patients_summary` | 51 | Pre-computed population summary statistics |
| `patients` | 100,000 | Patient records (UUID, gender, DOB, race, language, poverty %) |
| `admissions` | 361,760 | Hospital admission records |
| `diagnoses` | 361,760 | ICD-10 diagnosis records |
| `labs` | 100,000 | Lab result rows (sampled from 1M+ source rows) |
| `conversation_log` | (grows) | Every chat turn logged with session ID and timestamp |
| `user_feedback` | (grows) | Star ratings and comments |

---

## Paper

The academic paper (`Week10_Deploy_Paper_ShrutiMalik.pdf`) covers:

- Project planning arc from Week 6 to Week 10
- Flask and Streamlit deployment rationale
- Layered architecture integration (7 layers)
- Testing strategy and results (30 tests, per-turn timing)
- 100 probable user questions and expected chatbot answers
- Identified issues and proposed improvements
- Reflection on systems thinking for AI development
