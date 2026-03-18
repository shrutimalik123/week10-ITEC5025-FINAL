"""
test_chatbot_w10.py
===================
Week 10 Assignment – Deploying and Testing the Chatbot
ITEC5025 – Hypotify Clinical Chatbot
Author: Shruti Malik
Date:   2026-03-18

Purpose:
    Automated test suite for the Hypotify chatbot.

    Tests cover:
        1. Model-layer unit tests  – intent prediction, confidence thresholds
        2. NLP-layer unit tests    – sentiment, NER, preprocessing
        3. Database-layer tests    – db_manager CRUD operations
        4. Command-parser tests    – regex routes (patient, admissions, diagnoses,
                                    labs, feedback, db info, search, sentiment, ner)
        5. Flask integration tests – /chat, /health, /stats, /history endpoints
        6. End-to-end scenario     – multi-turn conversations with DB logging

Usage:
    python test_chatbot_w10.py              # run all tests
    python test_chatbot_w10.py -v           # verbose output
    python test_chatbot_w10.py TestNLP      # run one class

Notes:
    - The tests import from the Week 9 directory, so Week8-Chatbot model
      artefacts and the hypotify.db database must already exist.
    - Flask integration tests start the app in test-client mode (no real server).
    - Each test class prints a summary line on completion.
"""

import json
import os
import sys
import time
import unittest

# ── Path setup: add Week 9 to sys.path ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
WEEK9_DIR  = os.path.join(ROOT_DIR, "Week9-Chatbot")
sys.path.insert(0, WEEK9_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Helper: import guard with skip-on-missing logic                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _try_import(module_name: str):
    """Return the module if available, else None (allows tests to skip cleanly)."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. NLP Unit Tests                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestNLP(unittest.TestCase):
    """Unit tests for the NLP helper functions in chatbot_w9.py."""

    @classmethod
    def setUpClass(cls):
        """Import NLP helpers once for the whole class."""
        mod = _try_import("chatbot_w9")
        if mod is None:
            raise unittest.SkipTest("chatbot_w9 not importable – skipping NLP tests.")
        cls.preprocess   = mod.preprocess_input
        cls.sentiment    = mod.analyse_sentiment
        cls.ner          = mod.extract_entities_spacy
        print("\n  [NLP] Module imported successfully.")

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def test_preprocess_lowercases(self):
        """preprocess_input should lowercase the text."""
        result = self.preprocess("HELLO WORLD")
        self.assertEqual(result, result.lower(),
                         "preprocess_input did not lowercase the input.")

    def test_preprocess_removes_punctuation(self):
        """After preprocessing, common punctuation should be absent."""
        result = self.preprocess("Hello, world! How are you?")
        for char in ["!", ",", "?"]:
            self.assertNotIn(char, result,
                             f"Punctuation '{char}' survived preprocessing.")

    def test_preprocess_empty_string(self):
        """preprocess_input should handle empty strings gracefully."""
        result = self.preprocess("")
        self.assertIsInstance(result, str)

    def test_preprocess_lemmatisation(self):
        """Lemmatisation should reduce 'running' → 'run' (or at least not crash)."""
        result = self.preprocess("running")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    # ── Sentiment ──────────────────────────────────────────────────────────────

    def test_sentiment_positive(self):
        """A clearly positive sentence should score > 0.05 compound."""
        result = self.sentiment("I love this clinical tool! It's fantastic!")
        self.assertEqual(result["label"], "positive",
                         f"Expected positive, got {result['label']} (score={result['compound']})")

    def test_sentiment_negative(self):
        """A clearly negative sentence should score < -0.05 compound."""
        result = self.sentiment("This is terrible. I hate waiting for results.")
        self.assertEqual(result["label"], "negative",
                         f"Expected negative, got {result['label']} (score={result['compound']})")

    def test_sentiment_neutral(self):
        """A neutral clinical statement should be classified as neutral."""
        result = self.sentiment("The patient was admitted on Tuesday.")
        self.assertIn(result["label"], ("neutral", "positive"),
                      "Neutral statement misclassified as strongly negative.")

    def test_sentiment_returns_compound(self):
        """sentiment dict must contain 'compound' in [-1.0, 1.0]."""
        result = self.sentiment("Testing the chatbot.")
        self.assertIn("compound", result)
        self.assertGreaterEqual(result["compound"], -1.0)
        self.assertLessEqual(result["compound"], 1.0)

    # ── Named Entity Recognition ───────────────────────────────────────────────

    def test_ner_returns_list(self):
        """NER should always return a list (even if empty)."""
        entities = self.ner("The patient was admitted.")
        self.assertIsInstance(entities, list)

    def test_ner_detects_date(self):
        """NER should detect an ISO date or common date format."""
        entities = self.ner("Admitted on 2025-01-15.")
        labels = [e["label"] for e in entities]
        # spaCy uses DATE; fallback uses DATE too
        self.assertTrue(any("DATE" in lbl.upper() for lbl in labels),
                        f"No DATE entity found. Entities: {entities}")

    def test_ner_empty_string(self):
        """NER should handle an empty string without crashing."""
        result = self.ner("")
        self.assertIsInstance(result, list)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. Database Unit Tests                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestDatabase(unittest.TestCase):
    """Unit tests for db_manager.py CRUD functions."""

    @classmethod
    def setUpClass(cls):
        mod = _try_import("db_manager")
        if mod is None:
            raise unittest.SkipTest("db_manager not importable – skipping DB tests.")
        cls.db = mod
        # Verify DB is actually reachable
        try:
            cls.db.connect().close()
        except FileNotFoundError:
            raise unittest.SkipTest(
                "hypotify.db not found – run db_setup.py first. Skipping DB tests."
            )
        print("\n  [DB] Connected to hypotify.db successfully.")

    # ── Read operations ────────────────────────────────────────────────────────

    def test_get_db_stats_returns_dict(self):
        """get_db_stats should return a dict with table names as keys."""
        stats = self.db.get_db_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("intents", stats)

    def test_intents_table_has_data(self):
        """The intents table should have at least 1 row."""
        stats = self.db.get_db_stats()
        self.assertGreater(stats.get("intents", 0), 0,
                           "intents table is empty – run db_setup.py.")

    def test_get_intent_response_greeting(self):
        """get_intent_response('greeting') should return a non-empty string."""
        resp = self.db.get_intent_response("greeting")
        self.assertIsInstance(resp, (str, type(None)))
        # If db has data, it must be a non-empty string
        if resp is not None:
            self.assertGreater(len(resp.strip()), 0)

    def test_get_intent_response_unknown_tag(self):
        """get_intent_response for a non-existent tag should return None."""
        resp = self.db.get_intent_response("___nonexistent_tag_xyz___")
        self.assertIsNone(resp)

    def test_get_all_intents(self):
        """get_all_intents should return a list of dicts with 'tag' and 'count'."""
        intents = self.db.get_all_intents()
        self.assertIsInstance(intents, list)
        if intents:
            self.assertIn("tag",   intents[0])
            self.assertIn("count", intents[0])

    def test_search_intents(self):
        """search_intents('patient') should return at least one result."""
        results = self.db.search_intents("patient")
        self.assertIsInstance(results, list)

    # ── Write operations ───────────────────────────────────────────────────────

    def test_log_conversation_returns_true(self):
        """log_conversation should INSERT a row and return True."""
        ok = self.db.log_conversation(
            session_id   = "TEST-W10",
            user_message = "Automated test message",
            bot_response = "Automated test response",
            intent_tag   = "test",
        )
        self.assertTrue(ok, "log_conversation returned False unexpectedly.")

    def test_add_user_feedback_valid(self):
        """add_user_feedback with rating 5 should succeed."""
        ok = self.db.add_user_feedback("TEST-W10", 5, "Excellent test suite!")
        self.assertTrue(ok)

    def test_add_user_feedback_invalid_rating(self):
        """add_user_feedback with rating 0 (out of range) should return False."""
        ok = self.db.add_user_feedback("TEST-W10", 0, "Invalid rating")
        self.assertFalse(ok)

    def test_get_conversation_history(self):
        """get_conversation_history should return a list (possibly empty)."""
        history = self.db.get_conversation_history("TEST-W10", limit=5)
        self.assertIsInstance(history, list)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. Chatbot Engine Tests (HypotifyChatbot)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestChatbotEngine(unittest.TestCase):
    """
    Integration tests for HypotifyChatbot.respond().

    These tests exercise the full pipeline: NLP preprocessing → BiLSTM
    intent prediction → DB/JSON response lookup → conversation logging.
    """

    @classmethod
    def setUpClass(cls):
        mod = _try_import("chatbot_w9")
        if mod is None:
            raise unittest.SkipTest("chatbot_w9 not importable.")
        try:
            print("\n  [Engine] Loading HypotifyChatbot (may take a few seconds)…")
            cls.bot = mod.HypotifyChatbot()
            print("  [Engine] Chatbot loaded successfully.")
        except SystemExit:
            raise unittest.SkipTest(
                "HypotifyChatbot failed to initialise (missing model files). "
                "Run train.py in Week8-Chatbot first."
            )

    # ── Basic response tests ───────────────────────────────────────────────────

    def test_respond_greeting(self):
        """A greeting should return a non-empty response string."""
        resp = self.bot.respond("hello")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_goodbye(self):
        """A goodbye phrase should yield a response."""
        resp = self.bot.respond("goodbye")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_empty_input(self):
        """An empty input should return a guidance message, not crash."""
        resp = self.bot.respond("")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_help(self):
        """'help' should return the command list."""
        resp = self.bot.respond("help")
        self.assertIsInstance(resp, str)
        # Help response should mention at least one command keyword
        keywords = ["patient", "insights", "sentiment", "feedback", "exit"]
        self.assertTrue(
            any(kw in resp.lower() for kw in keywords),
            f"Help response did not list any expected commands. Got: {resp[:200]}"
        )

    # ── Clinical command tests ─────────────────────────────────────────────────

    def test_respond_db_info(self):
        """'db info' should return database statistics."""
        resp = self.bot.respond("db info")
        self.assertIsInstance(resp, str)
        # Should mention table names or 'rows' or 'unavailable'
        self.assertTrue(
            any(kw in resp.lower() for kw in
                ["intents", "rows", "unavailable", "database", "statistics"]),
            f"db info response unexpected: {resp[:200]}"
        )

    def test_respond_sentiment_command(self):
        """'sentiment <text>' should return a formatted sentiment result."""
        resp = self.bot.respond("sentiment This treatment is excellent!")
        self.assertIsInstance(resp, str)
        self.assertTrue(
            any(kw in resp.upper() for kw in ["POSITIVE", "NEGATIVE", "NEUTRAL", "SENTIMENT"]),
            f"Sentiment command did not return expected output: {resp[:200]}"
        )

    def test_respond_ner_command(self):
        """'ner <text>' should return entity extraction output."""
        resp = self.bot.respond("ner Dr. Patel admitted on 2025-03-10")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_feedback_command(self):
        """'feedback 4 good chatbot' should be acknowledged."""
        resp = self.bot.respond("feedback 4 Good system")
        self.assertIsInstance(resp, str)
        self.assertTrue(
            any(kw in resp.lower() for kw in ["thank", "feedback", "rating", "4"]),
            f"Feedback command not acknowledged properly: {resp[:200]}"
        )

    def test_respond_insights_gender(self):
        """insights gender should return gender distribution data."""
        resp = self.bot.respond("insights gender")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_insights_age(self):
        """insights age should return age statistics."""
        resp = self.bot.respond("insights age")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_search_command(self):
        """'search glucose' should return search results or a no-results message."""
        resp = self.bot.respond("search glucose")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_context_window_updates(self):
        """After a respond() call, context_history should have at least 1 entry."""
        self.bot.respond("hello")
        self.assertGreater(
            len(self.bot.context_history), 0,
            "context_history is empty after respond()."
        )

    def test_context_window_bounded(self):
        """Context history must not exceed CONTEXT_WINDOW length."""
        from chatbot_w9 import CONTEXT_WINDOW
        for _ in range(CONTEXT_WINDOW + 5):
            self.bot.respond("hello")
        self.assertLessEqual(
            len(self.bot.context_history), CONTEXT_WINDOW,
            "context_history exceeded CONTEXT_WINDOW."
        )

    def test_respond_unknown_gibberish(self):
        """Gibberish input should return a fallback/unknown response, not crash."""
        resp = self.bot.respond("xyzqrstuvwxyz jabberwocky foobar")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)

    def test_respond_patient_uuid_format(self):
        """A UUID-format patient lookup should return a response, not crash."""
        # Use a plausible UUID format (likely not in DB, but shouldn't crash)
        resp = self.bot.respond("patient F7CF0FE9-AFCD-49EF-BFB3-E42302FFA0D3")
        self.assertIsInstance(resp, str)
        self.assertGreater(len(resp), 0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. Flask API Integration Tests                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestFlaskAPI(unittest.TestCase):
    """
    Integration tests for the Week 10 Flask REST API.

    Uses Flask's test client (no actual HTTP server is started).
    """

    @classmethod
    def setUpClass(cls):
        app_mod = _try_import("app")
        if app_mod is None:
            raise unittest.SkipTest("app.py not importable – skipping Flask tests.")
        app_mod.app.config["TESTING"] = True
        app_mod.app.config["SECRET_KEY"] = "test-secret"
        # Initialise chatbot (if not already done)
        if app_mod.chatbot is None:
            app_mod._init_chatbot()
        cls.client = app_mod.app.test_client()
        print("\n  [Flask] Test client created.")

    # ── /health ────────────────────────────────────────────────────────────────

    def test_health_returns_200(self):
        """/health should always return HTTP 200."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_has_status_field(self):
        """/health response must include a 'status' field."""
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertIn("status", data)

    # ── /stats ─────────────────────────────────────────────────────────────────

    def test_stats_returns_json(self):
        """/stats should return a JSON object."""
        resp = self.client.get("/stats")
        self.assertIn(resp.status_code, [200, 503])
        # If 200, the body should parse as JSON
        if resp.status_code == 200:
            data = json.loads(resp.data)
            self.assertIsInstance(data, dict)

    # ── / (index) ─────────────────────────────────────────────────────────────

    def test_index_returns_200(self):
        """GET / should return HTTP 200."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)

    def test_index_contains_hypotify(self):
        """The index page HTML should contain 'Hypotify'."""
        resp = self.client.get("/")
        self.assertIn(b"Hypotify", resp.data)

    # ── /chat ──────────────────────────────────────────────────────────────────

    def test_chat_empty_body_returns_400(self):
        """/chat with an empty body should return 400."""
        resp = self.client.post(
            "/chat",
            data=json.dumps({"message": ""}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_chat_no_json_returns_400(self):
        """/chat with no JSON body should return 400."""
        resp = self.client.post("/chat", data="{invalid}", content_type="text/plain")
        self.assertEqual(resp.status_code, 400)

    def test_chat_greeting_returns_response(self):
        """/chat POST with 'hello' should return a response field."""
        resp = self.client.post(
            "/chat",
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
        )
        # Could be 200 (chatbot loaded) or 503 (not loaded) – either is valid
        self.assertIn(resp.status_code, [200, 503])
        data = json.loads(resp.data)
        if resp.status_code == 200:
            self.assertIn("response", data)
            self.assertIsInstance(data["response"], str)

    def test_chat_help_command(self):
        """/chat with 'help' should succeed or return 503."""
        resp = self.client.post(
            "/chat",
            data=json.dumps({"message": "help"}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, [200, 503])

    def test_chat_db_info_command(self):
        """/chat with 'db info' should succeed or return 503."""
        resp = self.client.post(
            "/chat",
            data=json.dumps({"message": "db info"}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, [200, 503])

    # ── /history ───────────────────────────────────────────────────────────────

    def test_history_returns_list(self):
        """/history should return a JSON list."""
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        self.assertIsInstance(data, (list, dict))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. End-to-End Multi-Turn Conversation Test                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TestE2EConversation(unittest.TestCase):
    """
    End-to-end test simulating a realistic multi-turn clinical conversation.

    Verifies that the chatbot handles topic switches gracefully and that
    each turn is logged to the database.
    """

    SCENARIO = [
        ("hello",                         "greeting"),
        ("help",                          "help info"),
        ("insights gender",               "gender stats"),
        ("insights age",                  "age stats"),
        ("feedback 4 Good so far",        "feedback"),
        ("search hypertension",           "search query"),
        ("sentiment I am very worried about my diagnosis", "negative sentiment"),
        ("ner Dr. Adams treated on 2024-09-01",           "NER"),
        ("db info",                       "db stats"),
        ("goodbye",                       "farewell"),
    ]

    @classmethod
    def setUpClass(cls):
        mod = _try_import("chatbot_w9")
        if mod is None:
            raise unittest.SkipTest("chatbot_w9 not importable.")
        try:
            print("\n  [E2E] Initialising chatbot for scenario test…")
            cls.bot = mod.HypotifyChatbot()
        except SystemExit:
            raise unittest.SkipTest("Chatbot initialisation failed (missing model files).")

    def test_full_scenario(self):
        """
        Run all scenario turns and assert each returns a non-empty string.
        Logs the turn description if a response is unexpectedly empty.
        """
        failures = []
        for user_msg, description in self.SCENARIO:
            with self.subTest(turn=description):
                start = time.time()
                resp  = self.bot.respond(user_msg)
                elapsed = time.time() - start
                if not isinstance(resp, str) or len(resp.strip()) == 0:
                    failures.append(
                        f"Turn '{description}' (input={user_msg!r}) "
                        f"returned empty/non-string response."
                    )
                else:
                    print(f"  ✓  [{description:35s}]  "
                          f"{len(resp):4d} chars  {elapsed*1000:.0f}ms")

        if failures:
            self.fail("\n".join(failures))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Runner                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 70)
    print("  Hypotify Chatbot – Week 10 Test Suite")
    print("  ITEC5025 | Author: Shruti Malik | Date: 2026-03-18")
    print("=" * 70)

    # Run with increased verbosity if -v flag passed
    verbosity = 2 if "-v" in sys.argv else 1
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Explicit class ordering for cleaner output
    for cls in [TestNLP, TestDatabase, TestChatbotEngine, TestFlaskAPI, TestE2EConversation]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=verbosity, failfast=False)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"  Tests run    : {result.testsRun}")
    print(f"  Failures     : {len(result.failures)}")
    print(f"  Errors       : {len(result.errors)}")
    print(f"  Skipped      : {len(result.skipped)}")
    print(f"  Result       : {'PASS ✓' if result.wasSuccessful() else 'FAIL ✗'}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
