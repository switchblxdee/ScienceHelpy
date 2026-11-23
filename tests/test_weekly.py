from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import requests

from config import Settings
from web import create_app
from weekly import WeeklyFeed, previous_week
from test_web import article


def listing(identifier, votes, comments, featured="2026-11-17", title="Transformer Study"):
    return {
        "paper": {"id": identifier, "title": title, "upvotes": votes,
                  "authors": [{"name": "Researcher"}], "summary": "About transformers."},
        "publishedAt": featured, "numComments": comments,
    }


class WeeklyTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.settings = Settings(root / "pdfs", root / "papers.txt", root / "events.db", "model", "", "")
        self.feed = WeeklyFeed(root / "weekly-papers.json")

    @patch("weekly.requests.get")
    def test_weekly_rank_is_scoped_to_week_cached_and_kept_on_failure(self, get):
        get.return_value.json.return_value = [
            listing("2611.00001", 9, 0), listing("2611.00002", 2, 5),
            listing("2611.00002", 1, 0), listing("2611.00003", 100, 0, "2026-11-12"),
            listing("../invalid", 100, 0),
        ]
        now = date(2026, 11, 23)
        self.assertEqual(previous_week(now), (date(2026, 11, 16), now))
        first = self.feed.refresh(now)
        self.assertEqual(first["week"], "2026-W47")
        self.assertEqual([paper["id"] for paper in first["papers"]], ["2611.00002", "2611.00001"])
        self.assertEqual(self.feed.refresh(date(2026, 11, 29)), first)
        get.assert_called_once()

        get.side_effect = requests.Timeout
        with self.assertRaises(requests.Timeout):
            self.feed.refresh(date(2026, 11, 30))
        self.assertEqual(self.feed.read(), first)

    def test_weekly_save_is_idempotent_and_metadata_survives_restart(self):
        with patch("weekly.requests.get") as get:
            get.return_value.json.return_value = [listing("2611.00001", 9, 4)]
            self.feed.refresh(date(2026, 11, 23))

        client = create_app(self.settings).test_client()
        self.assertEqual(client.get("/api/weekly").json["papers"][0]["id"], "2611.00001")
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.iter_content.return_value = [article("Original title", "A transformer uses attention.")]
        with patch("weekly.requests.get", return_value=response) as get:
            path = "/api/weekly/2611.00001/save"
            self.assertEqual(client.post(path).status_code, 201)
            self.assertEqual(client.post(path).status_code, 200)
            self.assertEqual(client.post("/api/weekly/2611.99999/save").status_code, 404)
            get.assert_called_once()
            self.assertEqual(get.call_args.args[0], "https://arxiv.org/pdf/2611.00001")
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
        fresh = create_app(self.settings).test_client()
        saved, = fresh.get("/api/papers").json
        self.assertEqual(saved["title"], "Transformer Study")
        self.assertEqual(saved["arxiv_id"], "2611.00001")

    def test_article_chat_passes_bounded_history_and_source_to_local_model(self):
        location = self.settings.pdf_dir
        location.mkdir()
        (location / "model-paper.pdf").write_bytes(article("Selected paper", "Attention yields alignment."))
        (location / "other.pdf").write_bytes(article("Other paper", "Attention guards a castle."))
        settings = replace(self.settings, ollama_model="test-model")
        client = create_app(settings).test_client()
        with patch("research.post_json", return_value={"message": {"content": "Alignment [1]."}}) as model:
            answer = client.post("/api/ask", json={
                "question": "What does attention yield?",
                "file": "model-paper.pdf",
                "history": [{"question": "Define attention", "answer": "Earlier response."}],
            })
        self.assertEqual(answer.json["mode"], "generated")
        self.assertEqual(answer.json["sources"][0]["file"], "model-paper.pdf")
        messages = model.call_args.args[2]["messages"]
        self.assertEqual(messages[2]["content"], "Earlier response.")
        self.assertIn("Attention yields alignment", messages[-1]["content"])
        self.assertNotIn("castle", messages[-1]["content"])
        self.assertEqual(client.post("/api/ask", json={
            "question": "attention", "file": "model-paper.pdf", "history": [None] * 4,
        }).status_code, 400)


if __name__ == "__main__":
    unittest.main()
