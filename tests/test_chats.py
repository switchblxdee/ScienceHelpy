from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from config import Settings
from test_web import article
from web import create_app


class ChatsTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        self.settings = Settings(folder / "pdfs", folder / "papers.txt", folder / "events.sqlite3",
                                 "example", "", "", ollama_model="example")
        self.app = create_app(self.settings)
        self.client = self.app.test_client()
        self.first = "/api/chats/first.pdf"
        self.second = "/api/chats/second.pdf"
        (self.settings.pdf_dir / "first.pdf").write_bytes(article("First paper", "Attention aligns tokens."))
        (self.settings.pdf_dir / "second.pdf").write_bytes(article("Second paper", "Attention guards castles."))

    def test_threads_are_persisted_and_scoped_to_paper_and_browser(self):
        first = self.client.post(self.first).json
        second = self.client.post(self.first).json
        other_article = self.client.post(self.second).json
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual({row["id"] for row in self.client.get(self.first).json},
                         {first["id"], second["id"]})
        self.assertEqual([row["id"] for row in self.client.get(self.second).json], [other_article["id"]])
        self.assertEqual(self.client.get(self.second + "/" + first["id"]).status_code, 404)
        another_browser = self.app.test_client()
        self.assertEqual(another_browser.get(self.first).json, [])
        self.assertEqual(another_browser.get(self.first + "/" + first["id"]).status_code, 404)

        path = self.first + "/" + first["id"]
        with patch("research.post_json", return_value={"message": {"content": "Token alignment [1]."}}):
            result = self.client.post(path, json={"question": "How does attention work?", "revision": 0})
        self.assertEqual(result.status_code, 201)
        self.assertEqual(result.json["title"], "How does attention work?")
        self.assertEqual(result.json["turn"]["sources"][0]["file"], "first.pdf")
        self.assertEqual(self.client.post(path, json={
            "question": "Attention", "revision": 0,
        }).status_code, 409)

        reopened = create_app(self.settings).test_client()
        reopened.set_cookie("sciencehelpy_chats", self.client.get_cookie("sciencehelpy_chats").value)
        thread = reopened.get(path).json
        self.assertEqual(thread["turns"], [result.json["turn"]])
        self.assertEqual(reopened.get(self.first + "/" + second["id"]).json["turns"], [])

    def test_follow_up_uses_saved_history_and_only_selected_pdf(self):
        thread = self.client.post(self.first).json
        path = self.first + "/" + thread["id"]
        with patch("research.post_json", return_value={"message": {"content": "The saved answer."}}) as model:
            self.client.post(path, json={"question": "Explain attention", "revision": 0})
            next_turn = self.client.post(path, json={"question": "What does attention align?", "revision": 1})
        self.assertEqual(next_turn.status_code, 201)
        payload = model.call_args.args[2]["messages"]
        self.assertEqual(payload[1]["content"], "Explain attention")
        self.assertEqual(payload[2]["content"], "The saved answer.")
        self.assertIn("aligns tokens", payload[-1]["content"])
        self.assertNotIn("castles", payload[-1]["content"])

    def test_assistant_route_and_pdf_pages(self):
        self.assertIn(b"First paper", self.client.get("/assistant").data)
        self.assertIn(b'"/static/assistant.js"', self.client.get("/assistant/first.pdf").data)
        self.assertEqual(self.client.get("/assistant/missing.pdf").status_code, 404)
        page = self.client.get("/papers/first.pdf/page/1.png")
        self.assertEqual(page.mimetype, "image/png")
        self.assertTrue(page.data.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get("/papers/first.pdf/page/2.png").status_code, 404)

    def test_weekly_discussion_opens_the_imported_paper(self):
        from weekly import WeeklyFeed
        from test_weekly import listing
        from datetime import date

        feed = WeeklyFeed(self.settings.events_db.with_name("weekly-papers.json"))
        with patch("weekly.requests.get") as get:
            get.return_value.json.return_value = [listing("2611.00001", 9, 4)]
            feed.refresh(date(2026, 11, 23))
        with patch("weekly.requests.get") as get:
            response = get.return_value.__enter__.return_value
            response.status_code = 200
            response.iter_content.return_value = [article("A weekly article", "Attention works.")]
            opened = self.client.post("/assistant/from-weekly/2611.00001")
        self.assertEqual(opened.status_code, 303)
        self.assertEqual(opened.location, "/assistant/arxiv-2611_00001.pdf")
        self.assertEqual(self.client.get(opened.location).status_code, 200)
