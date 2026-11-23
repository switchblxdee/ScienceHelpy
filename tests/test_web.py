from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
import requests

from config import Settings
from event_delivery import EventStore
from web import create_app


def article(title: str, text: str) -> bytes:
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 50), text)
        doc.set_metadata({"title": title, "author": "Example author"})
        return doc.tobytes()


class WebTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)
        self.settings = Settings(path / "pdfs", path / "papers.txt", path / "events.db", "example", "", "")
        self.client = create_app(self.settings).test_client()

    def upload(self, content):
        return self.client.post("/api/papers", data={"pdf": (BytesIO(content), "article.pdf")})

    def test_upload_search_and_open_pdf(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/papers").json, [])
        data = article("Learning Transformers", "Transformers use attention for sequence modelling.")
        first = self.upload(data)
        self.assertEqual(first.status_code, 201)
        file = first.json["paper"]["file"]
        self.assertEqual(self.upload(data).status_code, 200)
        self.assertEqual(len(self.client.get("/api/papers").json), 1)

        response = self.client.post("/api/ask", json={"question": "How does attention work?", "file": file})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["mode"], "extracts")
        source, = response.json["sources"]
        self.assertEqual((source["file"], source["page"]), (file, 1))
        self.assertIn("sequence modelling", source["text"])
        with self.client.get(f"/papers/{file}") as pdf:
            self.assertEqual(pdf.mimetype, "application/pdf")
            self.assertEqual(pdf.data, data)

        events = EventStore(self.settings.events_db)
        received = []
        events.deliver("audit", "paper-preparation", received.append)
        self.assertEqual([event.name for event in received],
                         ["paperContentPrepared", "paperMadeSearchable"])

    def test_selecting_paper_limits_search_and_handles_empty_result(self):
        unrelated = self.upload(article("Comets", "A comet orbits the Sun.")).json["paper"]["file"]
        relevant = self.upload(article("Neural nets", "Transformers use attention.")).json["paper"]["file"]
        response = self.client.post("/api/ask", json={"question": "attention", "file": unrelated})
        self.assertEqual(response.json["mode"], "empty")
        self.assertEqual(response.json["sources"], [])
        response = self.client.post("/api/ask", json={"question": "attention"})
        self.assertEqual(response.json["sources"][0]["file"], relevant)

    def test_bad_input_is_handled_without_leaving_a_pdf(self):
        self.assertEqual(self.upload(b"%PDF-1.4 invalid").status_code, 400)
        self.assertEqual(self.client.get("/api/papers").json, [])
        self.assertEqual(self.client.post("/api/ask", json=[]).status_code, 400)
        response = self.client.post("/api/ask", json={"question": "attention", "file": "../article.pdf"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/papers/missing.pdf").status_code, 404)

    def test_web_fallback_provides_sources_when_generation_fails(self):
        settings = replace(self.settings, tavily_key="fake-tavily", groq_key="fake-groq")
        client = create_app(settings).test_client()
        search = requests.Response()
        search.status_code = 200
        search._content = b'{"results":[{"title":"Research","url":"https://example.org/paper","content":"Evidence."}]}'
        with patch("research.requests.Session.post", side_effect=[search, requests.Timeout]) as provider:
            response = client.post("/api/ask", json={"question": "attention"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["mode"], "extracts")
        self.assertEqual(response.json["sources"][0]["title"], "Research")
        self.assertIn("Модель сейчас недоступна", response.json["warning"])
        self.assertEqual(provider.call_count, 2)

    def test_external_origins_cannot_submit_requests(self):
        response = self.client.post("/api/ask", json={"question": "attention"},
                                    headers={"Origin": "https://example.org"})
        self.assertEqual(response.status_code, 403)
