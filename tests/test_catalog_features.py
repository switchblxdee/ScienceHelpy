from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from config import Settings
from test_web import article
from web import create_app


class CatalogFeaturesTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.settings = Settings(root / "pdfs", root / "papers.txt", root / "events.db", "model", "", "")
        self.client = create_app(self.settings).test_client()

    def test_arxiv_import_accepts_legacy_id_and_modern_link_and_reuses_saved_file(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.iter_content.return_value = [article("Example paper", "Attention.")]
        with patch("arxiv_import.requests.get", return_value=response) as download:
            for ref, identifier in (
                ("hep-th/9711200", "hep-th/9711200"),
                ("https://arxiv.org/pdf/1706.03762v2.pdf#page=8", "1706.03762v2"),
            ):
                with self.subTest(ref=ref):
                    result = self.client.post("/api/arxiv/import", json={"reference": ref})
                    self.assertEqual(result.status_code, 201)
                    self.assertEqual(result.json["paper"]["arxiv_id"], identifier)
                    self.assertEqual(download.call_args.args[0], f"https://arxiv.org/pdf/{identifier}")
                    self.assertFalse(download.call_args.kwargs["allow_redirects"])
                    self.assertEqual(self.client.post("/api/arxiv/import", json={"reference": ref}).status_code, 200)
            bad = self.client.post("/api/arxiv/import", json={"reference": "https://arxiv.org@other.test/pdf/1706.03762"})
            self.assertEqual(bad.status_code, 400)
            self.assertEqual(download.call_count, 2)

    def test_language_is_saved_across_catalog_and_chat(self):
        (self.settings.pdf_dir / "paper.pdf").write_bytes(article("Example paper", "Attention."))
        result = self.client.get("/language/en", query_string={"next": "/assistant/paper.pdf?chat=example"},
                                 follow_redirects=True)
        self.assertIn(b'<html lang="en">', result.data)
        self.assertIn(b"Chats for this paper", result.data)
        self.assertIn(b'What is this paper about?' if self.settings.ai_enabled else b"Find the abstract", result.data)
        self.assertIn(b"Find and download from arXiv", self.client.get("/").data)
        error = self.client.post("/api/arxiv/import", json={"reference": "invalid"})
        self.assertEqual(error.json["error"], "Enter an arXiv ID or an arXiv link.")
        result = self.client.get("/language/ru", query_string={"next": "//other.test"})
        self.assertEqual(result.headers["Location"], "/")
        self.assertIn(b'<html lang="ru">', self.client.get("/").data)

    def test_catalog_does_not_index_text_and_conditional_preview_does_not_render(self):
        (self.settings.pdf_dir / "paper.pdf").write_bytes(article("Example paper", "Attention."))
        from library import read_pdf
        with patch("library.read_pdf", wraps=read_pdf) as parse:
            self.assertEqual(self.client.get("/api/papers").json[0]["title"], "Example paper")
            self.assertFalse(parse.call_args.kwargs["index"])
        response = self.client.get("/papers/paper.pdf/page/1.png")
        self.assertEqual(response.status_code, 200)
        with patch("web.render_pdf_page") as render:
            repeated = self.client.get("/papers/paper.pdf/page/1.png", headers={"If-None-Match": response.headers["ETag"]})
            self.assertEqual(repeated.status_code, 304)
            render.assert_not_called()
