from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

from config import DEFAULT_OPENROUTER_MODEL, Settings
from test_web import article
from web import create_app


class OpenRouterTest(unittest.TestCase):
    def test_shared_credentials_only_supply_openrouter_settings(self):
        with TemporaryDirectory() as temp:
            dotenv = Path(temp) / ".env"
            dotenv.write_text("OPENROUTER_API_KEY=example\nOPENROUTER_MODEL=nvidia/example\nGROQ_API=ignored\n")
            with patch.dict(os.environ, {"OPENROUTER_ENV_FILE": str(dotenv)}, clear=True):
                settings = Settings.from_env()
                self.assertTrue(settings.ai_enabled)
                self.assertEqual(settings.openrouter_key, "example")
                self.assertEqual(settings.openrouter_model, "nvidia/example")
                self.assertEqual(settings.groq_key, "")
                with patch.dict(os.environ, {"OPENROUTER_MODEL": "nvidia/override"}):
                    self.assertEqual(Settings.from_env().openrouter_model, "nvidia/override")

    def test_saved_chat_uses_openrouter_and_saves_generated_answer(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = Settings(root / "pdfs", root / "papers.txt", root / "events.db", "demo", "", "")
            settings = replace(baseline, openrouter_key="fake", ollama_model="local")
            client = create_app(settings).test_client()
            (settings.pdf_dir / "sample.pdf").write_bytes(article("Selected article", "Attention aligns words."))
            route = "/api/chats/sample.pdf"
            chat = client.post(route).json
            with patch("research.post_json", return_value={"choices": [{"message": {"content": "Alignment [1]."}}]}) as provider:
                saved = client.post(f"{route}/{chat['id']}",
                                    json={"question": "What does attention do?", "revision": 0})
            self.assertEqual(saved.status_code, 201)
            self.assertEqual(saved.json["turn"]["mode"], "generated")
            self.assertEqual(saved.json["turn"]["answer"], "Alignment [1].")
            url, _, payload = provider.call_args.args
            self.assertEqual(url, "https://openrouter.ai/api/v1/chat/completions")
            self.assertEqual(payload["model"], DEFAULT_OPENROUTER_MODEL)
            self.assertFalse(payload["reasoning"]["enabled"])
            self.assertIn("Attention aligns words", payload["messages"][-1]["content"])
