from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from config import Settings
from event_delivery import EventStore
from research import Research


class IntentTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        settings = Settings(root / "pdfs", root / "papers.txt", root / "events.db",
                            "model", "", "", ollama_model="example")
        self.events = EventStore(settings.events_db)
        self.library = Mock()
        self.research = Research(settings, self.library, self.events)

    def test_small_talk_and_general_questions_do_not_retrieve_or_send_paper_text(self):
        for question, responses in [
            ("Привет!", ["Привет! Чем помочь?"]),
            ("Посоветуй, как распределить время на учёбу.", [
                '{"intent":"chat","keywords":""}', "Выдели два коротких блока.",
            ]),
        ]:
            with self.subTest(question=question), patch.object(self.research, "_chat", side_effect=responses) as chat:
                result = self.research.ask(question, "thread", "selected.pdf")
                self.assertEqual(result["sources"], [])
                messages = chat.call_args.args[0]
                self.assertNotIn("Источники:", messages[-1]["content"])
        self.library.search.assert_not_called()
        self.library.overview.assert_not_called()
        self.assertEqual(self.events.deliver("verify", "source-discovery", self.fail), 0)

    def test_article_followup_uses_context_and_returns_citations(self):
        source = {"title": "Example", "text": "The method keeps positions distinct.", "page": 2}
        self.library.search.return_value = [source]
        with patch.object(self.research, "_chat", side_effect=[
            '{"intent":"article","keywords":"positional encoding motivation"}', "It distinguishes positions [1].",
        ]) as chat:
            reply = self.research.ask("А зачем они это добавили?", "thread", "selected.pdf",
                                      (("Как в статье устроено позиционное кодирование?", "Авторы добавили кодирование."),))
        self.assertIn("позиционное кодирование", chat.call_args_list[0].args[0][-1]["content"])
        self.library.search.assert_called_once_with("positional encoding motivation", "selected.pdf")
        self.assertEqual(reply["sources"], [source])
        self.assertIn("The method keeps positions distinct.", chat.call_args.args[0][-1]["content"])

    def test_unclassified_question_does_not_attach_pdf(self):
        with patch.object(self.research, "_chat", side_effect=["invalid output", "Хорошо."]):
            self.assertEqual(self.research.ask("Придумай название для клуба", "thread", "selected.pdf")["sources"], [])
        self.library.search.assert_not_called()
        self.library.overview.assert_not_called()


    def test_explicit_english_paper_question_only_calls_model_for_the_answer(self):
        self.library.search.return_value = [{"title": "Example", "text": "A result.", "page": 2}]
        with patch.object(self.research, "_chat", return_value="The paper reports a result [1].") as model:
            result = self.research.ask("What are the results in this paper?", "thread", "selected.pdf")
        self.assertEqual(result["mode"], "generated")
        model.assert_called_once()
