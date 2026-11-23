from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from event_delivery import EventStore


class DeliveryTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "nested" / "events.sqlite3"
        self.store = EventStore(self.path)

    def test_order_is_scoped_to_stream_and_checkpoint_to_consumer(self):
        first = self.store.publish("paperImportRequested", paper_file="1.pdf")
        self.store.publish("conversationStarted", conversation_id="c1")
        last = self.store.publish("paperAddedToLibrary", paper_file="1.pdf")
        received = []
        self.assertEqual(self.store.deliver("audit", "paper-library", received.append, 1), 1)
        self.assertEqual(EventStore(self.path).deliver("audit", "paper-library", received.append), 1)
        self.assertEqual(received, [first, last])
        self.assertEqual(self.store.deliver("audit", "paper-library", received.append), 0)
        self.assertEqual(self.store.deliver("independent", "paper-library", received.append), 2)
        self.assertEqual(received, [first, last, first, last])

    def test_failed_event_is_retried_before_later_events(self):
        first = self.store.publish("conversationStarted")
        second = self.store.publish("questionSubmitted")
        third = self.store.publish("answerProvided")
        received = []

        def fail_second(event):
            received.append(event)
            if event == second:
                raise RuntimeError("temporary error")

        with self.assertRaisesRegex(RuntimeError, "temporary error"):
            self.store.deliver("audit", "research-assistance", fail_second)
        self.assertEqual(received, [first, second])
        self.assertEqual(EventStore(self.path).deliver("audit", "research-assistance", received.append), 2)
        self.assertEqual(received, [first, second, second, third])

    def test_unknown_event_and_invalid_payload_do_not_publish(self):
        with self.assertRaises(ValueError):
            self.store.publish("notInCatalog")
        with self.assertRaises(TypeError):
            self.store.publish("questionSubmitted", data=object())
        self.assertEqual(self.store.deliver("audit", "research-assistance", self.fail), 0)


if __name__ == "__main__":
    unittest.main()
