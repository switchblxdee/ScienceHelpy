"""Wire the shared library, research workflow, and event store."""

from config import ROOT, Settings
from event_delivery import EventStore
from library import Library
from research import Research

BUNDLED_PAPERS = {
    "1.pdf": ("Attention Is All You Need", "Ashish Vaswani et al.", "1706.03762"),
    "2.pdf": ("Deep Residual Learning for Image Recognition", "Kaiming He et al.", "1512.03385"),
    "3.pdf": ("Generative Adversarial Networks", "Ian Goodfellow et al.", "1406.2661"),
    "4.pdf": ("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "Jacob Devlin et al.", "1810.04805"),
    "5.pdf": ("Language Models are Few-Shot Learners", "Tom B. Brown et al.", "2005.14165"),
}


def create_services(settings: Settings):
    events = EventStore(settings.events_db)
    metadata = BUNDLED_PAPERS if settings.pdf_dir.resolve() == ROOT / "pdfs" else {}
    library = Library(settings.pdf_dir, events, metadata)
    return library, Research(settings, library, events), events
