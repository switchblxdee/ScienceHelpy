"""Local PDF library and page-level text search."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from math import log
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from threading import RLock

import pymupdf

from event_delivery import EventStore

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 300
PDF_LOCK = RLock()  # PyMuPDF does not support concurrent calls from threads.
STOP_WORDS = set(
    "the and for are with what how does this that which from was were "
    "как что это для где чем или его она они при этих этой".split()
)


class InvalidPDF(ValueError):
    pass


def tokens(value: str):
    return [word for word in re.findall(r"[^\W_]+", value.casefold())
            if len(word) > 2 and word not in STOP_WORDS]


@dataclass(frozen=True)
class Paper:
    file: str
    title: str
    authors: str
    page_count: int
    arxiv_id: str = ""


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    words: Counter


@dataclass(frozen=True)
class IndexedPaper:
    paper: Paper
    pages: tuple[Page, ...]


def read_pdf(data: bytes, name: str, metadata: tuple[str, str, str] = ("", "", "")) -> IndexedPaper:
    if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise InvalidPDF("Нужен PDF размером до 25 МБ.")
    with PDF_LOCK:
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if document.needs_pass or not 0 < len(document) <= MAX_PAGES:
                    raise InvalidPDF("Нужен PDF без пароля, до 300 страниц.")
                meta = document.metadata or {}
                pages = []
                for index, page in enumerate(document, 1):
                    content = " ".join(page.get_text(sort=True).split())
                    words = Counter(tokens(content))
                    if words:
                        pages.append(Page(index, content, words))
                title_hint, author_hint, arxiv_id = metadata
                title = title_hint or " ".join((meta.get("title") or "").split())
                if not title and pages:
                    first_lines = document[0].get_text().splitlines()
                    title = next((line.strip() for line in first_lines
                                  if len(line.strip()) >= 12), "")
                authors = author_hint or " ".join((meta.get("author") or "").split())
                paper = Paper(name, (title or Path(name).stem)[:240],
                              authors[:300], len(document), arxiv_id)
        except (pymupdf.FileDataError, RuntimeError) as error:
            raise InvalidPDF("Не удалось прочитать PDF.") from error
    return IndexedPaper(paper, tuple(pages))


class Library:
    def __init__(self, path: Path, events: EventStore, metadata: dict[str, tuple[str, str, str]] | None = None):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.events = events
        self.metadata = metadata or {}
        self._cache = {}
        self._lock = RLock()

    def pdf_path(self, name: str) -> Path | None:
        if not name or Path(name).name != name or not name.lower().endswith(".pdf"):
            return None
        path = self.path / name
        return path if path.is_file() and not path.is_symlink() else None

    def entries(self) -> list[IndexedPaper]:
        with self._lock:
            paths = sorted(p for p in self.path.iterdir() if self.pdf_path(p.name))
            current = {}
            for path in paths:
                try:
                    info = path.stat()
                    version = (info.st_mtime_ns, info.st_size)
                except OSError:
                    continue
                cached = self._cache.get(path.name)
                if cached is not None and cached[0] == version:
                    current[path.name] = cached
                    continue
                try:
                    with path.open("rb") as file:
                        entry = read_pdf(file.read(MAX_PDF_BYTES + 1), path.name, self.metadata.get(path.name, ("", "", "")))
                except (InvalidPDF, OSError):
                    self.events.publish("paperPreparationFailed", paper_file=path.name, reason="unreadable_pdf")
                    entry = None
                else:
                    self._record_preparation(entry)
                current[path.name] = (version, entry)
            self._cache = current
            return [entry for _, entry in current.values() if entry is not None]

    def _record_preparation(self, entry: IndexedPaper) -> None:
        name = entry.paper.file
        if entry.pages:
            self.events.publish("paperContentPrepared", paper_file=name)
            self.events.publish("paperMadeSearchable", paper_file=name)
        else:
            self.events.publish("paperPreparationFailed", paper_file=name, reason="no_text")

    def add(self, data: bytes, name: str | None = None) -> tuple[Paper, bool]:
        name = name or f"{sha256(data).hexdigest()[:24]}.pdf"
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.pdf", name):
            raise InvalidPDF("Недопустимое имя PDF.")
        with self._lock:
            path = self.path / name
            cached = self._cache.get(name)
            if cached and cached[1] is not None and self.pdf_path(name):
                return cached[1].paper, False
            if self.pdf_path(name):
                with path.open("rb") as existing:
                    entry = read_pdf(existing.read(MAX_PDF_BYTES + 1), name, self.metadata.get(name, ("", "", "")))
                return entry.paper, False
            self.events.publish("paperImportRequested", paper_file=name)
            try:
                entry = read_pdf(data, name, self.metadata.get(name, ("", "", "")))
                with NamedTemporaryFile(dir=self.path, delete=False) as file:
                    temporary = Path(file.name)
                    try:
                        file.write(data)
                        file.close()
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
            except (InvalidPDF, OSError) as error:
                self.events.publish("paperImportFailed", paper_file=name, reason=type(error).__name__)
                raise
            info = path.stat()
            self._cache[name] = ((info.st_mtime_ns, info.st_size), entry)
            self.events.publish("paperAddedToLibrary", paper_file=name)
            self._record_preparation(entry)
            return entry.paper, True

    def search(self, query: str, file: str | None = None, limit: int = 5) -> list[dict]:
        terms = set(tokens(query))
        if not terms:
            return []
        pages = [(entry.paper, page) for entry in self.entries()
                 if file is None or entry.paper.file == file for page in entry.pages]
        frequency = Counter(term for _, page in pages for term in terms if term in page.words)
        ranked = []
        for paper, page in pages:
            score = sum((1 + log(page.words[term])) * log(1 + len(pages) / frequency[term])
                        for term in terms if term in page.words)
            if score:
                ranked.append((score, paper, page))
        ranked.sort(key=lambda result: (-result[0], result[1].file, result[2].number))
        return [
            {"file": paper.file, "title": paper.title, "page": page.number,
             "text": excerpt(page.text, terms)}
            for _, paper, page in ranked[:limit]
        ]


def excerpt(value: str, terms: set[str], length: int = 880) -> str:
    match = next((match for match in re.finditer(r"[^\W_]+", value)
                  if match.group().casefold() in terms), None)
    start = max(0, match.start() - 180) if match else 0
    end = start + length
    return ("…" if start else "") + value[start:end] + ("…" if end < len(value) else "")
