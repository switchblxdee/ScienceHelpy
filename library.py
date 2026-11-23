"""Local PDF library and page-level text search."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
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
EMPTY_METADATA = ("", "", "")
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


def read_pdf(data: bytes, name: str, metadata: tuple[str, str, str] = EMPTY_METADATA) -> IndexedPaper:
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

    def _metadata(self, path: Path) -> tuple[str, str, str]:
        if path.name in self.metadata:
            return self.metadata[path.name]
        sidecar = path.with_suffix(".meta.json")
        try:
            if sidecar.is_symlink() or sidecar.stat().st_size > 4096:
                return EMPTY_METADATA
            values = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(values, list) and len(values) == 3 and all(isinstance(value, str) for value in values):
                return tuple(values)
        except (OSError, ValueError):
            pass
        return EMPTY_METADATA

    def get(self, name: str) -> IndexedPaper | None:
        with self._lock:
            path = self.pdf_path(name)
            if path is None:
                self._cache.pop(name, None)
                return None
            try:
                info = path.stat()
            except OSError:
                return None
            version = (info.st_mtime_ns, info.st_size)
            cached = self._cache.get(name)
            if cached is not None and cached[0] == version:
                return cached[1]
            try:
                with path.open("rb") as file:
                    entry = read_pdf(file.read(MAX_PDF_BYTES + 1), name, self._metadata(path))
            except (InvalidPDF, OSError):
                self.events.publish("paperPreparationFailed", paper_file=name, reason="unreadable_pdf")
                entry = None
            else:
                self._record_preparation(entry)
            self._cache[name] = (version, entry)
            return entry

    def entries(self) -> list[IndexedPaper]:
        with self._lock:
            names = sorted(path.name for path in self.path.iterdir() if self.pdf_path(path.name))
            for removed in self._cache.keys() - set(names):
                del self._cache[removed]
            return [entry for name in names if (entry := self.get(name)) is not None]

    def _record_preparation(self, entry: IndexedPaper) -> None:
        name = entry.paper.file
        if entry.pages:
            self.events.publish("paperContentPrepared", paper_file=name)
            self.events.publish("paperMadeSearchable", paper_file=name)
        else:
            self.events.publish("paperPreparationFailed", paper_file=name, reason="no_text")

    def add(self, data: bytes, name: str | None = None,
            metadata: tuple[str, str, str] | None = None) -> tuple[Paper, bool]:
        name = name or f"{sha256(data).hexdigest()[:24]}.pdf"
        if not re.fullmatch(r"[a-zA-Z0-9_-]+\.pdf", name):
            raise InvalidPDF("Недопустимое имя PDF.")
        with self._lock:
            path = self.path / name
            existing = self.get(name)
            if existing is not None:
                return existing.paper, False
            self.events.publish("paperImportRequested", paper_file=name)
            try:
                entry = read_pdf(data, name, metadata or self._metadata(path))
                if metadata:
                    atomic_write(path.with_suffix(".meta.json"), json.dumps(metadata).encode())
                atomic_write(path, data)
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
        entries = [self.get(file)] if file else self.entries()
        pages = [(entry.paper, page) for entry in entries if entry for page in entry.pages]
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

    def page_image(self, name: str, number: int) -> tuple[bytes, str] | None:
        path = self.pdf_path(name)
        if path is None or number < 1:
            return None
        try:
            with PDF_LOCK, pymupdf.open(path) as doc:
                if doc.needs_pass or number > min(len(doc), MAX_PAGES):
                    return None
                page = doc[number - 1]
                scale = min(2, 1400 / max(page.rect.width, page.rect.height, 1))
                image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                return image.tobytes("png"), f"{path.stat().st_mtime_ns:x}-{number}"
        except (OSError, RuntimeError, pymupdf.FileDataError):
            return None

    def overview(self, file: str) -> list[dict]:
        entry = self.get(file)
        if not entry:
            return []
        return [
            {"file": file, "title": entry.paper.title, "page": page.number,
             "text": excerpt(page.text, set(), 1600)}
            for page in entry.pages[:3]
        ]


def excerpt(value: str, terms: set[str], length: int = 880) -> str:
    match = next((match for match in re.finditer(r"[^\W_]+", value)
                  if match.group().casefold() in terms), None)
    start = max(0, match.start() - 180) if match else 0
    end = start + length
    return ("…" if start else "") + value[start:end] + ("…" if end < len(value) else "")


def atomic_write(path: Path, data: bytes) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
