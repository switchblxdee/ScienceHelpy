"""Local PDF library and page-level text search."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
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


def read_pdf(data: bytes, name: str, metadata: tuple[str, str, str] = EMPTY_METADATA, *, index: bool = True) -> IndexedPaper:
    if len(data) > MAX_PDF_BYTES or not data.startswith(b"%PDF-"):
        raise InvalidPDF("Нужен PDF размером до 25 МБ.")
    with PDF_LOCK:
        try:
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if document.needs_pass or not 0 < len(document) <= MAX_PAGES:
                    raise InvalidPDF("Нужен PDF без пароля, до 300 страниц.")
                meta = document.metadata or {}
                pages = []
                for number, page in enumerate(document if index else (), 1):
                    content = " ".join(page.get_text(sort=True).split())
                    words = Counter(tokens(content))
                    if words:
                        pages.append(Page(number, content, words))
                title_hint, author_hint, arxiv_id = metadata
                title = title_hint or " ".join((meta.get("title") or "").split())
                if not title or title.lower().startswith("arxiv:"):
                    title = first_page_title(document[0])
                authors = author_hint or " ".join((meta.get("author") or "").split())
                paper = Paper(name, (title or Path(name).stem)[:240],
                              authors[:300], len(document), arxiv_id)
        except (pymupdf.FileDataError, RuntimeError) as error:
            raise InvalidPDF("Не удалось прочитать PDF.") from error
    return IndexedPaper(paper, tuple(pages))


def first_page_title(page: pymupdf.Page) -> str:
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    lines = []
    for block in page.get_text("dict", flags=flags)["blocks"]:
        for line in block.get("lines", ()):
            if abs(line["dir"][1]) > 0.1:
                continue
            y = line["bbox"][1]
            if y > page.rect.height * 0.42:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if len(text) >= 4 and not text.lower().startswith("arxiv:"):
                size = max(span["size"] for span in line["spans"])
                lines.append((y, size, text))
    if not lines:
        return ""
    size = max(line[1] for line in lines)
    selected = sorted(line for line in lines if line[1] >= size - 0.15)
    top = selected[0][0]
    return " ".join(text for y, _, text in selected if y - top <= 4 * size)


class Library:
    def __init__(self, path: Path, events: EventStore, metadata: dict[str, tuple[str, str, str]] | None = None):
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.events = events
        self.metadata = metadata or {}
        self._cache = {}
        self._paper_cache = {}
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

    def inspect(self, name: str) -> Paper | None:
        """Read catalog fields without extracting text from every page."""
        with self._lock:
            path = self.pdf_path(name)
            if path is None:
                return None
            try:
                info = path.stat()
                version = (info.st_mtime_ns, info.st_size)
                indexed = self._cache.get(name)
                if indexed is not None and indexed[0] == version and indexed[1] is not None:
                    return indexed[1].paper
                cached = self._paper_cache.get(name)
                if cached is not None and cached[0] == version:
                    return cached[1]
                with path.open("rb") as file:
                    paper = read_pdf(file.read(MAX_PDF_BYTES + 1), name, self._metadata(path), index=False).paper
            except (InvalidPDF, OSError):
                return None
            self._paper_cache[name] = (version, paper)
            return paper

    def catalog(self) -> list[Paper]:
        with self._lock:
            names = sorted(path.name for path in self.path.iterdir() if self.pdf_path(path.name))
            for removed in self._paper_cache.keys() - set(names):
                del self._paper_cache[removed]
            return [paper for name in names if (paper := self.inspect(name)) is not None]

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

    def page_preview(self, name: str, number: int) -> tuple[Path, str] | None:
        path = self.pdf_path(name)
        if path is None or not 1 <= number <= MAX_PAGES:
            return None
        try:
            info = path.stat()
        except OSError:
            return None
        return path, f"{info.st_ino:x}-{info.st_mtime_ns:x}-{info.st_size:x}-{number}"

    def overview(self, file: str) -> list[dict]:
        entry = self.get(file)
        if not entry:
            return []
        return [
            {"file": file, "title": entry.paper.title, "page": page.number,
             "text": excerpt(page.text, set(), 1600)}
            for page in entry.pages[:3]
        ]


@lru_cache(maxsize=32)
def render_pdf_page(path: Path, number: int, revision: str) -> bytes | None:
    """The revision in the cache key invalidates previews when the PDF changes."""
    try:
        with PDF_LOCK, pymupdf.open(path) as doc:
            if doc.needs_pass or number > len(doc):
                return None
            page = doc[number - 1]
            scale = min(2, 1400 / max(page.rect.width, page.rect.height, 1))
            return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
    except (OSError, RuntimeError, pymupdf.FileDataError):
        return None


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
