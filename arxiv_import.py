"""Validate public arXiv references and import their PDF."""

import re
from urllib.parse import unquote, urlsplit

import requests

from library import InvalidPDF, Library, MAX_PDF_BYTES

ARXIV_ID = re.compile(r"(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z.\-]{0,23}/[0-9]{7})(?:v[1-9][0-9]{0,5})?", re.I | re.ASCII)


class InvalidReference(ValueError):
    pass


class ArxivNotFound(ValueError):
    pass


def arxiv_id(reference: str) -> str:
    reference = reference.strip()
    if len(reference) > 350:
        raise InvalidReference
    if reference.lower().startswith("arxiv:"):
        reference = reference[6:].strip()
    elif "://" in reference:
        try:
            url = urlsplit(reference)
            if url.scheme not in {"http", "https"} or url.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}:
                raise InvalidReference
            section, separator, reference = unquote(url.path).lstrip("/").partition("/")
            if not separator or section not in {"abs", "pdf", "html"}:
                raise InvalidReference
            reference = reference.rstrip("/").removesuffix(".pdf")
        except ValueError as error:
            raise InvalidReference from error
    if not ARXIV_ID.fullmatch(reference):
        raise InvalidReference
    return reference.lower()


def save_arxiv_paper(library: Library, identifier: str, title: str = "", authors: str = ""):
    identifier = arxiv_id(identifier)
    name = f"arxiv-{identifier.replace('.', '_').replace('/', '--')}.pdf"
    existing = library.inspect(name)
    if existing:
        return existing, False
    for known_name, (_, _, known_id) in library.metadata.items():
        if identifier == known_id and (known := library.inspect(known_name)):
            return known, False

    data = bytearray()
    try:
        with requests.get(f"https://arxiv.org/pdf/{identifier}", stream=True,
                          timeout=(5, 20), allow_redirects=False) as response:
            if response.status_code == 404:
                raise ArxivNotFound
            if response.status_code != 200:
                raise requests.HTTPError(f"arXiv HTTP {response.status_code}")
            for chunk in response.iter_content(chunk_size=65536):
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    raise InvalidPDF("Размер PDF не должен превышать 25 МБ.")
    except (requests.RequestException, ValueError) as error:
        library.events.publish("paperImportRequested", paper_file=name)
        library.events.publish("paperImportFailed", paper_file=name, reason=type(error).__name__)
        raise
    return library.add(bytes(data), name, (title, authors, identifier))
