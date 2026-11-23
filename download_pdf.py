"""Import the PDF URLs listed in the local operator's papers.txt."""

from pathlib import Path
from urllib.parse import urlsplit

import requests

from library import InvalidPDF, Library, MAX_PDF_BYTES


def download_all_papers(pdf_list_path: Path, library: Library) -> int:
    failures = 0
    urls = [line.strip() for line in pdf_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, url in enumerate(urls, 1):
        name = f"{index}.pdf"
        if library.pdf_path(name):
            continue
        data = bytearray()
        try:
            if urlsplit(url).scheme not in {"http", "https"}:
                raise ValueError("Ожидается HTTP или HTTPS")
            with requests.get(url, timeout=(5, 30), stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=65536):
                    data.extend(chunk)
                    if len(data) > MAX_PDF_BYTES:
                        raise ValueError("PDF больше 25 МБ")
        except (requests.RequestException, ValueError) as error:
            library.events.publish("paperImportRequested", paper_file=name)
            library.events.publish("paperImportFailed", paper_file=name, reason=type(error).__name__)
            failures += 1
            print(f"{name}: загрузка не удалась ({type(error).__name__})")
            continue
        try:
            library.add(bytes(data), name=name)
        except (InvalidPDF, OSError) as error:
            failures += 1
            print(f"{name}: {error}")
        else:
            print(f"{name}: добавлен")
    return failures
