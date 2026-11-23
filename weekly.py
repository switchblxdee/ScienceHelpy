"""Cache the latest complete week of Hugging Face Daily Papers."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
from time import sleep

import requests

from config import Settings
from library import Library, MAX_PDF_BYTES, atomic_write

FEED_URL = "https://huggingface.co/api/daily_papers"
ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(?:v\d+)?")
logger = logging.getLogger(__name__)


def previous_week(today: date) -> tuple[date, date]:
    monday = today - timedelta(days=today.weekday())
    return monday - timedelta(days=7), monday


def count(value):
    return value if type(value) is int and value > 0 else 0


@dataclass(frozen=True)
class WeeklyFeed:
    path: Path

    def read(self) -> dict:
        try:
            snapshot = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(snapshot, dict) and isinstance(snapshot.get("papers"), list):
                return snapshot
        except (FileNotFoundError, ValueError):
            pass
        return {"week": "", "papers": []}

    def refresh(self, today: date | None = None) -> dict:
        start, end = previous_week(today or datetime.now(timezone.utc).date())
        year, week, _ = start.isocalendar()
        key = f"{year}-W{week:02d}"
        current = self.read()
        if current.get("week") == key and current.get("papers"):
            return current

        request = requests.get(FEED_URL, params={"week": key, "limit": 100, "sort": "publishedAt"},
                               timeout=(5, 15))
        request.raise_for_status()
        rows = request.json()
        if not isinstance(rows, list):
            raise ValueError("Unexpected response from Hugging Face")
        ranked = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("paper"), dict):
                continue
            paper = row["paper"]
            identifier, title = paper.get("id"), paper.get("title")
            if not isinstance(identifier, str) or not ARXIV_ID.fullmatch(identifier):
                continue
            if not isinstance(title, str) or not title.strip():
                continue
            try:
                featured = date.fromisoformat(row.get("publishedAt", "")[:10])
                if not start <= featured < end:
                    continue
            except (TypeError, ValueError):
                continue
            authors = paper.get("authors")
            authors = [item["name"] for item in authors if isinstance(item, dict)
                       and isinstance(item.get("name"), str)] if isinstance(authors, list) else []
            summary = paper.get("summary")
            identifier = identifier.split("v")[0]
            item = {
                "id": identifier,
                "title": title.strip()[:240],
                "authors": ", ".join(authors[:3])[:300],
                "summary": " ".join(summary.split())[:1400] if isinstance(summary, str) else "",
                "upvotes": count(paper.get("upvotes")),
                "comments": count(row.get("numComments")),
                "date": featured.isoformat(),
            }
            previous = ranked.get(identifier)
            if previous is None or popularity(item) > popularity(previous):
                ranked[identifier] = item

        if not ranked:
            raise ValueError(f"No dated papers found for {key}")
        snapshot = {
            "week": key,
            "start": start.isoformat(),
            "end": (end - timedelta(days=1)).isoformat(),
            "papers": sorted(ranked.values(), key=popularity, reverse=True)[:9],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(self.path, json.dumps(snapshot, ensure_ascii=False).encode())
        return snapshot

    def run(self, interval: int = 3600):
        while True:
            try:
                snapshot = self.refresh()
                logger.info("Weekly papers available for %s", snapshot["week"])
            except (OSError, ValueError):
                logger.warning("Could not update weekly papers; keeping the previous edition",
                               exc_info=True)
            sleep(interval)


def popularity(item):
    return item["upvotes"] + 2 * item["comments"], item["date"], item["id"]


def save_weekly_paper(library: Library, paper: dict):
    identifier = paper["id"]
    if not ARXIV_ID.fullmatch(identifier):
        raise ValueError("Invalid arXiv identifier")
    name = f"arxiv-{identifier.replace('.', '_')}.pdf"
    existing = library.get(name)
    if existing:
        return existing.paper, False

    data = bytearray()
    try:
        with requests.get(f"https://arxiv.org/pdf/{identifier}", stream=True,
                          timeout=(5, 15), allow_redirects=False) as response:
            if response.status_code != 200:
                raise requests.HTTPError(f"arXiv HTTP {response.status_code}")
            for chunk in response.iter_content(chunk_size=65536):
                data.extend(chunk)
                if len(data) > MAX_PDF_BYTES:
                    raise ValueError("PDF превысил допустимые 25 МБ.")
    except (requests.RequestException, ValueError) as error:
        library.events.publish("paperImportRequested", paper_file=name)
        library.events.publish("paperImportFailed", paper_file=name, reason=type(error).__name__)
        raise
    return library.add(bytes(data), name, (paper["title"], paper["authors"], identifier))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Обновлять подборку в фоне")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    feed = WeeklyFeed(Settings.from_env().events_db.with_name("weekly-papers.json"))
    if args.watch:
        feed.run()
    else:
        snapshot = feed.refresh()
        print(f"{snapshot['week']}: {len(snapshot['papers'])} papers")


if __name__ == "__main__":
    main()
