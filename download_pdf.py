import logging
import os
from pathlib import Path

import requests

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PATH_TO_PDFS = settings.PATH_TO_PDFS


def download_all_papers(pdf_list_path: Path) -> None:
    if not pdf_list_path.exists():
        logger.error("File with PDF URLs not found: %s", pdf_list_path)
        return

    urls = [
        line.strip()
        for line in pdf_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    os.makedirs(PATH_TO_PDFS, exist_ok=True)

    for index, url in enumerate(urls):
        dest = PATH_TO_PDFS / f"{index + 1}.pdf"

        if dest.exists():
            logger.info("File %d already exists: %s", index + 1, dest)
            continue

        logger.info("Downloading file %d from %s to %s", index + 1, url, dest)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
        except requests.RequestException as ex:
            logger.error("Failed to download file %d: %s", index + 1, ex)
            continue

        try:
            with open(dest, "wb") as file:
                file.write(response.content)
            logger.info("File %d downloaded successfully: %s", index + 1, dest)
        except IOError as ex:
            logger.error("Failed to save file %d: %s", index + 1, ex)
