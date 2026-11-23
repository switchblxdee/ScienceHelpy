"""ScienceHelpy console interface."""

import argparse
from uuid import uuid4

from application import create_services
from config import Settings
from download_pdf import download_all_papers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-papers", action="store_true", help="Импортировать PDF из papers.txt")
    args = parser.parse_args()
    settings = Settings.from_env()
    library, research, events = create_services(settings)
    if args.import_papers:
        try:
            return int(bool(download_all_papers(settings.paper_list, library)))
        except OSError as error:
            parser.exit(1, f"Не удалось открыть список статей: {error}\n")
    conversation = str(uuid4())
    events.publish("conversationStarted", conversation_id=conversation)
    try:
        while True:
            question = input("Ваш вопрос (0 — выход): ").strip()
            if question == "0":
                break
            if not question:
                continue
            result = research.ask(question, conversation)
            print(result["answer"])
            if result["warning"]:
                print(result["warning"])
            for index, source in enumerate(result["sources"], 1):
                where = f"{source['file']}, стр. {source['page']}" if "file" in source else source["url"]
                print(f"\n[{index}] {source['title']} · {where}\n{source['text']}")
            events.publish("answerProvided", conversation_id=conversation,
                           question_id=result["question_id"], mode=result["mode"])
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        events.publish("conversationEnded", conversation_id=conversation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
