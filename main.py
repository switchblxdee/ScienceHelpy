import os
from uuid import uuid4

from config import settings
from download_pdf import download_all_papers
from event_delivery import EventStore
from graph import Graph


def main():
    events = EventStore(
        os.getenv("SCIENCEHELPY_EVENTS_DB", ".sciencehelpy/events.sqlite3")
    )
    download_all_papers(settings.PATH_TO_PDFS_URL, events=events)
    model = Graph()
    conversation = {"conversation_id": str(uuid4())}
    events.publish("conversationStarted", **conversation)
    try:
        prompt = "Давай начнем диалог: "
        while True:
            query = input(prompt)
            if query == "0":
                break
            question = {**conversation, "question_id": str(uuid4())}
            events.publish("questionSubmitted", **question)
            try:
                answer = model.run(user_prompt=query)["messages"][-1].content
                print(answer)
            except Exception as error:
                events.publish(
                    "answerCouldNotBeProvided", **question, reason=type(error).__name__
                )
                raise
            events.publish("answerProvided", **question)
            prompt = "Ваше сообщение: "
    except EOFError:
        pass
    finally:
        events.publish("conversationEnded", **conversation)


if __name__ == "__main__":
    main()
