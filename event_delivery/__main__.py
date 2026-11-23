"""Publish an example event or read events from the command line."""

import argparse
from dataclasses import asdict
import json
import os

from . import Event, EventStore
from .catalog import EVENT_STREAMS


def show(event: Event) -> None:
    print(json.dumps(asdict(event), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.getenv("SCIENCEHELPY_EVENTS_DB", ".sciencehelpy/events.sqlite3"))
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("name", choices=EVENT_STREAMS)
    publish.add_argument("--payload", default="{}")
    consume = commands.add_parser("consume")
    consume.add_argument("stream", choices=sorted(set(EVENT_STREAMS.values())))
    consume.add_argument("--consumer", required=True)
    consume.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if args.command == "publish":
        try:
            payload = json.loads(args.payload)
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            show(EventStore(args.db).publish(args.name, **payload))
        except (ValueError, TypeError) as error:
            parser.error(str(error))
    else:
        try:
            EventStore(args.db).deliver(args.consumer, args.stream, show, args.limit)
        except ValueError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
