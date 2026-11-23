"""ScienceHelpy's local web application."""

from dataclasses import asdict
from threading import Thread
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import NotFound, RequestEntityTooLarge

from application import create_services
from config import Settings
from library import InvalidPDF, MAX_PDF_BYTES
from weekly import WeeklyFeed, save_weekly_paper


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    library, research, events = create_services(settings)
    feed = WeeklyFeed(settings.events_db.with_name("weekly-papers.json"))
    app = Flask(__name__)
    hosts = ["localhost", "127.0.0.1", "[::1]"]
    if settings.public_url:
        public = urlsplit(settings.public_url)
        if public.scheme not in {"http", "https"} or not public.hostname or public.path or public.query or public.fragment:
            raise ValueError("SCIENCEHELPY_PUBLIC_URL must be an HTTP(S) origin")
        hosts.append(public.hostname)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_PDF_BYTES + 64 * 1024,
        TRUSTED_HOSTS=hosts,
    )

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    @app.before_request
    def check_origin():
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and origin not in {request.host_url.rstrip("/"), settings.public_url or None}:
                return jsonify(error="Запрос с другого сайта отклонён."), 403

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_):
        return jsonify(error="Размер PDF не должен превышать 25 МБ."), 413

    @app.get("/")
    def index():
        return render_template("index.html", ai_enabled=settings.ai_enabled)

    @app.get("/api/papers")
    def papers():
        return jsonify([asdict(entry.paper) for entry in library.entries()])

    @app.post("/api/papers")
    def upload():
        file = request.files.get("pdf")
        if not file:
            return jsonify(error="Выберите PDF для загрузки."), 400
        try:
            paper, created = library.add(file.stream.read(MAX_PDF_BYTES + 1))
        except InvalidPDF as error:
            return jsonify(error=str(error)), 400
        return jsonify(paper=asdict(paper), created=created), 201 if created else 200

    @app.get("/api/weekly")
    def weekly():
        return jsonify(feed.read())

    @app.post("/api/weekly/<identifier>/save")
    def save_paper(identifier):
        paper = next((item for item in feed.read().get("papers", [])
                      if item["id"] == identifier), None)
        if paper is None:
            return jsonify(error="Статья больше не входит в текущую подборку."), 404
        try:
            saved, created = save_weekly_paper(library, paper)
        except (requests.RequestException, ValueError):
            app.logger.warning("Could not import arXiv paper %s", identifier, exc_info=True)
            return jsonify(error="Не удалось получить PDF с arXiv. Попробуйте ещё раз."), 502
        return jsonify(paper=asdict(saved), created=created), 201 if created else 200

    @app.get("/papers/<name>")
    def pdf(name):
        if not library.pdf_path(name):
            raise NotFound
        return send_from_directory(library.path.resolve(), name, mimetype="application/pdf")

    @app.post("/api/ask")
    def ask():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("question"), str):
            return jsonify(error="Введите вопрос."), 400
        question = body["question"].strip()
        if not 3 <= len(question) <= 1500:
            return jsonify(error="Длина вопроса должна быть от 3 до 1500 символов."), 400
        file = body.get("file")
        if file is not None and (not isinstance(file, str) or not library.pdf_path(file)):
            return jsonify(error="Статья больше не доступна."), 404
        turns = body.get("history", [])
        if not isinstance(turns, list) or len(turns) > 3 or any(
            not isinstance(turn, dict)
            or not isinstance(turn.get("question"), str)
            or not isinstance(turn.get("answer"), str)
            or not 3 <= len(turn["question"]) <= 1500
            or not 1 <= len(turn["answer"]) <= 2500
            for turn in turns
        ):
            return jsonify(error="Некорректная история диалога."), 400
        history = tuple((turn["question"], turn["answer"]) for turn in turns)
        conversation = str(uuid4())
        events.publish("conversationStarted", conversation_id=conversation)
        try:
            result = research.ask(question, conversation, file, history)
            events.publish("answerProvided", conversation_id=conversation,
                           question_id=result["question_id"], mode=result["mode"])
            return jsonify({key: result[key] for key in ("answer", "sources", "mode", "warning")})
        finally:
            events.publish("conversationEnded", conversation_id=conversation)

    return app


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    feed = WeeklyFeed(settings.events_db.with_name("weekly-papers.json"))
    Thread(target=feed.run, daemon=True, name="weekly-papers").start()
    create_app(settings).run(host="127.0.0.1", port=8000)
