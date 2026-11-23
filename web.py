"""ScienceHelpy's local web application."""

from dataclasses import asdict
import re
from secrets import token_hex
from threading import Thread
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from flask import Flask, Response, g, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.exceptions import NotFound, RequestEntityTooLarge

from application import create_services
from chats import ChatConflict, ChatStore
from config import Settings
from library import InvalidPDF, MAX_PDF_BYTES
from weekly import WeeklyFeed, save_weekly_paper


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    library, research, events = create_services(settings)
    feed = WeeklyFeed(settings.events_db.with_name("weekly-papers.json"))
    chats = ChatStore(settings.events_db.with_name("chats.sqlite3"))
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

    def chat_owner():
        if "chat_owner" not in g:
            token = request.cookies.get("sciencehelpy_chats", "")
            if not re.fullmatch(r"[a-f0-9]{64}", token):
                token = token_hex(32)
                g.new_chat_owner = True
            g.chat_owner = token
        return g.chat_owner

    @app.after_request
    def chat_headers(response):
        if getattr(g, "new_chat_owner", False):
            response.set_cookie(
                "sciencehelpy_chats", g.chat_owner, max_age=365 * 24 * 60 * 60,
                secure=request.is_secure or settings.public_url.startswith("https://"),
                httponly=True, samesite="Lax",
            )
        if "chat_owner" in g or request.path.startswith("/api/chats/"):
            response.headers["Cache-Control"] = "private, no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_):
        return jsonify(error="Размер PDF не должен превышать 25 МБ."), 413

    @app.get("/")
    def index():
        chat_owner()
        return render_template("index.html")

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

    def import_weekly(identifier):
        paper = next((item for item in feed.read().get("papers", [])
                      if item["id"] == identifier), None)
        if paper is None:
            raise NotFound("Статья больше не входит в текущую подборку.")
        return save_weekly_paper(library, paper)

    @app.post("/api/weekly/<identifier>/save")
    def save_paper(identifier):
        try:
            saved, created = import_weekly(identifier)
        except NotFound as error:
            return jsonify(error=error.description), 404
        except (requests.RequestException, ValueError):
            app.logger.warning("Could not import arXiv paper %s", identifier, exc_info=True)
            return jsonify(error="Не удалось получить PDF с arXiv. Попробуйте ещё раз."), 502
        return jsonify(paper=asdict(saved), created=created), 201 if created else 200

    @app.get("/assistant")
    def assistant():
        chat_owner()
        return render_template("assistant_start.html", papers=[entry.paper for entry in library.entries()])

    @app.post("/assistant/from-weekly/<identifier>")
    def assistant_weekly(identifier):
        try:
            paper, _ = import_weekly(identifier)
        except NotFound as error:
            return render_template("assistant_start.html", papers=[], error=error.description), 404
        except (requests.RequestException, ValueError):
            return render_template("assistant_start.html", papers=[],
                                   error="Не удалось скачать PDF с arXiv. Попробуйте снова из каталога."), 502
        return redirect(url_for("assistant_paper", name=paper.file), code=303)

    @app.get("/assistant/<name>")
    def assistant_paper(name):
        entry = library.get(name)
        if entry is None:
            return render_template("assistant_start.html", papers=[],
                                   error="Эта статья больше не доступна."), 404
        chat_owner()
        return render_template("assistant.html", paper=entry.paper, ai_enabled=settings.ai_enabled)

    @app.get("/papers/<name>/page/<int:number>.png")
    def pdf_page(name, number):
        result = library.page_image(name, number)
        if result is None:
            raise NotFound
        data, etag = result
        response = Response(data, mimetype="image/png")
        response.set_etag(etag)
        response.cache_control.private = True
        response.cache_control.no_cache = True
        return response.make_conditional(request)

    @app.get("/api/chats/<name>")
    def list_chats(name):
        if not library.pdf_path(name):
            return jsonify(error="Статья больше не доступна."), 404
        return jsonify(chats.list(chat_owner(), name))

    @app.post("/api/chats/<name>")
    def new_chat(name):
        if not library.pdf_path(name):
            return jsonify(error="Статья больше не доступна."), 404
        thread = chats.create(chat_owner(), name)
        events.publish("conversationStarted", conversation_id=thread["id"])
        return jsonify(thread), 201

    @app.get("/api/chats/<name>/<thread_id>")
    def read_chat(name, thread_id):
        thread = chats.get(chat_owner(), name, thread_id)
        if thread is None:
            return jsonify(error="Чат не найден."), 404
        return jsonify(thread)

    @app.post("/api/chats/<name>/<thread_id>")
    def chat_turn(name, thread_id):
        if not library.pdf_path(name):
            return jsonify(error="Статья больше не доступна."), 404
        try:
            body, question = read_question()
        except ValueError as error:
            return jsonify(error=str(error)), 400
        owner = chat_owner()
        thread = chats.get(owner, name, thread_id, recent=3)
        if thread is None:
            return jsonify(error="Чат не найден."), 404
        revision = body.get("revision")
        if type(revision) is not int or thread["revision"] != revision:
            return jsonify(error="В чате есть новое сообщение. Обновили историю, отправь вопрос снова."), 409
        history = tuple((turn["question"], turn["answer"][:2500]) for turn in thread["turns"])
        result = research.ask(question, thread_id, name, history)
        reply = {key: result[key] for key in ("answer", "sources", "mode", "warning")}
        try:
            saved = chats.append(owner, name, thread_id, revision, question, reply)
        except ChatConflict:
            return jsonify(error="В чате есть новое сообщение. Обновили историю, отправь вопрос снова."), 409
        events.publish("answerProvided", conversation_id=thread_id,
                       question_id=result["question_id"], mode=result["mode"])
        return jsonify(saved), 201

    @app.get("/papers/<name>")
    def pdf(name):
        if not library.pdf_path(name):
            raise NotFound
        return send_from_directory(library.path.resolve(), name, mimetype="application/pdf")

    @app.post("/api/ask")
    def ask():
        try:
            body, question = read_question()
        except ValueError as error:
            return jsonify(error=str(error)), 400
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



def read_question() -> tuple[dict, str]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("question"), str):
        raise ValueError("Введите вопрос.")
    question = body["question"].strip()
    if not 3 <= len(question) <= 1500:
        raise ValueError("Длина вопроса должна быть от 3 до 1500 символов.")
    return body, question


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_env()
    feed = WeeklyFeed(settings.events_db.with_name("weekly-papers.json"))
    Thread(target=feed.run, daemon=True, name="weekly-papers").start()
    create_app(settings).run(host="127.0.0.1", port=8000)
