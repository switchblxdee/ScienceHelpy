"""ScienceHelpy's local web application."""

from dataclasses import asdict
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import NotFound, RequestEntityTooLarge

from application import create_services
from config import Settings
from library import InvalidPDF, MAX_PDF_BYTES


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    library, research, events = create_services(settings)
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_PDF_BYTES + 64 * 1024,
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
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
            if origin and origin != request.host_url.rstrip("/"):
                return jsonify(error="Запрос с другого сайта отклонён."), 403

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_):
        return jsonify(error="Размер PDF не должен превышать 25 МБ."), 413

    @app.get("/")
    def index():
        return render_template("index.html", ai_enabled=bool(settings.groq_key))

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
        conversation = str(uuid4())
        events.publish("conversationStarted", conversation_id=conversation)
        try:
            result = research.ask(question, conversation, file)
            events.publish("answerProvided", conversation_id=conversation,
                           question_id=result["question_id"], mode=result["mode"])
            return jsonify({key: result[key] for key in ("answer", "sources", "mode", "warning")})
        finally:
            events.publish("conversationEnded", conversation_id=conversation)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000)
