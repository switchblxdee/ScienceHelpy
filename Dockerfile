FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SCIENCEHELPY_EVENTS_DB=/data/events.sqlite3
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY event_delivery ./event_delivery
COPY static ./static
COPY templates ./templates
COPY pdfs ./pdfs
COPY papers.txt ./
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "240", "web:create_app()"]
