FROM python:3.11-slim

# Kaleido 1.x renders PNG/SVG exports with a system Chromium (#267). DejaVu is the
# sans-serif that Kaleido 0.2 bundled, so exported text looks the same.
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium-headless-shell fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
ENV BROWSER_PATH=/usr/bin/chromium-headless-shell

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
