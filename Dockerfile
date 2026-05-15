FROM python:3.10-slim-bookworm
ARG APT_MIRROR

WORKDIR /app

RUN set -e; \
    if [ -n "$APT_MIRROR" ]; then \
        echo "deb http://$APT_MIRROR/debian bookworm main" > /etc/apt/sources.list; \
        echo "deb http://$APT_MIRROR/debian bookworm-updates main" >> /etc/apt/sources.list; \
        echo "deb http://security.debian.org/debian-security bookworm-security main" >> /etc/apt/sources.list; \
        rm -f /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only (~220 MB) — устанавливается отдельно через официальный wheel-индекс
RUN pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/
RUN mkdir -p /app/models /app/data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
