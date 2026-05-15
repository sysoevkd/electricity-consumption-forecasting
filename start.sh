#!/bin/bash
# Запуск ML сервиса (PJME) на любой машине с Docker.
# Использование: ./start.sh [полный|минимальный]
#   минимальный (по умолчанию) — только postgres + api
#   полный — все сервисы (Airflow, Prometheus, Grafana и т.д.)

set -e
MODE="${1:-минимальный}"

echo "=== Запуск ML сервиса (PJME) ==="

if ! docker info > /dev/null 2>&1; then
    echo "Ошибка: Docker не запущен. Запустите Docker Desktop или демон Docker."
    exit 1
fi

mkdir -p data models

if [ ! -f "models/best_boost_pjme.pkl" ]; then
  echo "Модель models/best_boost_pjme.pkl не найдена."
  echo "После запуска загрузите данные (POST /upload-data) и вызовите POST /retrain для обучения."
fi

# Linux: AIRFLOW_UID для полного стека
if [ -z "${AIRFLOW_UID:-}" ] && [ "$MODE" = "полный" ]; then
    export AIRFLOW_UID=$(id -u 2>/dev/null || echo 50000)
    echo "AIRFLOW_UID=$AIRFLOW_UID"
fi

if [ "$MODE" = "полный" ]; then
    echo "Запуск всех сервисов..."
    docker compose up -d
else
    echo "Запуск postgres + api..."
    docker compose up -d postgres api
fi

echo "Ожидание готовности API (до 120 сек)..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health | grep -q '"model_loaded":true'; then
        echo "API готов."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Таймаут. Проверьте логи: docker compose logs api"
        exit 1
    fi
    sleep 2
done

echo ""
echo "=== Сервис запущен ==="
echo "API:           http://localhost:8000"
echo "Документация:  http://localhost:8000/docs"
echo "Проверка:      curl http://localhost:8000/health"
echo ""
echo "Остановка:     docker compose stop api postgres  (или docker compose down)"
echo ""
