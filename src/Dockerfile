FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

ENV PYTHONPATH=/app/src
ENV APP_MODULE=llm_api:app

CMD ["uvicorn", "src.llm_api:app", "--host", "0.0.0.0", "--port", "8000"]
