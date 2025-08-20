FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y git curl gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# (If you still need build deps temporarily)
RUN apt-get update \
    && apt-get install -y gcc g++ libpq-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && apt-get remove -y gcc g++ libpq-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV PYTHONPATH=/app/src
ENV APP_MODULE=llm_api:app

EXPOSE 8000
CMD ["uvicorn", "src.llm_api:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.10-slim

# WORKDIR /app
# COPY main.py .
# COPY requirements.txt .

# # Install build dependencies for some wheels (especially for qdrant-client, sentence-transformers, psycopg2-binary)
# RUN apt-get update && apt-get install -y gcc g++ libpq-dev && \
#     pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     apt-get remove -y gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# EXPOSE 8000

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


