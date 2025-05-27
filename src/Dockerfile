FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && apt-get clean

WORKDIR /app

# Copy entire project from root (you already have context: .)
COPY . .

# ✅ Requirements.txt is in root, so refer to full path
RUN pip install --no-cache-dir -r /app/src/requirements.txt


# Set Python path so imports like rag, mlflow_logger work
ENV PYTHONPATH=/app

CMD ["sh", "-c", "uvicorn $APP_MODULE --host 0.0.0.0 --port 8000"]
