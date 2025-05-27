import mlflow
import os
import psycopg2


mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")  # fallback for local dev
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("llm-to-sql")

# def log_to_mlflow(prompt, sql, result):
def log_to_mlflow(prompt, sql, result):
    conn = psycopg2.connect(
        dbname="mlflow",  # hardcoded or separately configured
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
        host=os.getenv("POSTGRES_URL", "postgres"),
        port=os.getenv("POSTGRES_PORT", 5432),
    )
    with mlflow.start_run():
        mlflow.log_param("prompt", prompt)
        mlflow.log_param("generated_sql", sql)
        mlflow.log_metric("result_length", len(str(result)))

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")  # fallback for local dev
