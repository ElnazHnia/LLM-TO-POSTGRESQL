from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
import os

'''

# pg_mcp.py
FastAPI app that gives table schema, example rows, and executes SQL 

'''
# Initialize the FastAPI app
app = FastAPI()

# Data model for the SQL request body
class SQLRequest(BaseModel):
    sql: str

# Helper function to establish connection to the PostgreSQL database
def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DATABASE"),   # Use env vars for security
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "postgres"),  # default host name
        port=os.getenv("POSTGRES_PORT", 5432)
    )

# Endpoint to fetch the schema (column names and types) of a table
@app.get("/schema/{table_name}")
def get_schema(table_name: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return {"schema": [f"{col} ({dtype})" for col, dtype in rows]}
    except Exception as e:
        return {"error": str(e)}

# Endpoint to fetch one example row from a table
@app.get("/example/{table_name}")
def get_example_row(table_name: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table_name} LIMIT 1")
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()
        return {"example": dict(zip(columns, row)) if row else {}}
    except Exception as e:
        return {"error": str(e)} 

# Endpoint to execute a raw SQL query and return the first row of the result
@app.post("/execute")
def execute_query(request: SQLRequest):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(request.sql)
        result = cur.fetchall()
        cur.close()
        conn.close()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
