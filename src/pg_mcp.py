from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
import os
import requests
from dotenv import load_dotenv
from typing import List
from datetime import datetime
'''

# pg_mcp.py
FastAPI app that gives table schema, example rows, and executes SQL 

'''
load_dotenv()
GRAFANA_URL = os.getenv("GRAFANA_URL")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY")

headers = {
    "Authorization": f"Bearer {GRAFANA_API_KEY}",
    "Content-Type": "application/json"
}
# Initialize the FastAPI app
app = FastAPI()

# Data model for the SQL request body
class SQLRequest(BaseModel):
    sql: List[str]
    chart_type: List[str]
    title: str = "LLM: Multi-panel Dashboard"

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
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor()

        for sql in request.sql:
            cur.execute(sql)
            rows = cur.fetchall()
            results.append(rows)

        cur.close()
        conn.close()
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}



@app.post("/grafana_json")
def create_grafana_dashboard(request: SQLRequest):
    sql = request.sql
    
    if isinstance(request.chart_type, list):
        chart_types = [ct.lower() for ct in request.chart_type]
    else:
        chart_types = [request.chart_type.lower()]

    # 🔍 Dynamically get UID for "fastapi-dashboards"
    # folders = requests.get(f"{GRAFANA_URL}/api/folders", headers=headers).json()
    response = requests.get(f"{GRAFANA_URL}/api/folders", headers=headers)
    print("📦 pg-mcp Folders raw response:", response.text)
    print("📦 pg-mcp Folders status code:", response.status_code)
    folders = response.json()
    if isinstance(folders, dict) and "message" in folders:
        print("❌ Grafana error:", folders)
        return {"error": folders["message"]}
    folder_uid = next((f["uid"] for f in folders if f["title"] == "LLM_To_POSTGRESQL_FOLDER"), None)
    print("📦 pg-mcp Folder UID:", folder_uid)
     # Map natural language types to Grafana panel types
    type_map = {
        "line chart": "timeseries",
        "bar chart": "barchart",
        "pie chart": "piechart",
        "table": "table",
        "scatter plot": "scatter",
        "area chart": "timeseries"
    }
    format_map = {
        "timeseries": "time_series",
        "barchart": "table",
        "piechart": "table",
        "table": "table",
        "scatter": "table"
    }
    
    panels = []
    for i, (sql, chart_type) in enumerate(zip(request.sql, request.chart_type)):
        panel_type = type_map.get(chart_type.lower(), "timeseries")
        format_type = format_map.get(panel_type, "time_series")
        print(f"▶️ Panel {i+1}: type={panel_type}, format={format_type}")

        panel = {
            "title": f"Panel {i+1}: {chart_type.title()}",
            "type": panel_type,
            "datasource": {
                "type": "grafana-postgresql-datasource",
                "uid": "aeo8prusu1i4gc"
            },
            "targets": [
                {
                    "refId": chr(65 + i),
                    "rawSql": sql,
                    "format": format_type,
                    "interval": "auto"
                }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": i * 8},
            "maxDataPoints": 1,
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "lineWidth": 1,
                        "fillOpacity": 80,
                        "axisPlacement": "auto"
                    }
                },
                "overrides": []
            },
            "options": {
                "tooltip": {"mode": "single"},
                "legend": {"displayMode": "list", "placement": "bottom"},
                **({"reduceOptions": {
                    "calcs": ["lastNotNull"],
                    "fields": "",
                    "values": True
                }} if panel_type == "piechart" else {})
            }
        }
        panels.append(panel)

    dashboard = {
        "dashboard": {
            "title": request.title + datetime.now().strftime("%H:%M:%S"),
            "refresh": "5s",
            "schemaVersion": 36,
            "version": 1,
            "panels": panels
        },
        "folderUid": folder_uid,
        "overwrite": False
    }

    # Clean conflicting fields
    for key in ["uid", "id", "folderId"]:
        dashboard["dashboard"].pop(key, None)

    print("📦 pg-mcp Dashboard JSON:", dashboard)
    return dashboard
