# main.py
from typing import List
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

app = FastAPI()


@app.get("/tables", response_model=List[str], summary="List all DB tables")
def list_tables():
    return ["users", "products", "orders"]

mcp = FastApiMCP(app)
mcp.mount_http()   # ← exposes both GET /mcp and GET /mcp/openapi.json