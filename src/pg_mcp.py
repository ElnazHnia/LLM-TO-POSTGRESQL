# pg_mcp.py — FASTAPI + fastapi-mcp (no register_tool calls)

from fastapi import FastAPI
from pydantic import BaseModel
import os, re, time
import psycopg2
from psycopg2 import OperationalError
from psycopg2 import sql as pg_sql
from typing import List, Dict, Any, Set
from collections import defaultdict
from dotenv import load_dotenv

# Exposes FastAPI endpoints to MCP clients
from fastapi_mcp import FastApiMCP
from langchain_community.utilities import SQLDatabase
# Optional: only used if you later re-enable the RAG endpoint
try:
    from rag.context_retriever import get_panel_intents  # noqa: F401
except Exception:
    get_panel_intents = None  # placeholder to avoid import errors

load_dotenv()

# ===== Env / DB
PG_USER     = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_HOST     = os.getenv("POSTGRES_HOST", "db")
PG_PORT     = os.getenv("POSTGRES_PORT", "5432")
# Accept either POSTGRES_DATABASE or POSTGRES_DB
PG_DB       = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
DB_URI = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
db_sql = SQLDatabase.from_uri(DB_URI)

app = FastAPI()

# ===== Models
class SQLRequest(BaseModel):
    sql: List[str]
    chart_type: List[str] = []     # kept for compat, not used here
    title: str = "LLM: Multi-panel Dashboard"

class PanelIntentsRequest(BaseModel):
    prompt: str

class CheckRequest(BaseModel):
    sql: str

# ===== Connection
def get_connection(max_retries: int = 5, backoff_base: float = 1.0):
    for attempt in range(1, max_retries + 1):
        try:
            return psycopg2.connect(
                dbname=PG_DB, user=PG_USER, password=PG_PASSWORD,
                host=PG_HOST, port=PG_PORT
            )
        except OperationalError as e:
            if "recovery mode" in str(e).lower() and attempt < max_retries:
                time.sleep(backoff_base * 2 ** (attempt - 1))
            else:
                raise
    raise OperationalError("Exceeded max_retries connecting to Postgres")

@app.get("/health", status_code=200, include_in_schema=False)
def health_check():
    return {"status": "ok"}

# ===== Table filter (business tables only)
def is_business_table(table_name: str) -> bool:
    t = table_name.lower()
    patterns = [
        r'^alembic_', r'_metrics$', r'_tags$', r'version', r'^trace_',
        r'^latest_', r'^registered_', r'datasets', r'experiments',
        r'inputs', r'metrics', r'params', r'runs', r'tags',
    ]
    return not any(re.search(p, t) for p in patterns)

# ===== Catalog (tables -> columns) with TTL
CATALOG_TTL_SECS = 60
_catalog: Dict[str, Any] = {}
_catalog_built_at = 0.0

def _build_catalog(conn) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    by_table = defaultdict(list)
    for t, c in cur.fetchall():
        if is_business_table(t):
            by_table[t].append(c)
    cur.close()
    tables = sorted(by_table.keys())
    columns = {t: by_table[t] for t in tables}
    return {"tables": tables, "columns": columns}

def get_catalog(force: bool = False) -> Dict[str, Any]:
    global _catalog, _catalog_built_at
    now = time.time()
    if force or not _catalog or (now - _catalog_built_at) > CATALOG_TTL_SECS:
        conn = get_connection()
        try:
            _catalog = _build_catalog(conn)
            _catalog_built_at = now
        finally:
            conn.close()
    return _catalog

# ===== SQL validation (mirrors your Safe* logic)
SQL_KEYWORDS = {
    "select","from","where","group","by","order","limit","offset","having","join","inner","left","right",
    "full","outer","on","as","and","or","not","in","is","null","distinct","case","when","then","else","end",
    "asc","desc","between","like","ilike","exists","union","all","true","false","with","over","partition",
    "avg","sum","min","max","count","date_trunc","coalesce","concat","round","cast","extract"
}

# ===== Routes exposing safe info
@app.get("/db/dialect", operation_id="get_db_dialect")
def get_db_dialect():
    return {"dialect": db_sql.dialect}

@app.get("/db/tables", operation_id="get_usable_table_names")
def get_usable_table_names():
    return db_sql.get_usable_table_names()

def _has_select_star(sql: str) -> bool:
    return re.search(r"\bSELECT\s+\*\b", sql, re.IGNORECASE) is not None

def _extract_alias_map(sql: str) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for tbl, ali in re.findall(r"\bFROM\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)", sql, re.IGNORECASE):
        alias_map[ali] = tbl
    for tbl, ali in re.findall(r"\bJOIN\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)", sql, re.IGNORECASE):
        alias_map[ali] = tbl
    return alias_map

def _extract_identifiers(sql: str):
    alias_map = _extract_alias_map(sql)
    tables = set(re.findall(r"\bFROM\s+([a-zA-Z_][\w\.]*)", sql, re.IGNORECASE))
    tables |= set(re.findall(r"\bJOIN\s+([a-zA-Z_][\w\.]*)", sql, re.IGNORECASE))
    cols = set(re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql))
    bare = set(re.findall(r"\b([a-zA-Z_][\w]*)\b", sql))
    return tables, cols, bare, alias_map

def _find_unqualified_or_unknown(sql: str, tables_used: Set[str], alias_map: Dict[str,str], catalog: Dict[str,Any]):
    known_cols = set()
    for t in tables_used:
        t_short = t.split(".")[-1]
        known_cols |= set(catalog["columns"].get(t_short, []))

    tokens = [tok.lower() for tok in re.findall(r"\b([a-zA-Z_][\w]*)\b", sql)]
    tokens = [t for t in tokens if t not in SQL_KEYWORDS and not t.isdigit()]
    aliases = set(alias_map.keys())
    tables_short = {t.split(".")[-1].lower() for t in tables_used}
    tokens = [t for t in tokens if t not in aliases and t not in tables_short]

    qualified_cols = {c.lower() for _, c in re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql)}

    unqualified, unknown = [], []
    for tok in tokens:
        if tok in qualified_cols:
            continue
        if tok in {c.lower() for c in known_cols}:
            unqualified.append(tok)
        elif re.search(r"(_id$|_name$|^name$|date$|value$)", tok) and tok not in {c.lower() for c in known_cols}:
            unknown.append(tok)

    def dedupe(seq):
        seen=set(); out=[]
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return dedupe(unqualified), dedupe(unknown)

def validate_sql(sql: str) -> str | None:
    if not isinstance(sql, str) or not sql.strip():
        return "ERROR: Provide a non-empty SQL string."
    if _has_select_star(sql):
        return "ERROR: Do not use SELECT *. List columns explicitly, and qualify them with table aliases/names."

    catalog = get_catalog()
    tables, cols, _, alias_map = _extract_identifiers(sql)

    tables_short = {t.split(".")[-1] for t in tables}
    tables_short |= {alias_map.get(a, a) for a in alias_map.keys()}

    unknown_tables = {t for t in tables_short if t not in set(catalog["tables"])}
    if unknown_tables:
        return f"ERROR: Unknown tables {sorted(list(unknown_tables))}. Allowed: {sorted(catalog['tables'])}."

    bad_cols = []
    for t, c in cols:
        base = alias_map.get(t, t).split(".")[-1]
        if base in catalog["columns"]:
            if c not in catalog["columns"][base]:
                bad_cols.append(f"{t}.{c}")
        else:
            bad_cols.append(f"{t}.{c}")
    if bad_cols:
        return ("ERROR: Unknown columns "
                f"{bad_cols}. Use only columns from SCHEMA and qualify every column.")

    unq, unk = _find_unqualified_or_unknown(sql, tables_short, alias_map, catalog)
    if unq:
        return ("ERROR: Unqualified column(s) detected: "
                f"{unq}. Always table-qualify every column (e.g., sales.value, person.name).")
    if unk:
        return ("ERROR: Unknown column name(s) referenced: "
                f"{unk}. Check SCHEMA for the actual column names.")
    return None

# ===== REST endpoints (operation_id defines MCP tool names)

@app.get("/tables", operation_id="sql.list_tables")
def list_tables():
    try:
        cat = get_catalog()
        return cat["tables"]
    except Exception as e:
        return {"error": str(e)}

@app.get("/schema/{table_name}", operation_id="sql.schema")
def get_schema(table_name: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        rows = cur.fetchall()
        cur.close(); conn.close()
        if not rows:
            return {"error": f"Unknown table '{table_name}'."}
        return {"schema": [f"{col} ({dtype})" for col, dtype in rows]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/example/{table_name}", operation_id="sql.example")
def get_example_row(table_name: str):
    try:
        cat = get_catalog()
        if table_name not in set(cat["tables"]):
            return {"error": f"Unknown table '{table_name}'."}
        conn = get_connection(); cur = conn.cursor()
        q = pg_sql.SQL("SELECT * FROM {} LIMIT 1").format(pg_sql.Identifier(table_name))
        cur.execute(q)
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description]
        cur.close(); conn.close()
        return {"example": dict(zip(columns, row)) if row else {}}
    except Exception as e:
        return {"error": str(e)}

@app.post("/check", operation_id="sql.check")
def check_sql(req: CheckRequest):
    err = validate_sql(req.sql)
    return {"status": "OK" if not err else "ERROR", "message": err or "OK"}

@app.post("/execute", operation_id="sql.query")
def execute_query(request: SQLRequest):
    results = []
    try:
        conn = get_connection(); cur = conn.cursor()
        for sql_text in request.sql:
            err = validate_sql(sql_text)
            if err:
                cur.close(); conn.close()
                return {"error": err}
            cur.execute(sql_text)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            results.append([dict(zip(cols, r)) for r in rows])
        cur.close(); conn.close()
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

# ===== Optional RAG passthrough (disabled by default)
# @app.post("/panel_intents", operation_id="rag.panel_intents")
# def panel_intents_endpoint(req: PanelIntentsRequest):
#     if get_panel_intents is None:
#         return {"error": "RAG module not available"}
#     return get_panel_intents(req.prompt)

# ===== MCP exposure (no register_tool; tools come from operation_id)
mcp = FastApiMCP(app)
mcp.mount_http()
