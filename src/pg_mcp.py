# pg_mcp.py — FASTAPI + fastapi-mcp (no register_tool calls)

from fastapi import FastAPI, Request
from pydantic import BaseModel
import os, re, time, logging, json, logging.config, requests
import psycopg2
from psycopg2 import OperationalError
from psycopg2 import sql as pg_sql
from typing import List, Dict, Any, Set
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime, timezone
# Exposes FastAPI endpoints to MCP clients
from fastapi_mcp import FastApiMCP
from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
# Optional: only used if you later re-enable the RAG endpoint
try:
    from rag.context_retriever import get_panel_intents  # noqa: F401
except Exception:
    get_panel_intents = None  # placeholder to avoid import errors

# ===== Logging
# ---- Logging (file + console)

LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)  # ensure directory exists

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,  # let uvicorn/fastapi loggers be re-configured
    "formatters": {
        "default": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "formatter": "default",
        },
        "rotating_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": os.getenv("LOG_LEVEL", "INFO").upper(),
            "formatter": "default",
            "filename": os.path.join(LOG_DIR, "mcp.log"),
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    # configure uvicorn too so all app logs land in the file
    "loggers": {
        "pg_mcp":       {"level": os.getenv("LOG_LEVEL","INFO").upper(), "handlers": ["console","rotating_file"], "propagate": False},
        "uvicorn":      {"level": os.getenv("LOG_LEVEL","INFO").upper(), "handlers": ["console","rotating_file"], "propagate": False},
        "uvicorn.error":{"level": os.getenv("LOG_LEVEL","INFO").upper(), "handlers": ["console","rotating_file"], "propagate": False},
        "uvicorn.access":{"level": os.getenv("LOG_LEVEL","INFO").upper(), "handlers": ["console","rotating_file"], "propagate": False},
        "sqlalchemy.engine": {"level": "WARNING", "handlers": ["console","rotating_file"], "propagate": False},
    },
    "root": {  # anything uncaught
        "level": os.getenv("LOG_LEVEL", "INFO").upper(),
        "handlers": ["console", "rotating_file"],
    },
}

logging.config.dictConfig(LOGGING)

logger = logging.getLogger("pg_mcp")
logger.info("[FASTAPIMCP] logging configured; file=%s", os.path.join(LOG_DIR, "mcp.log"))
# ===== FastAPI app + middleware + DB init

# 1) Load .env first
load_dotenv()

# 2) Configure a dedicated logger (non-propagating)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("pg_mcp")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S%z")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(LOG_LEVEL)
logger.propagate = False

# 3) Create the FastAPI app BEFORE any decorators use it
app = FastAPI()

# 4) Now it's safe to add middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        logger.info(f"[FASTAPIMCP] REQUEST {request.method} {request.url.path}")
        resp = await call_next(request)
        dur = int((time.time() - start) * 1000)
        logger.info(f"[FASTAPIMCP] RESPONSE {request.method} {request.url.path} "
                    f"status={resp.status_code} duration_ms={dur}")
        return resp
    except Exception:
        dur = int((time.time() - start) * 1000)
        logger.exception(f"[FASTAPIMCP] EXC {request.method} {request.url.path} duration_ms={dur}")
        raise

# 5) Now read the rest of env & init DB
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_HOST = os.getenv("POSTGRES_HOST", "db")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DATABASE") or os.getenv("POSTGRES_DB")
DB_URI = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

db_sql = SQLDatabase.from_uri(DB_URI)


# ===== Models
class SQLRequest(BaseModel):
    sql: List[str]
    chart_type: List[str] = []     # kept for compat, not used here
    title: str = "LLM: Multi-panel Dashboard"

class PanelIntentsRequest(BaseModel):
    prompt: str

class CheckRequest(BaseModel):
    sql: str

def _extract_select_aliases(sql: str) -> Set[str]:
    """Extract column aliases from SELECT clause (AS clauses)"""
    select_part = re.split(r'\bFROM\b', sql, flags=re.IGNORECASE)[0]
    aliases = set()
    for match in re.findall(r'\bAS\s+([a-zA-Z_][\w]*)\b', select_part, re.IGNORECASE):
        aliases.add(match.lower())
    return aliases

# ===== Connection
def get_connection(max_retries: int = 5, backoff_base: float = 1.0):
    logger.info(f"[FASTAPIMCP] GET CONNECTION {PG_HOST}:{PG_PORT}, DB: {PG_DB}, User: {PG_USER}")
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
    logger.info("[FASTAPIMCP] HEALTH CHECK")
    return {"status": "ok"}

# ===== Table filter (business tables only)
def is_business_table(table_name: str) -> bool:
    logger.info(f"[FASTAPIMCP] IS BUSINESS TABLE: {table_name}")
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
    logger.info("[FASTAPIMCP] BUILD CATALOG")
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
    logger.info("[FASTAPIMCP] GET CATALOG")
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
    logger.info("[FASTAPIMCP] GET DB DIALECT")
    return {"dialect": db_sql.dialect}

@app.get("/db/tables", operation_id="get_usable_table_names")
def get_usable_table_names():
    logger.info(f"[FASTAPIMCP] GET USABLE TABLE NAMES {db_sql.get_usable_table_names()}, type {type(db_sql.get_usable_table_names())} ")
    return db_sql.get_usable_table_names()

def _has_select_star(sql: str) -> bool:
    logger.info(f"[FASTAPIMCP] HAS SELECT STAR: {sql}")
    return re.search(r"\bSELECT\s+\*\b", sql, re.IGNORECASE) is not None

def _extract_alias_map(sql: str) -> Dict[str, str]:
    logger.info(f"[FASTAPIMCP] EXTRACT ALIAS MAP: {sql}")
    alias_map: Dict[str, str] = {}
    for tbl, ali in re.findall(r"\bFROM\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)", sql, re.IGNORECASE):
        alias_map[ali] = tbl
    for tbl, ali in re.findall(r"\bJOIN\s+([a-zA-Z_][\w]*)\s+(?:AS\s+)?([a-zA-Z_][\w]*)", sql, re.IGNORECASE):
        alias_map[ali] = tbl
    return alias_map

def _extract_identifiers(sql: str):
    logger.info(f"[FASTAPIMCP] EXTRACT IDENTIFIERS: {sql}")
    alias_map = _extract_alias_map(sql)
    tables = set(re.findall(r"\bFROM\s+([a-zA-Z_][\w\.]*)", sql, re.IGNORECASE))
    tables |= set(re.findall(r"\bJOIN\s+([a-zA-Z_][\w\.]*)", sql, re.IGNORECASE))
    cols = set(re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql))
    bare = set(re.findall(r"\b([a-zA-Z_][\w]*)\b", sql))
    return tables, cols, bare, alias_map

def _find_unqualified_or_unknown(sql: str, tables_used: Set[str], alias_map: Dict[str,str], catalog: Dict[str,Any]):
    try:
        logger.info(f"[FASTAPIMCP] FIND UNQUALIFIED OR UNKNOWN:  {sql}")
        known_cols = set()
        logger.info(f"[FASTAPIMCP] TABLES USED: {tables_used}, ALIAS MAP: {alias_map}")
        for t in tables_used:
            t_short = t.split(".")[-1]
            known_cols |= set(catalog["columns"].get(t_short, []))
            logger.info(f"[FASTAPIMCP] KNOWN COLS FOR {t_short}: {catalog['columns'].get(t_short, [])}")
        logger.info(f"[FASTAPIMCP] KNOWN COLS: {known_cols}")
        tokens = [tok.lower() for tok in re.findall(r"\b([a-zA-Z_][\w]*)\b", sql)]
        tokens = [t for t in tokens if t not in SQL_KEYWORDS and not t.isdigit()]
        logger.info(f"[FASTAPIMCP] TOKENS BEFORE FILTERING: {tokens}")
        # Get SELECT aliases and exclude them from validation
        select_aliases = _extract_select_aliases(sql)
        logger.info(f"[FASTAPIMCP] SELECT ALIASES TO IGNORE: {select_aliases}")
        
        aliases = set(alias_map.keys())
        tables_short = {t.split(".")[-1].lower() for t in tables_used}
        logger.info(f"[FASTAPIMCP] TABLES SHORT: {tables_short}, ALIASES TO IGNORE: {aliases}")
        # Remove both table aliases and SELECT aliases from validation
        tokens = [t for t in tokens if t not in aliases and t not in tables_short and t not in select_aliases]
        logger.info(f"[FASTAPIMCP] TOKENS AFTER FILTERING: {tokens}")
        qualified_cols = {c.lower() for _, c in re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", sql)}
        logger.info(f"[FASTAPIMCP] QUALIFIED COLS: {qualified_cols}")
        unqualified, unknown = [], []
        for tok in tokens:
            if tok in qualified_cols:
                continue
            if tok in {c.lower() for c in known_cols}:
                unqualified.append(tok)
            elif re.search(r"(_id$|_name$|^name$|date$|value$)", tok) and tok not in {c.lower() for c in known_cols}:
                unknown.append(tok)
        logger.info(f"[FASTAPIMCP] PRE-DEDUPE UNQUALIFIED: {unqualified}, UNKNOWN: {unknown}")
        def dedupe(seq):
            seen=set(); out=[]
            for x in seq:
                if x not in seen:
                    seen.add(x); out.append(x)
            logger.info(f"[FASTAPIMCP] DEDUPED LIST: {out}")
            return out

        logger.info(f"[FASTAPIMCP] UNQUALIFIED: {unqualified}, UNKNOWN: {unknown}")
        return dedupe(unqualified), dedupe(unknown)
    except Exception as e:
        logger.exception(f"[FASTAPIMCP] EXCEPTION IN FIND UNQUALIFIED OR UNKNOWN: {e}")
        return [], []

def validate_sql(sql: str) -> str | None:
    logger.info(f"[FASTAPIMCP] VALIDATE SQL: {sql}")
    if not isinstance(sql, str) or not sql.strip():
        return "ERROR: Provide a non-empty SQL string."
    if _has_select_star(sql):
        logger.info("[FASTAPIMCP] SELECT * DETECTED")
        return "ERROR: Do not use SELECT *. List columns explicitly, and qualify them with table aliases/names."

    catalog = get_catalog()
    tables, cols, _, alias_map = _extract_identifiers(sql)

    tables_short = {t.split(".")[-1] for t in tables}
    tables_short |= {alias_map.get(a, a) for a in alias_map.keys()}

    unknown_tables = {t for t in tables_short if t not in set(catalog["tables"])}
    if unknown_tables:
        logger.info(f"[FASTAPIMCP] UNKNOWN TABLES: {unknown_tables}")
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
        logger.info(f"[FASTAPIMCP] BAD COLUMNS: {bad_cols}")
        return ("ERROR: Unknown columns "
                f"{bad_cols}. Use only columns from SCHEMA and qualify every column.")

    unq, unk = _find_unqualified_or_unknown(sql, tables_short, alias_map, catalog)
    if unq:
        logger.info(f"[FASTAPIMCP] UNQUALIFIED COLUMNS: {unq}")
        return ("ERROR: Unqualified column(s) detected: "
                f"{unq}. Always table-qualify every column (e.g., sales.value, person.name).")
    if unk:
        logger.info(f"[FASTAPIMCP] UNKNOWN COLUMNS: {unk}")
        return ("ERROR: Unknown column name(s) referenced: "
                f"{unk}. Check SCHEMA for the actual column names.")
    return None

# ===== REST endpoints (operation_id defines MCP tool names)

@app.get("/tables", operation_id="sql.list_tables")
def list_tables():
    logger.info("[FASTAPIMCP] LIST TABLES")
    try:
        cat = get_catalog()
        logger.info(f"[FASTAPIMCP] CATALOG: {cat}")
        return cat["tables"]
    except Exception as e:
        logger.info(f"[FASTAPIMCP] EXCEPTION IN LIST TABLES: {e}")
        return {"error": str(e)}

# --- add near your other module-level caches
SCHEMA_TTL_SECS = 300
_schema_cache_cols: Dict[str, List[str]] = {}
_schema_cache_at: Dict[str, float] = {}

@app.get("/schema/{table_name}", operation_id="sql.schema")
def get_schema(table_name: str):
    logger.info(f"[FASTAPIMCP] GET SCHEMA for {table_name}")
    try:
        # serve from cache if fresh
        now = time.time()
        if (table_name in _schema_cache_cols
            and (now - _schema_cache_at.get(table_name, 0.0)) < SCHEMA_TTL_SECS):
            cols = _schema_cache_cols[table_name]
            return {"schema": [f"{c} (cached)"], "columns": cols, "cached": True}

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
            logger.info(f"[FASTAPIMCP] UNKNOWN TABLE: {table_name}")
            return {"error": f"Unknown table '{table_name}'."}

        schema = [f"{col} ({dtype})" for col, dtype in rows]
        cols = [col for col, _dtype in rows]

        # update cache
        _schema_cache_cols[table_name] = cols
        _schema_cache_at[table_name] = now
        logger.info(f"[FASTAPIMCP] CACHED SCHEMA for {table_name}: {schema}")
        return {"schema": schema, "columns": cols, "cached": False}
    except Exception as e:
        logger.info(f"[FASTAPIMCP] EXCEPTION IN GET SCHEMA: {e}")
        return {"error": str(e)}
    
# @app.get("/schema/{table_name}", operation_id="sql.schema")
# def get_schema(table_name: str):
#     logger.info(f"[FASTAPIMCP] GET SCHEMA for {table_name}")
#     try:
#         # serve from cache if fresh
#         now = time.time()
#         if (table_name in _schema_cache_cols
#             and (now - _schema_cache_at.get(table_name, 0.0)) < SCHEMA_TTL_SECS):
#             cols = _schema_cache_cols[table_name]
#             logger.info(f"[FASTAPIMCP] Returning cached schema for {table_name}")
#             return {
#                 "schema": [f"{c} (cached)" for c in cols],  # ✅ Fixed: iterate cols
#                 "columns": cols,
#                 "cached": True
#             }

#         conn = get_connection()
#         try:
#             cur = conn.cursor()
#             cur.execute("""
#                 SELECT column_name, data_type
#                 FROM information_schema.columns
#                 WHERE table_schema='public' AND table_name = %s
#                 ORDER BY ordinal_position
#             """, (table_name,))
#             rows = cur.fetchall()
#             cur.close()
            
#             if not rows:
#                 logger.warning(f"[FASTAPIMCP] UNKNOWN TABLE: {table_name}")
#                 return {
#                     "error": f"Unknown table '{table_name}'.",
#                     "schema": [],
#                     "columns": [],
#                     "cached": False
#                 }

#             schema = [f"{col} ({dtype})" for col, dtype in rows]
#             cols = [col for col, _dtype in rows]

#             # update cache
#             _schema_cache_cols[table_name] = cols
#             _schema_cache_at[table_name] = now
#             logger.info(f"[FASTAPIMCP] CACHED SCHEMA for {table_name}: {schema}")
            
#             return {
#                 "schema": schema,
#                 "columns": cols,
#                 "cached": False
#             }
#         finally:
#             conn.close()  # ✅ Ensure connection is closed
            
#     except Exception as e:
#         logger.error(f"[FASTAPIMCP] EXCEPTION IN GET SCHEMA for {table_name}: {e}", exc_info=True)
#         # ✅ Return proper error response
#         return {
#             "error": f"Failed to fetch schema for '{table_name}': {str(e)}",
#             "schema": [],
#             "columns": [],
#             "cached": False
#         }


@app.get("/example/{table_name}", operation_id="sql.example")
def get_example_row(table_name: str):
    logger.info(f"[FASTAPIMCP] GET EXAMPLE ROW for {table_name}")
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
        logger.info(f"[FASTAPIMCP] EXAMPLE ROW for {table_name}: {row}")
        return {"example": dict(zip(columns, row)) if row else {}}
    except Exception as e:
        logger.info(f"[FASTAPIMCP] EXCEPTION IN GET EXAMPLE ROW: {e}")
        return {"error": str(e)}

@app.post("/check", operation_id="sql.check")
def check_sql(req: CheckRequest):
    logger.info(f"[FASTAPIMCP] CHECK SQL: {req.sql}")
    err = validate_sql(req.sql)
    return {"status": "OK" if not err else "ERROR", "message": err or "OK"}

@app.post("/execute", operation_id="sql.query")
def execute_query(request: SQLRequest):
    logger.info(f"[FASTAPIMCP] EXECUTE QUERY: {request.sql}")
    results = []
    try:
        conn = get_connection(); cur = conn.cursor()
        logger.info(f"[FASTAPIMCP] CONNECTION ESTABLISHED")
        for sql_text in request.sql:
            logger.info(f"[FASTAPIMCP] VALIDATE AND EXECUTE SQL: {sql_text}")
            err = validate_sql(sql_text)
            logger.info(f"[FASTAPIMCP] VALIDATION RESULT: {err or 'OK'}")
            if err:
                cur.close(); conn.close()
                return {"error": err}
            cur.execute(sql_text)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            logger.info(f"[FASTAPIMCP] QUERY RETURNED {len(rows)} ROWS, COLS: {cols}")
            results.append([dict(zip(cols, r)) for r in rows])
        logger.info(f"[FASTAPIMCP] ALL QUERIES EXECUTED")
        cur.close(); conn.close()
        logger.info(f"[FASTAPIMCP] QUERY RESULTS: {results}")
        return {"results": results}
    except Exception as e:
        logger.exception(f"[FASTAPIMCP] EXCEPTION DURING QUERY")
        return {"error": str(e)}

# ===== Optional RAG passthrough (disabled by default)
# @app.post("/panel_intents", operation_id="rag.panel_intents")
# def panel_intents_endpoint(req: PanelIntentsRequest):
#     if get_panel_intents is None:
#         return {"error": "RAG module not available"}
#     return get_panel_intents(req.prompt)

# ===== MCP exposure (no register_tool; tools come from operation_id)
logger.info("[FASTAPIMCP] MOUNT MCP")
mcp = FastApiMCP(app)
logger.info("[FASTAPIMCP] MOUNT HTTP")
mcp.mount_http()
logger.info("[FASTAPIMCP] READY")
