
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
import os, re, json, ast, time, asyncio, requests, logging, logging.config, httpx
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable, Type, Literal
from contextlib import suppress
from src.metrics import record_metric, time_call
from src.models import DashboardSpec
# ----------------------------
# Logging
# ----------------------------
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain_core.globals import set_debug, set_verbose

# keep terminal prints if you like
set_debug(True)

LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(LOG_DIR, exist_ok=True)

# <-- create callbacks AFTER LOG_DIR exists
LC_FILE_CB = FileCallbackHandler(os.path.join(LOG_DIR, "app.log"))  # write LC trace to app.log
LC_STDOUT_CB = StdOutCallbackHandler()

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)s %(name)s - %(message)s"}
    },
    "handlers": {
        "rotating_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": os.path.join(LOG_DIR, "app.log"),
            "maxBytes": 5_000_000,
            "backupCount": 5,
            "encoding": "utf-8",
            "formatter": "default",
            "level": "DEBUG",      # <-- add this
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",      # <-- and this
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "rotating_file"],
    },
    "loggers": {
        # These are the LangChain namespaces that actually emit logs
        "langchain":            {"level": "DEBUG", "handlers": ["console","rotating_file"], "propagate": False},
        "langchain_core":       {"level": "DEBUG", "handlers": ["console","rotating_file"], "propagate": False},
        "langchain_community":  {"level": "DEBUG", "handlers": ["console","rotating_file"], "propagate": False},
        # (optional) if you use LangGraph
        # "langgraph":            {"level": "DEBUG", "handlers": ["console","rotating_file"], "propagate": False},
    },
}
logging.config.dictConfig(LOGGING)

logger = logging.getLogger(__name__)

# ----------------------------
# LangChain / LLM
# ----------------------------
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import BaseTool
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda
import langchain
langchain.debug = True
# logging.getLogger("langchain").setLevel(logging.WARNING)

try:
    from langchain_core.tools import ToolException
except Exception:
    from langchain.tools.base import ToolException  # type: ignore

# ----------------------------
# App config
# ----------------------------
app = FastAPI()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_HTTP_URL = os.getenv("OLLAMA_HTTP_URL") or f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_JSON_TIMEOUT") or os.getenv("OLLAMA_TIMEOUT", "300"))
MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8000")
MAX_PANELS = int(os.getenv("MAX_PANELS", "10"))

# ----------------------------
# HTTP MCP client tools
# ----------------------------

MCP_HTTP_URL = os.getenv("MCP_HTTP_URL", "http://code-pg:8000")  # service name/port for code-pg.py

class HttpListTables:
    async def arun(self, _):
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{MCP_HTTP_URL}/tables")
            # ListTablesGuard expects a JSON string
            return json.dumps({"tables": r.json()} if isinstance(r.json(), list) else r.json())

class HttpSchema:
    async def arun(self, payload):
        table = (payload or {}).get("table_name")
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{MCP_HTTP_URL}/schema/{table}")
            return json.dumps(r.json())

class HttpExecute:
    async def arun(self, payload):
        sql = (payload or {}).get("sql")
        sql_list = [sql] if isinstance(sql, str) else sql
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{MCP_HTTP_URL}/execute", json={"sql": sql_list})
            return json.dumps(r.json())

# ----------------------------
# Default system text
# ----------------------------
DEFAULT_SYSTEM_TEXT = (
    "You are a SQL analysis agent with MCP tools:\n"
    "- sql.list_tables()\n"
    "- sql.schema(table_name)\n"
    "- sql.query(sql)\n\n"
    "{tools}\n\n"
    "Valid tool names: {tool_names}\n\n"
    "DE-DUP RULE (RUNTIME)\n"
    "- Never call sql.schema twice for the same table in one run.\n"
    "- If a schema is already present in the agent_scratchpad OR the tool returns {{\"cached\": true}}, skip additional calls for that table.\n"
    "- Call sql.list_tables at most once and ONLY as the first step if table discovery is truly required.\n\n"
    "QUALIFICATION POLICY (HARD RULE):\n"
    "- Always table-qualify every column in SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY (e.g., pe.city, s.value).\n"
    "- Aliases (e.g., AS city) are for display only; do not use them in GROUP BY/HAVING.\n\n"
    "TIME GROUPING (HARD RULE):\n"
    "- Never reference a bare column named \"year\".\n"
    "- If you need yearly grouping, derive a label and reuse it:\n"
    "SELECT date_trunc('year', s.sale_date) AS year, SUM(s.value) AS total\n"
    "FROM sales s\n"
    "GROUP BY date_trunc('year', s.sale_date)\n"
    "- For monthly/weekly/daily, use date_trunc('month'|'week'|'day') similarly.\n"
    "- GROUP BY the original expression (e.g., GROUP BY date_trunc('year', s.sale_date)).\n"
    "FORMAT HARD RULE:\n"
    "- After every \"Thought:\" you MUST immediately write either:\n"
    "  a) \"Action:\" followed by a valid tool call and \"Action Input: {{…}}\"\n"
    "  b) \"Final Answer:\" followed by the final JSON on ONE LINE\n"
    "- NEVER include backticks, code fences, or extra prose.\n"
    "CHART COUNT CONTRACT (HARD RULES):\n"
    "- Determine the set of chart INTENTS explicitly requested by the human. Let N = number of distinct chart intents (do not invent new ones).\n"
    "- Execute at most N queries (one per intent), in the same order as the human request.\n"
    "- Final Answer MUST contain exactly N items in \"results\" (no more, no less).\n"
    "- EXTRA=NO is a HARD GATE.\n"
    "- RULE: Never mix Final Answer with Action/Observation in the same message.\n"
    "- If the user asks for more than MAX_PANELS (MAX_PANELS=10), finish with:\n"
    "  Final Answer: {{ \"error\": \"Requested X charts exceeds MAX_PANELS=10. Please reduce or split the request.\" }}\n\n"
    "TRACE FORMAT (CRITICAL)\n"
    "Thought: <brief reasoning> | Checklist: CAST=OK; AGG=OK; GBY=OK; EXTRA=NO\n"
    "Action: <one of: sql.list_tables, sql.schema, sql.query>\n"
    "Action Input: {{ \"…\": \"…\" }}\n"
    "Observation: <tool result or ERROR: …>\n"
    "Final Answer: {{ \"results\": [ {{ \"sql\":\"…\", \"viz\":{{ \"type\":\"<barchart|line|piechart|table|stat|gauge>\", \"format\":\"<table|time_series>\", \"title\":\"<short>\" }} }} ] }}\n\n"
    "------------------------------------------------------------\n"
    "SQL LINT CHECKLIST (MANDATORY in Thought line):\n"
    "- CAST=OK → non-aggregate label expressions in SELECT may use ::text for readability; do not require ::text in GROUP BY.\n"
    "- AGG=OK → never cast aggregates.\n"
    "- GBY=OK (FORMAT-AWARE — HARD RULES):\n"
    " - BAR/PIE (viz.type in {{barchart, piechart}}):\n"
    "  • SELECT exactly one non-aggregate label + ≥1 aggregate(s).\n"
    "  • GROUP BY the label alias (never by *_id or by the raw expression if an alias exists).\n"
    " - TIME SERIES (format == \"time_series\"):\n"
    "  • Build a time label: date_trunc('<year|quarter|month|week|day>', <ts>) AS tlabel.\n"
    "  • SELECT tlabel + ≥1 aggregate(s).\n"
    "  • GROUP BY tlabel. No other non-aggregate columns.\n"
    " - STAT/GAUGE (viz.type in {{stat, gauge}}):\n"
    "  • SELECT only aggregate expression(s).\n"
    "  • NO GROUP BY is allowed.\n"
    " - TABLE (viz.type == \"table\"):\n"
    "  • If the intent is an aggregated table: follow BAR/PIE or TIME SERIES rules above.\n"
    "  • If the intent is a raw listing (TABLE VIEW POLICY): no aggregates, NO GROUP BY; join FKs to human labels and select explicit columns.\n"
    " - UNIVERSAL:\n"
    "  • Every non-aggregate in SELECT must appear in GROUP BY (except in STAT/GAUGE or raw listing where no non-aggregates are aggregated).\n"
    "- FK=OK → Final SQL SELECT contains NO raw \"id\" or \"*_id\" columns; they were replaced by label aliases; GROUP BY uses the label alias.\n"
    "- EXTRA=NO → there are no auxiliary/duplicate queries for the same intent.\n\n"
    "SELECT COLUMN POLICY (HUMAN-FRIENDLY OUTPUT) — HARD REQUIREMENT:\n"
    "- Do NOT select raw primary key or foreign key columns (named \"id\" or ending with \"_id\") in SELECT.\n"
    "- JOIN to referenced tables and select a human-readable field with ::text and a clear alias:\n"
    " s.person_id → FROM sales s JOIN person pe ON s.person_id=pe.id → pe.name::text AS person_name\n" 
    " s.product_id → FROM sales s JOIN products pr ON s.product_id=pr.id → pr.name::text AS product_name\n" 
    " o.customer_id → FROM orders o JOIN person p ON o.customer_id=p.id → p.name::text AS customer_name\n" 
    " o.deliverer_id → FROM orders o JOIN person p ON o.deliverer_id=p.id → p.name::text AS deliverer_name\n" 
    " s.payment_id → FROM sales s JOIN paymenttype pt ON s.payment_id=pt.id → pt.method::text AS payment_type_method\n\n"
    "TABLE VIEW POLICY (HARD RULE):\n"
    "1. NEVER use SELECT *.\n"
    "2. When listing all information for a sales record, you MUST replace foreign key IDs with human-readable names using LEFT JOINs.\n"
    "3. Specifically, replace 'person_id' with 'name' from 'person' table (alias person_name), 'product_id' with 'name' from 'products' (alias product_name).\n"
    "4. Select every remaining column from main table explicitly, fully qualified (e.g. s.value, s.sale_date).\n"
    "5. Example:\n"
    "   SELECT s.sale_date, s.value, pr.name AS product_name, pe.name AS person_name, pt.method::text AS payment_type_method\n"
    "   FROM sales s\n"
    "   LEFT JOIN products pr ON s.product_id = pr.id\n"
    "   LEFT JOIN person pe ON s.person_id = pe.id\n"
    "   LEFT JOIN paymenttype pt ON s.payment_id = pt.id\n\n"
    "BAR/PIE CHART POLICY (HARD REQUIREMENT):\n"
    "- SELECT must include exactly one non-aggregate dimension = the user-requested grouping (e.g. product_name for “per products”).\n"
    "- GROUP BY must use that same dimension (via original expression or alias).\n"
    "- SELECT must also include at least one aggregate metric (e.g. AVG(s.value), SUM(s.value), COUNT(*)).\n"
    "- Do not substitute another dimension or metric during repair.\n\n"
    "DIMENSION CANONICALIZATION (PK) — HARD RULE:\n"
    "- Do NOT select raw primary key (e.g. o.id, s.id, p.id) in non-agretation columns in SELECT, GROUP BY and ORDER BY.\n\n"
    "DIMENSION CANONICALIZATION (FK→LABEL) — HARD RULE:\n"
    "- If the requested dimension is “products”:\n"
    "   SELECT pr.name::text AS product_name, AVG(s.value) AS avg_value\n"
    "   FROM sales s\n"
    "   LEFT JOIN products pr ON s.product_id = pr.id\n"
    "   GROUP BY pr.name\n"
    "- For persons, payments, etc., follow the same pattern: always replace *_id with the human-readable name in both SELECT and GROUP BY.\n\n"
    "RETRY CORRECTION CONTRACT (HARD RULES):\n"
    "- LOCK the intent’s metric and dimension after the first draft SQL.\n"
    "   * Metric LOCK = the aggregate function + target (e.g. AVG(s.value))\n"
    "   * Dimension LOCK = the grouping key (e.g. product_name)\n"
    "- On any SQL error, you may only make the smallest schema-aligned syntactic changes needed to respect the LOCKs:\n"
    "   * You may replace a raw *_id in SELECT with the corresponding label, adjusting GROUP BY accordingly.\n"
    "   * You may add the missing non-aggregate dimension to GROUP BY if SELECT includes it.\n"
    "   * You may correct mis-qualified columns or add necessary JOINs for that dimension.\n"
    "   * You may NOT change the aggregate (e.g. AVG → SUM or COUNT), or change to a different grouping dimension.\n"
    "- At most ONE retry per intent is allowed.\n\n"
    "SCHEMA-STRICT MODE:\n"
    "- Only reference columns you personally saw via sql.schema() this run.\n"
    "- When a foreign-key → label mapping exists (e.g. product_id → pr.name), prefer the label in SELECT and GROUP BY.\n\n"
    "ERROR BEHAVIOR:\n"
    "- If an Observation starts with \"ERROR:\" while executing SQL, apply the Retry Correction Contract above and retry exactly once.\n"
    "- If a tool indicates a cached schema, do not treat it as an error.\n\n"
    "FINAL OUTPUT SHAPE (ENFORCEMENT):\n"
    "- On success: one line:\n"
    "   Final Answer: {{ \"results\": [ {{ \"sql\":\"…\", \"viz\":{{ \"type\":\"...\", \"format\":\"...\", \"title\":\"...\" }} }} , … ] }}\n"
    "- The number of items in \"results\" must equal N.\n"
    "- On error: one line:\n"
    "   Final Answer: {{ \"error\":\"… clear message …\" }}\n"
)



# ----------------------------
# Output parsing (StructuredOutputParser)
# ----------------------------
response_schemas = [
    ResponseSchema(
        name="results",
        description=(
            "A JSON array with EXACTLY N items (N = number of chart intents). "
            "Each item is an object: {"
            '"sql": string (the full SQL to run), '
            '"viz": {'
            '"type": "barchart|line|piechart|table|stat|gauge", '
            '"format": "table|time_series", '
            '"title": string'
            "}"
            "}."
        ),
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
FORMAT_INSTRUCTIONS = parser.get_format_instructions()

# ----------------------------
# Minimal models
# ----------------------------
class PromptRequest(BaseModel):
    prompt: str
    system_text: Optional[str] = Field(
        default=None,
        description="Full ReAct system prompt controlling tool order and output"
    )




class ExecArgs(BaseModel):
    sql: str = Field(..., description="Full SQL to run")
    chart_type: Optional[str] = Field(default=None)
    format: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)

# ----------------------------
# Utility helpers
# ----------------------------
def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _short(obj: Any, maxlen: int = 600) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False) if isinstance(obj, (dict, list)) else str(obj)
    except Exception:
        s = str(obj)
    return (s[:maxlen] + "…") if len(s) > maxlen else s

def _remove_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s[: s.rfind("```")].rstrip()
    return s.strip()

def _parse_final_answer_obj(text: str) -> Optional[dict]:
    """Extract and parse the FIRST top-level JSON object after 'Final Answer:'."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    idx = s.lower().find("final answer:")
    if idx != -1:
        s = s[idx + len("final answer:") :].lstrip()
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    end: Optional[int] = None
    in_str = False
    esc = False
    for i, ch in enumerate(s[start:], start):
        if esc:
            esc = False; continue
        if ch == "\\":
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1; break
    candidate = s[start:end] if end else s[start:]
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            try:
                val = ast.literal_eval(candidate)
                if isinstance(val, dict):
                    return val
                return None
            except Exception:
                return None

def _sql_fingerprint(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    s = sql.strip().rstrip(";")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    return s.lower()

def _normalize_result_item(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    sql = item.get("sql")
    if isinstance(sql, list):
        sql = " ".join(str(x) for x in sql)
    if not isinstance(sql, str) or not sql.strip():
        return None
    viz = item.get("viz") or {}
    if not isinstance(viz, dict):
        viz = {}
    viz.setdefault("type", item.get("chart_type","table"))
    viz.setdefault("format", item.get("format","table"))
    viz.setdefault("title", item.get("title","Result"))
    return {"sql": sql.strip(), "viz": viz}

# ----------------------------
# Intent parsing (from your previous code, simplified but compatible)
# ----------------------------
SYNONYM_MAPPINGS: Dict[str, str] = {
    "customer": "person", "user": "person",
    "product": "products", "item": "products",
    "sale": "sales", "order": "orders",
    "payment method": "paymenttype",
}
def normalize_synonyms(text: str) -> str:
    keys = set(SYNONYM_MAPPINGS.keys()) | {k + "s" for k in SYNONYM_MAPPINGS.keys()}
    if not text: return ""
    pattern = rf"(?<!\w)({'|'.join(map(re.escape, sorted(keys, key=len, reverse=True)))})(?!\w)"
    def to_singular(token: str) -> str:
        t = token.lower()
        return t[:-1] if t.endswith("s") and t[:-1] in SYNONYM_MAPPINGS else t
    def repl(m):
        tok = m.group(1)
        base = to_singular(tok)
        out = SYNONYM_MAPPINGS.get(base, tok)
        if tok.lower().endswith("s") and not out.lower().endswith("s"):
            out = (out[:-1] + "ies") if out.endswith("y") else (out + "s")
        return out if tok.islower() else out.capitalize() if tok.istitle() else out.upper()
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

_CHART_SYNONYMS = {
    "bar":"barchart","bar chart":"barchart","bars":"barchart",
    "pie":"piechart","pie chart":"piechart",
    "line":"line","line chart":"line",
    "table":"table","stat":"stat","metric":"stat","single value":"stat","kpi":"stat",
    "gauge":"gauge"
}
def _canon_chart_type(raw: str) -> Optional[str]:
    t = (raw or "").strip().lower()
    return _CHART_SYNONYMS.get(t) or _CHART_SYNONYMS.get(t.replace("-", " "))

_TIME_WORDS = {
    "annually": ("Annual", "year"), "annual": ("Annual","year"),
    "yearly": ("Annual","year"), "monthly": ("Monthly","month"),
    "weekly": ("Weekly","week"), "daily": ("Daily","day"),
    "quarterly": ("Quarterly","quarter")
}
def _detect_time_label(phrase: str) -> Optional[str]:
    p = phrase.lower()
    for k,(label,_) in _TIME_WORDS.items():
        if re.search(rf"\b{k}\b", p): return label
    return None

def _clean_phrase(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,.;:]+$", "", s)
    return s.strip()

def _extract_chart_type_segment(seg: str) -> Optional[str]:
    m = re.search(r"\b(?:as|in)\s+(?:a\s+|an\s+)?([a-z-]+(?:\s+chart)?)\b", seg, flags=re.I)
    if not m: return None
    raw = m.group(1).strip().lower().replace(" chart","")
    return _canon_chart_type(raw)

def _has_metric(phrase: str) -> bool:
    p = phrase.lower()
    if _detect_time_label(phrase):  # annually, monthly, etc.
        return True
    if re.search(r"\bsales?\b", p):  # treat "sales" itself as a metric-like target
        return True
    return bool(re.search(r"\b(sum|total|count|number|avg|average|mean|median|max|min)\b", p))


def _guess_grouping(phrase: str) -> Optional[str]:
    p = phrase.lower()
    m = re.search(r"\b(?:per|by)\s+([a-z_][a-z0-9_\s-]{1,40})\b", p)
    if m:
        grp = m.group(1)
        grp = re.sub(r"\b(as|and|the|a|an)\b.*$", "", grp).strip()
        grp = re.sub(r"[^a-z0-9\s_-]", "", grp).strip().replace("_"," ")
        return grp if grp else None
    tl = _detect_time_label(phrase)
    return tl

def _title_case(s: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"\s+", s.strip()))

def extract_intents_from_prompt(text: str) -> List[Dict[str, str]]:
    if not isinstance(text, str): return []
    t = normalize_synonyms(text)
    parts = re.split(r"\s+(?:and|&)\s+|[,;]\s*", t, flags=re.I)
    intents: List[Dict[str, str]] = []
    seen = set()
    for raw_seg in parts:
        seg = _clean_phrase(raw_seg)
        if len(seg) < 6: continue
        chart = _extract_chart_type_segment(seg)
        if not chart: continue
        if chart != "stat" and not _has_metric(seg): continue
        grouping = _guess_grouping(seg)
        metric_part = re.split(r"\b(?:as|in)\b\s+(?:a\s+|an\s+)?", seg, flags=re.I)[0]
        title = _title_case(metric_part if not grouping or grouping in metric_part.lower()
                            else f"{metric_part} by {grouping}")
        key = (title.lower(), chart)
        if key in seen: continue
        seen.add(key)
        intents.append({"title": title, "chart_type": chart, "note": seg})
    return intents

# ADD: tolerant parser that accepts Final Answer:, fenced code, or raw JSON
def _parse_any_json(text: str) -> Optional[dict]:
    if not isinstance(text, str) or not text.strip():
        return None
    s = text.strip()

    # try your existing tolerant "Final Answer:" extractor first
    fa = _parse_final_answer_obj(s)
    if isinstance(fa, dict):
        return fa

    # strip code fences (keep largest fenced block first)
    s = s.replace("\r", "")
    if "```" in s:
        blocks = []
        parts = s.split("```")
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.lower().startswith("json\n"):
                part = part.split("\n", 1)[1].strip()
            blocks.append(part)
        for b in sorted(blocks, key=len, reverse=True):
            try:
                return json.loads(b)
            except Exception:
                pass

    # last chance: find first top-level JSON object anywhere
    start = s.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        end = None
        for i, ch in enumerate(s[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end:
            candidate = s[start:end]
            for attempt in (candidate, candidate.replace("'", '"')):
                with suppress(Exception):
                    return json.loads(attempt)
            with suppress(Exception):
                val = ast.literal_eval(candidate)
                if isinstance(val, dict):
                    return val
    return None

# ----------------------------
# Stub MCP client placeholders (we keep tools as async callables)
# ----------------------------

class ListTablesGuard(BaseTool):
    name: str = "sql.list_tables"
    description: str = "List available tables. Accepts NO parameters."
    inner_tool: Any = None
    _called_once: bool = False
    _memo: Any = None
    def reset_run(self): self._called_once=False; self._memo=None
    def _run(self, *args, **kwargs): return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        if not self.inner_tool:
            print("⚠️ [ListTablesGuard] No inner_tool; using demo data")
            return json.dumps({"tables":["sales","person","products","paymenttype","orders"]})
        if not self._called_once:
            res = await self.inner_tool.arun({})
            self._called_once = True; self._memo = res; return res
        return self._memo

class GetTableSchemaGuard(BaseTool):
    name: str = "sql.schema"
    description: str = "Get table schema: {'table_name':'...'}"
    inner_tool: Any = None
    list_tool: Optional[Any] = None
    _available_tables_cache: Optional[List[str]] = None
    _memo_cache: Dict[str, str] = {}
    _seen_this_run: set = set()
    def reset_run(self):
        self._available_tables_cache=None; self._memo_cache={}; self._seen_this_run=set()
    def _run(self, *args, **kwargs): return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        ti = kwargs.get("tool_input", None) or (args[0] if args else None)
        if isinstance(ti, dict):
            table_name = ti.get("table_name")
        elif isinstance(ti, str):
            try: table_name = json.loads(ti).get("table_name")
            except Exception: table_name = ti
        else:
            return "ERROR: get_table_schema requires table_name"
        if not isinstance(table_name, str) or not table_name.strip():
            return "ERROR: get_table_schema requires table_name"
        table_name = table_name.strip()
        if table_name in self._seen_this_run and table_name in self._memo_cache:
            obj = json.loads(self._memo_cache[table_name])
            obj["cached"] = True
            return json.dumps(obj)
        if self.inner_tool:
            raw = await self.inner_tool.arun({"table_name": table_name})
            try: data = json.loads(raw) if isinstance(raw, str) else raw
            except Exception: data = raw
            if isinstance(data, dict) and ("columns" in data or "schema" in data):
                cols = data.get("columns") or []
                schema = data.get("schema") or cols
                out = {"schema": schema, "columns": cols, "cached": False}
            else:
                out = {"schema": [], "columns": [], "cached": False}
        else:
            demo = {
                "sales": ["id","person_id","product_id","sale_date","value","payment_id","order_id"],
                "person": ["id","name","email","city","role"],
                "products": ["id","name","category"],
                "paymenttype": ["id","method"],
                "orders": ["id","number"]
            }
            cols = demo.get(table_name.lower(), [])
            if not cols:
                return f"ERROR: Unknown table '{table_name}'"
            out = {"schema": [c+" (text)" for c in cols], "columns": cols, "cached": False}
        self._seen_this_run.add(table_name)
        self._memo_cache[table_name] = json.dumps(out)
        return json.dumps(out)


class ExecuteQueryGuard(BaseTool):
    name: str = "sql.query"
    description: str = "Execute a SQL query with optional viz hints."
    args_schema: Type[BaseModel] = ExecArgs
    inner_tool: Any = None        # wire your real executor here
    schema_tool: Optional[Any] = None
    _result_cache: Dict[Tuple[str, str, str], Any] = {}

    PREVIEW_LIMIT: int = 20

    def reset_run(self):
        self._result_cache = {}

    def _run(self, *args, **kwargs):
        return asyncio.run(self._arun(*args, **kwargs))

    async def _arun(self, *args, **kwargs):
        ti = kwargs.get("tool_input", None) or (args[0] if args else None)
        if isinstance(ti, dict):
            payload = dict(ti)
        elif isinstance(ti, str):
            try:
                payload = json.loads(ti)
                print(f"[ExecuteQueryGuard] Parsed JSON input: {_short(payload)}")
            except Exception:
                payload = {"sql": ti}
                print(f"[ExecuteQueryGuard] Using raw string input as SQL")
        else:
            return "ERROR: invalid input for sql.query"

        sql = (payload.get("sql") or "").strip()
        title = (payload.get("title") or "").strip()
        if not sql or "select" not in sql.lower():
            return json.dumps({"status":"error","message":"Missing or invalid SQL"})

        ctype = (payload.get("chart_type") or "table").lower().strip()
        fmt = (payload.get("format") or ("time_series" if ctype in ("line", "timeseries") else "table")).lower()
        key = (_sql_fingerprint(sql), ctype, fmt)
        if key in self._result_cache:
            return self._result_cache[key]

        # ---- run query (no hardcoded demo data) ----
        try:
            if self.inner_tool:
                raw = await self.inner_tool.arun(payload)  # pass-through
            else:
                # Strict behavior by default: surface a clear error
                if os.getenv("SQL_QUERY_NOOP_OK", "").lower() in ("1", "true", "yes"):
                    # Optional soft-noop (returns a structured “no executor” observation)
                    norm = {
                        "status": "noop",
                        "columns": [],
                        "rows": 0,
                        "data_preview": [],
                        "message": "sql.query has no inner_tool configured; no query executed."
                    }
                    out = json.dumps(norm, ensure_ascii=False)
                    self._result_cache[key] = out
                    print(f"[ExecuteQueryGuard] No executor; noop for SQL: {_short(sql)}")
                    return out
                # Default: error
                return json.dumps({"status":"error","message":"sql.query has no inner_tool configured; cannot execute SQL."})

        except Exception as e:
            return json.dumps({"status":"error","message": str(e)})


        # ---- normalize common shapes ----
        def normalize(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except Exception:
                    return {
                        "status": "ok",
                        "columns": ["text"],
                        "rows": 1,
                        "data_preview": [{"text": obj}],
                    }

            if isinstance(obj, dict) and "columns" in obj and "rows" in obj and isinstance(obj["rows"], list):
                cols = obj.get("columns") or []
                rows = obj.get("rows") or []
                preview = []
                for r in rows[: self.PREVIEW_LIMIT]:
                    if isinstance(r, dict):
                        preview.append(r)
                    elif isinstance(r, list):
                        preview.append({c: r[i] if i < len(r) else None for i, c in enumerate(cols)})
                    else:
                        preview.append({"value": r})
                return {
                    "status": obj.get("status", "ok"),
                    "columns": cols,
                    "rows": len(rows),
                    "data_preview": preview,
                }

            if isinstance(obj, list) and (not obj or isinstance(obj[0], dict)):
                cols = list(obj[0].keys()) if obj else []
                return {
                    "status": "ok",
                    "columns": cols,
                    "rows": len(obj),
                    "data_preview": obj[: self.PREVIEW_LIMIT],
                }

            if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list) and obj["results"]:
                first = obj["results"][0]
                return normalize(first)

            return {
                "status": "ok",
                "columns": [],
                "rows": 0,
                "data_preview": [obj],
            }

        norm = normalize(raw)
        out = json.dumps(norm, ensure_ascii=False)
        self._result_cache[key] = out
        
        logger.info(
            "[ExecuteQueryGuard]\n title = %s \n SQL: %s \n Rows=%s \n Preview=%s",
            title,
            sql,
            norm.get("rows"),
            json.dumps( norm.get("data_preview", [])[:3], ensure_ascii=False)
        )
        print(f"[ExecuteQueryGuard] Executed SQL: {_short(sql)} => preview_rows={norm.get('rows')}")
        
        return out


class ToolAliasNoInput(BaseTool):
    name: str
    description: str = "Alias tool that forwards to another tool"
    target_tool: Any
    def reset_run(self):
        if hasattr(self.target_tool, "reset_run"):
            self.target_tool.reset_run()
    def _run(self, *args, **kwargs): return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        print(f"[ToolAliasNoInput] Calling {self.target_tool.name}")
        return await self.target_tool.arun({})

class ToolAliasPassthrough(BaseTool):
    name: str
    description: str = "Alias tool that forwards to another tool"
    target_tool: Any
    def reset_run(self): pass
    def _run(self, *args, **kwargs): return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        ti = kwargs.get("tool_input", None) or (args[0] if args else None)
        print(f"[ToolAliasPassthrough] Calling {self.target_tool.name} with input: {_short(ti)}")
        return await self.target_tool.arun(tool_input=ti)

def generate_dashboard_with_ollama(
    sqls: List[str],
    chart_types: List[str],
    titles: List[str],
    dashboard_title: str,
    time_from: str = "now-5y",
    time_to: str = "now"
) -> dict:
    """
    Given lists of SQLs, chart types, and panel titles, asks Ollama to assemble a full Grafana dashboard JSON.
    Works with plain Ollama /api/chat (e.g., llama3.1:8b). Returns the parsed dashboard dict.
    """

    # Build panel specs for the prompt (LLM must not change SQL)
    panels = []
    for sql, ctype, title in zip(sqls, chart_types, titles):
        # Grafana format hint
        fmt = "table" if (ctype or "").lower() in ("barchart", "piechart", "table", "stat", "gauge") else "time_series"
        panels.append({
            "title": title,
            "type": (ctype or "table"),
            "format": fmt,
            "rawSql": sql,
        })
    # Create the JSON schema for the LLM to follow
    ts = now_timestamp()
    json_schema = {
        "dashboard": {
            "title": f"String with timestamp - {ts}",
            "schemaVersion": 36,
            "version": 1,
            "refresh": "5s",
            "time": {"from": "now-5y", "to": "now"},
            "timepicker": {
                "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"],
                "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d", "90d", "6M", "1y", "5y"]
            },
            "panels": [
                {
                    "id": "Integer - Panel ID",
                    "type": "String - Panel type (barchart, piechart, table, timeseries, stat)",
                    "title": "String - Panel title",
                    "datasource": {"type": "postgres", "uid": "feyd4obe5zb40b"},
                    "targets": [
                        {
                            "refId": "A",
                            "format": "String - time_series or table",
                            "rawSql": "String - PostgreSQL query"
                        }
                    ],
                    "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
                    "options": {
                        "tooltip": {"mode": "single"},
                        "legend": {"displayMode": "list", "placement": "bottom"},
                        "reduceOptions": {
                            "calcs": ["lastNotNull"],
                            "fields": "",
                            "values": True
                        }
                    }
                }
            ]
        },
        "overwrite": False
    }
    # System instructions kept lean and JSON-only
    system_msg = (
        "You are an expert in Grafana dashboard JSON (schemaVersion 36). "
        "Return EXACTLY one JSON object, no prose, no code fences. "
        "OUTPUT  JSON SCHEMA SHAPE TO FOLLOW (MANDATORY):\n"
        + json.dumps(json_schema, indent=2) + "\n"
        "PANEL JSON (MANDATORY — must match Grafana HTTP API expectations):\n"
        "- Each item in dashboard.panels[] MUST include:\n"
        "  id (int), gridPos {x,y,w,h}, title (str), type (str), targets [ {refId, rawSql, format} ]\n"
        "- Do NOT put rawSql at the panel root; it MUST be inside targets[0].\n"
        "- Example (single panel):\n"
        "{\n"
        '  "id": 0,\n'
        '  "title": "Panel Title",\n'
        '  "type": "stat",\n'
        '  "gridPos": {"x":0,"y":0,"w":12,"h":8},\n'
        '  "targets": [ {"refId":"A","rawSql":"SELECT 1","format":"table"} ]\n'
        "}\n"
        "LAYOUT (24-col grid):\n"
        "- Two panels per row: w=12, h=8; x=0 (left) or x=12 (right); y increases by 8 each row.\n"
        "- If N is odd, make the last panel full-width: w=24, x=0.\n"
        "Algorithm for panel i (0-based), total N:\n"
        "  if i == N-1 and N % 2 == 1:  x=0; y=(i//2)*8; w=24; h=8\n"
        "  else if i % 2 == 0:          x=0; y=(i//2)*8; w=12; h=8\n"
        "  else:                        x=12; y=(i//2)*8; w=12; h=8\n"
        "TARGETS:\n"
        "- For each panel create targets[0] = {\"refId\":\"A\",\"rawSql\":\"<SQL>\",\"format\":\"table|time_series\"}.\n"
        "- Use the SQL from the user message verbatim.\n"
        "STRICT JSON RULES:\n"
        "- Double quotes for all keys/strings; no trailing commas; no extra top-level keys; "
        "no additional quoting of values (use the value as-is)."
        "TITLE RULE:\n"
        "- Set dashboard.title EXACTLY to the provided 'dashboard_title' from the user message. Do not invent or copy the prompt.\n"

    )
    
    
    user_msg = json.dumps({
        "dashboard_title": dashboard_title,
        "time_from": time_from,
        "time_to": time_to,
        "panels": panels
    }, ensure_ascii=False, indent=2)

    # Ask Ollama (chat API)
    
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 4096,
            "num_ctx": 8192,
            "seed": 42
        },
        "keep_alive": "10m",
        "stream": False
    }


    resp = requests.post(OLLAMA_HTTP_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()

    # Ollama chat returns either a single message or streaming;
    # in non-streaming, we get {"message":{"content": "..."}}
    try:
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            raw = data["message"]["content"].strip()
        elif "choices" in data and data["choices"]:
            raw = (data["choices"][0]["message"]["content"] or "").strip()
        else:
            raw = (data.get("content") or "").strip()
    except json.JSONDecodeError:
        # Fallback for NDJSON streaming
        raw_chunks = []
        for line in resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = (obj.get("message") or {}).get("content")
            if isinstance(msg, str):
                raw_chunks.append(msg)
        raw = "".join(raw_chunks).strip()

    cleaned = _remove_code_fences(raw)
    # Some models prepend 'json' as a tag line
    if cleaned.lower().startswith("json\n"):
        cleaned = cleaned.split("\n", 1)[1].strip()
    
    # Try direct parse first
    try:
        dashboard = json.loads(cleaned)
        
    except Exception as e:
        print(f"⚠️ Could not parse Ollama JSON: {cleaned}")       
        raise RuntimeError(f"Ollama returned invalid JSON: {e}. Snippet: {cleaned}")
    
    return dashboard


def validate_with_pydantic(data: dict) -> Tuple[bool, str, Optional[DashboardSpec]]:
    try:
        
        spec = DashboardSpec(**data)
        
        return True, "", spec
    except ValidationError as e:
        print(f"[ERROR] Pydantic validation → {e}")
        return False, str(e), None
    
def _ensure_panel_formats(payload: dict) -> None:
    """Guarantee each panel target has 'format' (Grafana requires it)."""
    panels = (payload or {}).get("dashboard", {}).get("panels", [])
    for panel in panels:
        ptype = (panel.get("type") or "").lower()
        default_fmt = "time_series" if ptype in ("timeseries", "line") else "table"
        for t in panel.get("targets", []) or []:
            if isinstance(t, dict) and not t.get("format"):
                t["format"] = default_fmt

# ----------------------------
# intent prompt
# ----------------------------
ChartType = Literal["barchart","piechart","line","table","stat","gauge"]

class Intent(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    chart_type: ChartType
    note: str = Field(default="", min_length=0)

class IntentList(BaseModel):
    intents: List[Intent]
    
def _ollama_json(system: str, user: str) -> dict:
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "format": "json",
        "options": {"temperature": 0.0, "top_p": 1.0, "num_ctx": 8192},
        "stream": False,
    }
    r = requests.post(OLLAMA_HTTP_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    # Handle both normal and NDJSON-ish replies
    raw = (data.get("message", {}) or {}).get("content", "") or (data.get("content") or "")
    raw = str(raw).strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        logger.info(f"[ollama_json] Extracted JSON: {_short(raw, 1000)}")
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[ollama_json] JSON decode failed: {e}; raw={_short(raw, 1000)}")
        raise

INTENT_SYSTEM = (
    "You split one analytics request into multiple chart intents.\n"
    "Return ONLY a single JSON object with EXACTLY this shape and key names:\n"
    '{{"intents":[{{"title":"...", "chart_type":"barchart|piechart|line|table|stat|gauge", "note":"..."}}]}}\n'
    "RULES:\n"
    "- note MUST be a non-empty phrase (>= 3 chars) copied from the user's request segment for that intent.\n"
    "- No prose. No code fences. No extra keys."
)


INTENT_USER_TMPL = """Prompt:
{prompt}

Respond exactly as:
{{"intents":[{{"title":"...", "chart_type":"...", "note":"..."}}, ...]}}"""

def llm_extract_intents_ollama_http(prompt: str) -> List[dict]:
    try:
        obj = _ollama_json(INTENT_SYSTEM, INTENT_USER_TMPL.format(prompt=prompt))
        logger.info(f"[llm_extract_intents_ollama_http] input prompt: {prompt} Raw LLM output: {_short(obj)}")
    except requests.RequestException as e:
        logger.error(f"[intents/ollama] HTTP error: {e}")
        raise
    except Exception as e:
        logger.error(f"[intents/ollama] Parse error before validation: {e}")
        raise

    # ---- normalize top-level shape ----
    try:
        if isinstance(obj, list):
            obj = {"intents": obj}
        elif isinstance(obj, dict) and "intents" not in obj:
            for it in obj["intents"]:
                if not isinstance(it.get("note"), str) or len(it["note"].strip()) < 3:
                    # backfill with the title (or original prompt segment if you keep that)
                    it["note"] = it.get("title", "").strip()
            if {"title","chart_type","note"}.issubset(set(obj.keys())):
                obj = {"intents": [obj]}
            elif "intent" in obj and isinstance(obj["intent"], dict):
                obj = {"intents": [obj["intent"]]}
            elif "items" in obj and isinstance(obj["items"], list):
                obj = {"intents": obj["items"]}
            else:
                # Log what we got so we can refine later
                logger.error(f"[intents/ollama] Unexpected shape: {obj}")
                obj = {"intents": []}
    except Exception as e:
        logger.error(f"[intents/ollama] Normalization error: {e}; raw={obj}")
        obj = {"intents": []}

    # ---- validate ----
    try:
        validated = IntentList(**obj)
    except ValidationError as ve:
        logger.error(f"[intents/ollama] Validation failed: {ve}; raw_obj={_short(obj, 600)}")
        raise

    # ---- de-dupe ----
    out, seen = [], set()
    for it in validated.intents:
        key = (it.title.strip().lower(), it.chart_type)
        if key in seen: 
            continue
        seen.add(key)
        out.append({"title": it.title.strip(),
                    "chart_type": it.chart_type,
                    "note": it.note.strip()})
    return out



# ---------------------------
# Startup: configure LLM and base guards (with placeholders for MCP)
# ----------------------------
@app.on_event("startup")
async def startup_event():
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        top_p=1.0,
        model_kwargs={"keep_alive": "10m", "num_ctx": 4096},
        disable_streaming=True,
    )
    app.state.llm = llm

    # In a real setup, discover MCP tools here. We set inner_tool=None so our guards use demo fallbacks.
    list_guard  = ListTablesGuard(inner_tool=HttpListTables())
    schema_guard = GetTableSchemaGuard(inner_tool=HttpSchema())
    schema_guard.list_tool = list_guard
    exec_guard   = ExecuteQueryGuard(inner_tool=HttpExecute(), schema_tool=schema_guard)
    print(f"[startup] Guards initialized with inner tools: list={list_guard.inner_tool}, schema={schema_guard.inner_tool}, exec={exec_guard.inner_tool}")
    logger.info(f"[startup] Guards initialized with inner tools: list={list_guard.inner_tool}, schema={schema_guard.inner_tool}, exec={exec_guard.inner_tool}")
    app.state.list_guard = list_guard
    app.state.schema_guard = schema_guard
    app.state.exec_guard = exec_guard
    app.state.tools = {
        "sql.list_tables": list_guard,
        "sql.schema": schema_guard,
        "sql.query": exec_guard,
    }
    
    logger.info("[STARTUP] LLM and guards ready")
    print("[STARTUP] LLM and guards ready")
# ----------------------------
# /ask endpoint: per-intent ReAct agents
# ----------------------------
@app.post("/ask")
async def ask_sql(req: PromptRequest):
    ts = now_timestamp()
    start = time.perf_counter()
    
    raw_input = (req.prompt or "").strip()
    if not raw_input:
        return {"status": "error", "timestamp": ts, "summary": "Final Answer: " + json.dumps({"error":"Empty prompt"})}
    
    # system_text = (req.system_text or "").strip()
    system_text = (req.system_text or DEFAULT_SYSTEM_TEXT).strip()
    
    if not system_text:
        return {"status":"error","timestamp":ts,"summary":"Final Answer: " + json.dumps({"error":"Missing system_text"})}
    print(f"🧠 [ask_sql] Received prompt: {_short(raw_input)}")
    user_input = normalize_synonyms(raw_input)
    # intents = extract_intents_from_prompt(user_input)
    try:
        intents = llm_extract_intents_ollama_http(user_input)
        print(f"[ask_sql] LLM extracted {len(intents)} intents")    
    except Exception:
        intents = extract_intents_from_prompt(user_input)  
        print(f"[ask_sql] Fallback extracted {len(intents)} intents")

    N = len(intents)
    if N == 0:
        return {"status":"error","timestamp":ts,"summary":"Final Answer: " + json.dumps({"error":"No intents found."})}
    if N > MAX_PANELS:
        return {"status":"error","timestamp":ts,"summary":"Final Answer: " + json.dumps({"error":f"Requested {N} charts exceeds MAX_PANELS={MAX_PANELS}."})}
    logger.info(f"[ask_sql] {N} intents extracted: {[_short(i) for i in intents]}")
    print(f"[ask_sql] {N} intents extracted: {[_short(i) for i in intents]}")
    # Run one agent per intent concurrently
    tasks = []
    for intent in intents:
        intent_text = intent.get("note") or intent.get("title") or ""
        print(f"[ask_sql] Launching agent for intent: {intent_text}")
        logger.info(f"[ask_sql] Launching agent for intent: {intent_text}")
        # Fresh guards per agent (so memo/caches are per-run)
        inner_list = app.state.list_guard.inner_tool
        inner_schema = app.state.schema_guard.inner_tool
        inner_exec = app.state.exec_guard.inner_tool
        print(f"[ask_sql] Using inner tools: list={inner_list}, schema={inner_schema}, exec={inner_exec}")
        logger.info(f"[ask_sql] Using inner tools: list={inner_list}, schema={inner_schema}, exec={inner_exec}")
        list_guard = ListTablesGuard(inner_tool=inner_list)
        schema_guard = GetTableSchemaGuard(inner_tool=inner_schema)
        schema_guard.list_tool = list_guard
        exec_guard = ExecuteQueryGuard(inner_tool=inner_exec, schema_tool=schema_guard)

        agent_tools = []
        agent_tools.append(ToolAliasNoInput(name="sql.list_tables", target_tool=list_guard))
        agent_tools.append(ToolAliasPassthrough(name="sql.schema", target_tool=schema_guard))
        agent_tools.append(ToolAliasPassthrough(name="sql.query", target_tool=exec_guard))
        print(f"[ask_sql] Agent tools: {[t.name for t in agent_tools]}")
        logger.info(f"[ask_sql] Agent tools: {[t.name for t in agent_tools]}")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_text),
            ("system", "{tools}\n\nValid tool names: {tool_names}"),
            # ("system", "When you produce the Final Answer, follow this JSON schema:\n{format_instructions}"),
            ("system",
            "When you finish, emit exactly one line:\n"
            "Final Answer: {{\"results\": [ {{\"sql\": \"...\", \"viz\": {{\"type\":\"barchart|line|piechart|table|stat|gauge\", \"format\":\"table|time_series\", \"title\":\"...\"}} }} ] }}\n"
            "No code fences. No extra text. Do not say anything like 'Here is the corrected output:'."),
            ("human", "{input}\n\n{agent_scratchpad}")
        ])


        agent = create_react_agent(llm=app.state.llm, tools=agent_tools, prompt=prompt_template)

        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=agent_tools,
            verbose=True,
            max_iterations=12,            # ↓ lower to avoid long loops
            early_stopping_method="force",
            # return_intermediate_steps=False,
            handle_parsing_errors=(
                "Reformat your last message to comply with ReAct:\n"
                "It must be either:\n"
                '1) Thought: ...\\nAction: <tool name>\\nAction Input: { ... }\n'
                "or\n"
                '2) Final Answer: {"results":[{"sql":"...","viz":{"type":"...","format":"...","title":"..."}}]}\n'
                "No code fences. No extra text."
            ),
        )

        print(f"[ask_sql] Agent created with max_iterations={executor.max_iterations}")
        

        tasks.append(asyncio.create_task(executor.ainvoke({
            "input": intent_text,
            "tools": "\n".join(f"- {t.name}" for t in agent_tools),
            "tool_names": ", ".join(t.name for t in agent_tools),
            "format_instructions": FORMAT_INSTRUCTIONS,
        })))

        
                                               
    print(f"[ask_sql] Launched {len(tasks)} agents concurrently")
    logger.info(f"[ask_sql] Launched {len(tasks)} agents concurrently")
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results: List[Dict[str, Any]] = []
    for resp in responses:
        if isinstance(resp, Exception):
            logger.error(f"Agent error: {resp}")
            continue

        output = resp.get("output") if isinstance(resp, dict) else str(resp)
        if not output:
            logger.error("Empty agent output")
            continue

        # fa_obj = _parse_final_answer_obj(output)
        fa_obj = _parse_any_json(output) 
        if not isinstance(fa_obj, dict):
            logger.error(f"Could not parse JSON from agent output: {output[:300]}")
            continue

        items = fa_obj.get("results", [])
        if not isinstance(items, list):
            logger.error("Parse error: 'results' is not a list")
            continue

        for item in items:
            norm = _normalize_result_item(item)
            if norm:
                results.append(norm)


      
    print(f"[ask_sql] Collected {len(results)} raw results from agents")
    logger.info(f"[ask_sql] Collected {len(results)} raw results from agents")
    # Dedupe and cap to N
    seen=set(); dedup=[]
    for r in results:
        fp = _sql_fingerprint(r["sql"])
        if fp not in seen:
            seen.add(fp); 
            dedup.append(r)
    dedup = dedup[:N]
    
    seen_title = { (r.get("viz") or {}).get("title") or r.get("title") for r in dedup if (r.get("viz") or {}).get("title") or r.get("title") }
    
    short_title =  ", ".join(seen_title) 
    print(f"🖼️ titles: {short_title}")
    logger.info(f"[/ASK] Deduped final results:\n {dedup}")
    final_answer = {"results": dedup}
    sql_preview = ";\n".join([r["sql"] for r in dedup])
    logger.info(f"[/ASK] Final SQL preview:\n {sql_preview}")
    print("📝 Final Answer SQL preview:")
    print(sql_preview[:200] + ("..." if len(sql_preview) > 200 else ""))

    print(f"📝 Building dashboard JSON with {len(dedup)} panels (Ollama) + validate + push...")
        # --- NEW: execute final deduped SQLs and include normalized previews ---
    exec_inner = app.state.exec_guard.inner_tool  # HttpExecute()
    exec_guard_for_final = ExecuteQueryGuard(inner_tool=exec_inner)  # reuse normalizer/logging
    exec_outputs = []
    try:
        for r in dedup:
                   
            title = (r.get("viz") or {}).get("title") or "Result"
            ctype = (r.get("viz") or {}).get("type", "table")
            fmt   = (r.get("viz") or {}).get("format", "table")
            title = (r.get("viz") or {}).get("title") or "Result"
            logger.info(
                "[/ASK] Executing final:\n title: %s,\n SQL: %s",
                _short(title, 300),
                _short(r.get("sql",""), 300),
            )
            payload = {
                "sql": r["sql"],
                "chart_type": ctype,
                "format": fmt,
                "title": title,
            }
            raw = await exec_guard_for_final._arun(tool_input=payload)
            # normalize to dict for response
            try:
                norm = json.loads(raw) if isinstance(raw, str) else raw
                logger.info(f"[/ASK] Final exec raw output: {_short(norm)}")
            except Exception:
                norm = {"status": "ok", "columns": [], "rows": 0, "data_preview": [raw]}
            exec_outputs.append({
                "sql": r["sql"],
                "viz": r["viz"],
                "result": norm,
            })
        # mark success if we executed at least one
        logger.info(
            "[/ASK] Final exec outputs:\n %s",
            ", ".join(_short(o, 300) for o in exec_outputs)
        )
        success = len(exec_outputs) > 0        
    except Exception as e:
        logger.exception("Final execution of deduped SQL failed")
        error_msg = f"Final execution failed: {e}"
        success = False

    # prepare shared containers
    extra: Dict[str, Any] = {}
    fn_times: Dict[str, int] = {}
    # success = False
    # error_msg = ""

    # if dedup:  # we have at least one SQL/panel
    #     try:
    #         sqls = [r["sql"] for r in dedup]
    #         chart_types = [(r.get("viz") or {}).get("type", "table") for r in dedup]
    #         titles = [(r.get("viz") or {}).get("title", "Panel") for r in dedup]

    #         short_title = f"{short_title} — {ts}"
    #         logger.info(f"[/ASK] Generating dashboard with title: {short_title}")

    #         dashboard_obj = time_call(
    #             fn_times,
    #             "ollama_generate",
    #             generate_dashboard_with_ollama,
    #             sqls=sqls,
    #             chart_types=chart_types,
    #             titles=titles,
    #             dashboard_title=short_title,
    #             time_from="now-5y",
    #             time_to="now",
    #         )

    #         _ensure_panel_formats(dashboard_obj)
    #         ok, err, validated = time_call(fn_times, "validate", validate_with_pydantic, dashboard_obj)
    #         logger.info(f"[/ASK] Dashboard validation result: ok={ok}, err={err}")
    #         if not ok:
    #             raise RuntimeError(f"Final dashboard invalid: {err}")

    #         validated_json = validated.dict(by_alias=True) if hasattr(validated, "dict") else validated

    #         logger.info(f"[/ASK] Posting dashboard to Grafana MCP endpoint.")
    #         graf = time_call(
    #             fn_times,
    #             "post_grafana",
    #             requests.post,
    #             os.getenv("GRAFANA_MCP_URL", "http://grafana-mcp:8000/create_dashboard"),
    #             headers={"Content-Type": "application/json"},
    #             json=validated_json,
    #             timeout=30,
    #         )
    #         graf.raise_for_status()

    #         try:
    #             graf_result = graf.json()
    #             logger.info(f"[/ASK] Grafana MCP response: {graf_result}")
    #         except Exception:
    #             graf_result = {"text": graf.text}

    #         extra.update(
    #             {
    #                 "grafana_result": graf_result,
    #                 "validated": (
    #                     validated.dict(by_alias=True, exclude_none=True) if hasattr(validated, "dict") else validated
    #                 ),
    #                 "function_times_ms": fn_times,
    #             }
    #         )
    #         logger.info(f"[/ASK] Dashboard created successfully.")
    #         print(f"✅ Dashboard created: {graf_result}")

    #         success = True

    #     except Exception as e:
    #         print(f"❌ Dashboard step failed: {e}")
    #         logger.exception("Dashboard pipeline failed")
    #         extra["dashboard_error"] = str(e)
    #         error_msg = str(e)
    # else:
    #     print("❌ No valid SQL generated; skipping dashboard creation.")
    #     logger.error("No valid SQL generated; skipping dashboard creation.")
    #     error_msg = "No valid SQL generated"

    # # one place to record metrics (no undefined 'steps')
    # try:
    #     record_metric(
    #         prompt=req.prompt,
    #         start_ts=start,
    #         success=success,
    #         error_msg=error_msg,
    #         iterations=len(dedup),
    #         function_times=fn_times,
    #     )
    # except Exception:
    #     logger.exception("record_metric failed")

    if success:
        return {
            "status": "success",
            "timestamp": ts,
            "sql": sql_preview,
            "summary": "Final Answer: " + json.dumps({"results": dedup}),
            "executions": exec_outputs,   # <-- preview rows/columns per SQL
        }
    else:
        return {
            "status": "error",
            "timestamp": ts,
            "sql": sql_preview,
            "summary": "Final Answer: " + json.dumps({"error": error_msg or "No valid SQL generated."}),
        }
