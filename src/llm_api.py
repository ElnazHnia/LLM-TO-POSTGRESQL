from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import os, time, re, json, asyncio, requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from src.metrics import record_metric, time_call
from src.models import DashboardSpec
# LLM / LangChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.tools import BaseTool

# MCP
from langchain_mcp_adapters.client import MultiServerMCPClient

# ----------------------------
# FastAPI app + config
# ----------------------------
app = FastAPI()
# Time budget (seconds) for Ollama to return the dashboard JSON
OLLAMA_JSON_TIMEOUT = int(os.getenv("OLLAMA_JSON_TIMEOUT") or os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8000")
OLLAMA_HTTP_URL = os.getenv("OLLAMA_HTTP_URL") or f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
OLLAMA_TIMEOUT  = int(os.getenv("OLLAMA_JSON_TIMEOUT") or os.getenv("OLLAMA_TIMEOUT", "300"))
# ----------------------------
# Models / helpers
# ----------------------------
class PromptRequest(BaseModel):
    prompt: str


def _short(obj: Any, maxlen: int = 600) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False) if isinstance(obj, (dict, list)) else str(obj)
    except Exception:
        s = str(obj)
    return (s[:maxlen] + "…") if len(s) > maxlen else s


def _parse_final_answer_obj(text: str) -> Optional[dict]:
    """Extract and parse the FIRST top-level JSON object after 'Final Answer:'.
    Ignores any trailing notes/comments the model appends after the JSON.
    """
    if not isinstance(text, str):
        return None

    s = text.strip()
    idx = s.lower().find("final answer:")
    if idx != -1:
        s = s[idx + len("final answer:"):].lstrip()

    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    end = None
    in_str = False
    esc = False

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

    candidate = s[start:end] if end else s[start:]
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            return None


def _extract_all_sql_blocks(text: str) -> List[str]:
    """Extract multiple SELECT ... FROM ... statements from free text."""
    if not isinstance(text, str):
        return []
    pattern = re.compile(r'(?is)\bselect\b[\s\S]{10,4000}?\bfrom\b[\s\S]{1,4000}?(?=(?:\bselect\b|$))')
    blocks = []
    for m in pattern.finditer(text):
        sql = ' '.join(m.group(0).strip().split())
        blocks.append(sql)
    seen = set()
    unique = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def _extract_sql_block(text: str) -> Optional[str]:
    """Find a plausible SQL SELECT... statement inside free text / code blocks."""
    if not isinstance(text, str):
        return None
    m = re.search(r'(?is)sql\s*[:=]\s*(?P<q>"""|\'\'\')\s*(.+?)\s*(?P=q)', text)
    if m:
        sql = m.group(2).strip()
        if 'select' in sql.lower() and 'from' in sql.lower():
            return ' '.join(sql.split())
    m = re.search(r'(?is)\bselect\b[\s\S]{10,4000}\bfrom\b[\s\S]{1,4000}', text)
    if m:
        sql = m.group(0).strip()
        return ' '.join(sql.split())
    return None


class DebugCallbackHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name")
        print(f"🔧 TOOL START → {name}")
        if input_str:
            print(f"   ↳ input_str: {_short(input_str)}")
        inputs = kwargs.get("inputs") or kwargs.get("kwargs")
        if inputs:
            print(f"   ↳ inputs:    {_short(inputs)}")

    def on_tool_end(self, serialized, output, **kwargs):
        name = (serialized or {}).get("name")
        print(f"✅ TOOL END   → {name}")
        print(f"   ↳ output:    {_short(output)}")


SYNONYM_MAPPINGS = {
    "customer":       "person",
    "user":           "person",
    "product":        "products",
    "item":           "products",
    "sale":           "sales",
    "order":          "orders",
    "payment method": "payment_type",
}

def normalize_synonyms(text: str) -> str:
    keys = set(SYNONYM_MAPPINGS.keys())
    keys |= {k + "s" for k in SYNONYM_MAPPINGS.keys()}
    pattern = rf"(?<!\w)({'|'.join(map(re.escape, sorted(keys, key=len, reverse=True)))})(?!\w)"

    def to_singular(token: str) -> str:
        t = token.lower()
        return t[:-1] if t.endswith("s") and t[:-1] in SYNONYM_MAPPINGS else t

    def repl(m):
        tok = m.group(1)
        base = to_singular(tok)
        out = SYNONYM_MAPPINGS.get(base, tok)
        if tok.lower().endswith("s") and not out.lower().endswith("s"):
            if out.endswith("y"):
                out = out[:-1] + "ies"
            else:
                out = out + "s"
        return out if tok.islower() else out.capitalize() if tok.istitle() else out.upper()

    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Lightweight SQL parsing helpers (for validation)
# ----------------------------
_SQL_IDENT = r"[a-zA-Z_][\w]*"

def _extract_tables_and_aliases(sql: str) -> Tuple[List[str], Dict[str, str]]:
    tables: List[str] = []
    aliases: Dict[str, str] = {}

    for m in re.finditer(rf"\bFROM\s+({_SQL_IDENT})(?:\s+(?:AS\s+)?({_SQL_IDENT}))?", sql, flags=re.IGNORECASE):
        tbl = m.group(1)
        ali = m.group(2)
        tables.append(tbl)
        if ali:
            aliases[ali] = tbl

    for m in re.finditer(rf"\bJOIN\s+({_SQL_IDENT})(?:\s+(?:AS\s+)?({_SQL_IDENT}))?", sql, flags=re.IGNORECASE):
        tbl = m.group(1)
        ali = m.group(2)
        tables.append(tbl)
        if ali:
            aliases[ali] = tbl

    seen = set()
    ordered_tables = []
    for t in tables:
        if t not in seen:
            seen.add(t)
            ordered_tables.append(t)
    return ordered_tables, aliases

def _extract_qualified_columns(sql: str) -> List[Tuple[str, str]]:
    cols: List[Tuple[str, str]] = []
    for t, c in re.findall(rf"\b({_SQL_IDENT})\.({_SQL_IDENT})\b", sql):
        cols.append((t, c))
    return cols


# ----------------------------
# Guards for MCP tools (async-safe)
# ----------------------------
class ExecuteQueryGuard(BaseTool):
    name: str = "execute_query"
    description: str = (
        "Run a COMPLETE SQL query. Signature: execute_query(sql: string) – pass the SQL via dict {'sql': '<...>'}. "
        "No placeholders. No arrays."
    )
    inner_tool: Any
    schema_tool: Optional[Any] = None
    _schema_cache: Dict[str, Optional[List[str]]] = {}

    _FMT_MAP = {
        "barchart": "table",
        "piechart": "table",
        "table": "table",
        "timeseries": "time_series",
        "stat": "table",
        "gauge": "table",
        "line": "time_series",
    }

    def _clean_query_and_passthrough(self, tool_input: Any) -> Tuple[Optional[str], Dict[str, Any]]:
        extras: Dict[str, Any] = {}
        if isinstance(tool_input, dict):
            if "kwargs" in tool_input and isinstance(tool_input["kwargs"], dict):
                tool_input = tool_input["kwargs"]
            q = tool_input.get("query")
            if q is None:
                q = tool_input.get("sql")
            for k, v in tool_input.items():
                if k in ("query", "sql", "kwargs", "args"):
                    continue
                extras[k] = v
            if isinstance(q, list):
                q = " ".join(str(x) for x in q)
            if isinstance(q, str):
                return q.strip(), extras
            return None, extras
        if isinstance(tool_input, str):
            s = tool_input.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        return self._clean_query_and_passthrough(obj)
                except Exception:
                    pass
            return s, extras
        return None, extras

    def _is_placeholder(self, q: str) -> bool:
        if not q or len(q.strip()) < 12:
            return True
        condensed = " ".join(q.upper().split())
        return condensed in {"SELECT", "FROM", "GROUP BY", "ORDER BY"}

    async def _load_schema(self, table: str) -> Optional[List[str]]:
        if table in self._schema_cache:
            return self._schema_cache[table]
        if not self.schema_tool:
            self._schema_cache[table] = None
            return None
        try:
            raw = await self.schema_tool.arun(tool_input={"table_name": table})
        except Exception as e:
            print(f"⚠️ schema fetch failed for {table}: {e}")
            self._schema_cache[table] = None
            return None
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            parsed = raw
        if isinstance(parsed, str) and parsed.strip().upper().startswith("ERROR"):
            self._schema_cache[table] = None
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("columns"), list):
            col_list = parsed["columns"]
        elif isinstance(parsed, list):
            col_list = parsed
        else:
            self._schema_cache[table] = None
            return None
        cleaned: List[str] = []
        for c in col_list:
            if not isinstance(c, str):
                continue
            name = _clean_col_name(c)
            if name and name not in cleaned:
                cleaned.append(name)
        self._schema_cache[table] = cleaned
        return cleaned

    async def _validate_against_schema(self, q: str) -> Optional[str]:
        tables, alias_map = _extract_tables_and_aliases(q)
        qcols = _extract_qualified_columns(q)
        def resolve(token: str) -> str:
            return alias_map.get(token, token)
        per_table_known: Dict[str, List[str]] = {}
        targets = set(resolve(tok) for tok, _ in qcols) | set(tables)
        for t in targets:
            cols = await self._load_schema(t)
            if cols:
                per_table_known[t] = cols
        if not per_table_known:
            return None
        unknown: List[str] = []
        for token, col in qcols:
            base = resolve(token)
            known = per_table_known.get(base)
            if known is not None and col not in known:
                unknown.append(f"{token}.{col} (table {base})")
        if unknown:
            hints = []
            for t, cols in per_table_known.items():
                sample = ", ".join(cols[:20]) if cols else "(none)"
                hints.append(f"- {t}: {sample}")
            return (
                "ERROR: Unknown column(s): " + ", ".join(unknown) + "\n"
                "Known columns by table:\n" + "\n".join(hints)
            )
        return None

    def _coerce_format(self, payload: Dict[str, Any]) -> None:
        # 1) Respect explicit "format" if present; normalize via _FMT_MAP.
        fmt = payload.get("format")
        if isinstance(fmt, str) and fmt.strip():
            val = fmt.strip().lower()
            payload["format"] = self._FMT_MAP.get(val, val)
            return

        # 2) If caller gave a chart_type, try to map it into a format hint.
        ctype = payload.get("chart_type")
        if isinstance(ctype, str) and ctype.strip():
            key = ctype.strip().lower()
            mapped = self._FMT_MAP.get(key)
            if mapped:
                payload["format"] = mapped  # provisional; may be refined by SQL below

        # 3) Infer from SQL only when no explicit format was given.
        sql = (payload.get("sql") or "").strip()
        sql_u = sql.lower()

        # Heuristics for time series:
        time_expr_patterns = [
            r"date_trunc\s*\(",
            r"\bto_char\s*\(.*?(yyyy|yy|mm|mon|month|dd|hh|mi|min|ss)\b",
            r"\bextract\s*\(\s*(year|month|day|hour|minute|second)\s+from",
        ]
        time_col_tokens = [
            "sale_date", "created_at", "updated_at", "order_date",
            "date", "timestamp", "time"
        ]

        has_time_expr = any(re.search(p, sql_u) for p in time_expr_patterns)
        has_time_col = any(re.search(rf"\b{re.escape(tok)}\b", sql_u) for tok in time_col_tokens)

        has_group_by = "group by" in sql_u
        has_agg = re.search(r"\b(sum|avg|count|min|max|median|percentile|stddev|stddev_pop|stddev_samp)\s*\(", sql_u) is not None
        has_order_by_time = re.search(r"order\s+by\s+.*?(date_trunc\s*\(|\b(date|timestamp|time|sale_date)\b)", sql_u) is not None

        is_time_series = (
            has_time_expr
            or (has_time_col and (has_group_by or has_agg or has_order_by_time))
        )

        # 4) Decide the final format.
        if is_time_series:
            payload["format"] = "time_series"
        else:
            payload["format"] = "table"
        print(f"Coerced format: {payload['format']} for SQL: {sql}")


    def _run(self, *args, **kwargs):
        return asyncio.run(self._arun(*args, **kwargs))

    async def _arun(self, *args, **kwargs):
        tool_input = kwargs.get("tool_input", None)
        if tool_input is None and args:
            tool_input = args[0]
        q, extras = self._clean_query_and_passthrough(tool_input)
        if not isinstance(q, str) or not q.strip():
            return "ERROR: execute_query requires a single string param 'sql'."
        q = q.strip()
        if self._is_placeholder(q):
            return "ERROR: Incomplete SQL."
        try:
            err = await self._validate_against_schema(q)
            if err:
                print(f"🛑 execute_query blocked by schema validator:\n{err}")
                return err
        except Exception as e:
            print(f"⚠️ schema validation failed: {e}")
        payload: Dict[str, Any] = {"sql": q}
        for k, v in extras.items():
            if k in ("sql", "query"):
                continue
            payload[k] = v
        if not isinstance(payload.get("chart_type"), str) or not payload["chart_type"].strip():
            payload["chart_type"] = "table"
        self._coerce_format(payload)
        try:
            return await self.inner_tool.arun(payload)
        except Exception as e1:
            msg1 = e1.args[0] if getattr(e1, "args", None) else str(e1)
            low = msg1.lower()
            if "'sql' is not of type 'array'" in low:
                retry_payload = dict(payload)
                if isinstance(retry_payload.get("sql"), str):
                    retry_payload["sql"] = [retry_payload["sql"]]
                try:
                    return await self.inner_tool.arun(retry_payload)
                except Exception as e2:
                    msg2 = e2.args[0] if getattr(e2, "args", None) else str(e2)
                    return f"ERROR: {msg2}"
            if "not of type 'array'" in low:
                p = dict(payload)
                if isinstance(p.get("sql"), str):
                    p["sql"] = [p["sql"]]
                if isinstance(p.get("chart_type"), str):
                    p["chart_type"] = [p["chart_type"]]
                if isinstance(p.get("format"), str):
                    p["format"] = [p["format"]]
                try:
                    return await self.inner_tool.arun(p)
                except Exception as e2:
                    msg2 = e2.args[0] if getattr(e2, "args", None) else str(e2)
                    return f"ERROR: {msg2}"
            return f"ERROR: {msg1}"


class ListTablesGuard(BaseTool):
    name: str = "list_tables"
    description: str = "List available tables. Accepts NO parameters."
    inner_tool: Any
    def _run(self, *args, **kwargs):
        return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        return await self.inner_tool.arun({})


def _clean_col_name(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    s = re.split(r"\s*\(|\s*:\s*", s)[0].strip().strip('"').strip("'")
    m = re.match(rf"{_SQL_IDENT}", s)
    return m.group(0) if m else None


class GetTableSchemaGuard(BaseTool):
    name: str = "get_table_schema"
    description: str = "Get the schema for a table. Signature: get_table_schema(table_name: string) – pass via dict {'table_name': '<name>'}."
    inner_tool: Optional[Any] = None
    list_tool: Optional[Any] = None

    # NEW: memoized normalized schemas across runs; and per-run seen set
    _memo_cache: Dict[str, str] = {}   # table_name -> normalized JSON string
    _seen_this_run: set = set()        # cleared at the start of each /ask

    def _run(self, *args, **kwargs):
        return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        ti = kwargs.get("tool_input", None)
        if ti is None and args:
            ti = args[0]
        table_name = None
        if isinstance(ti, dict):
            table_name = ti.get("table_name")
        elif isinstance(ti, str):
            s = ti.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, dict):
                        table_name = obj.get("table_name")
                except Exception:
                    table_name = s
            else:
                table_name = s
        if not isinstance(table_name, str) or not table_name:
            return "ERROR: get_table_schema requires table_name (string). Example: get_table_schema(table_name='orders')."

        # Optional: validate table exists
        if self.list_tool:
            try:
                available = await self.list_tool.arun({})
                if isinstance(available, str):
                    try:
                        available = json.loads(available)
                    except Exception:
                        pass
                if isinstance(available, list) and table_name not in available:
                    return f"ERROR: Unknown table '{table_name}'."
            except Exception as e:
                print(f"⚠️ list_tables validation failed: {e}")

        # If we've already fetched this table in this run, nudge the agent forward.
        if table_name in self._seen_this_run:
            return (
                f"ERROR: Schema for '{table_name}' already retrieved in this run. "
                "Proceed to execute_query or fetch the next needed table (e.g., 'products' for per-product charts)."
            )

        # If cached from a previous run, return quickly but also mark as seen for this run.
        if table_name in self._memo_cache:
            self._seen_this_run.add(table_name)
            return self._memo_cache[table_name]

        # Normal path: fetch once, normalize, memoize, mark seen
        raw = await self.inner_tool.arun({"table_name": table_name})
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = raw
        cols: List[str] = []
        if isinstance(data, dict):
            sch = data.get("schema")
            if isinstance(sch, list):
                for item in sch:
                    if isinstance(item, dict) and isinstance(item.get("name"), str):
                        cols.append(item["name"])
                    elif isinstance(item, str):
                        cols.append(item)
            if not cols and isinstance(data.get("columns"), list):
                for c in data["columns"]:
                    if isinstance(c, str):
                        cols.append(c)
        elif isinstance(data, list):
            for c in data:
                if isinstance(c, str):
                    cols.append(c)
        seen = set()
        cols_norm: List[str] = []
        for c in cols:
            name = _clean_col_name(c)
            if name and name not in seen:
                seen.add(name)
                cols_norm.append(name)
        normalized = json.dumps({"columns": cols_norm})
        self._memo_cache[table_name] = normalized
        self._seen_this_run.add(table_name)
        return normalized


# ----------------------------
# Tool alias wrappers
# ----------------------------
class ToolAliasNoInput(BaseTool):
    name: str
    description: str = "Alias tool that forwards to another tool"
    target_tool: Any
    def _run(self, *args, **kwargs):
        return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        return await self.target_tool.arun({})

class ToolAliasPassthrough(BaseTool):
    name: str
    description: str = "Alias tool that forwards to another tool"
    target_tool: Any
    def _run(self, *args, **kwargs):
        return asyncio.run(self._arun(*args, **kwargs))
    async def _arun(self, *args, **kwargs):
        ti = kwargs.get("tool_input", {})
        if isinstance(ti, str):
            s = ti.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    ti = json.loads(s)
                except Exception:
                    ti = {}
            else:
                ti = {}
        if not isinstance(ti, dict):
            ti = {}
        return await self.target_tool.arun(ti)


# ----------------------------
# Startup: LLM + MCP tools + Agent
# ----------------------------
@app.on_event("startup")
async def startup_event():
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        top_p=1.0,
        model_kwargs={"keep_alive": "10m"}
    ).bind(stop=["```", "<|python_tag|>", "</tool>"])
    app.state.llm_base = llm
    try:
        llm.invoke("hello")
    except Exception as e:
        print(f"⚠️ LLM warmup failed: {e}")

    tools: List[Any] = []
    mcp_endpoint = f"{MCP_URL.rstrip('/')}/mcp/"
    mcp_client = MultiServerMCPClient({
        "default": {
            "url": mcp_endpoint,
            "transport": "streamable_http",
        }
    })

    try:
        mcp_tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=10)
        print("🧩 Loaded MCP tools:", [t.name for t in mcp_tools])
    except Exception as e:
        print(f"⚠️ Skipping MCP tools (error reaching {mcp_endpoint}): {e}")
        mcp_tools = []

    # want = {"list_tables", "get_table_schema", "execute_query"}
    # base_tools = [t for t in mcp_tools if t.name in want]
    # by_name = {t.name: t for t in base_tools}
    by_name = {t.name: t for t in mcp_tools}
    # Accept either plain names or the sql.* variants exposed by fastapi-mcp
    def pick_tool(candidates):
        for name in candidates:
            if name in by_name:
                return by_name[name]
        return None

    list_inner  = pick_tool(["list_tables", "sql.list_tables"])
    schema_inner= pick_tool(["get_table_schema", "sql.schema"])
    exec_inner  = pick_tool(["execute_query", "sql.query"])

    list_guard = None
    schema_guard = None
    exec_guard = None
    
    if list_inner:
        list_guard = ListTablesGuard(inner_tool=list_inner)
        tools.append(list_guard)
    else:
        print("⚠️ MCP tool for listing tables not found (tried: list_tables, sql.list_tables)")

    if schema_inner:
        schema_guard = GetTableSchemaGuard(inner_tool=schema_inner)
        if list_guard:
            schema_guard.list_tool = list_guard
        tools.append(schema_guard)
    else:
        print("⚠️ MCP tool for schema not found (tried: get_table_schema, sql.schema)")

    if exec_inner:
        exec_guard = ExecuteQueryGuard(inner_tool=exec_inner, schema_tool=schema_guard)
        tools.append(exec_guard)
    else:
        print("⚠️ MCP tool for execute not found (tried: execute_query, sql.query)")
        

    # Aliases
    alias_specs: List[Tuple[Any, str, Any]] = []
    if list_guard is not None:
        alias_specs += [
            (ToolAliasNoInput, "List available tables.", list_guard),
            (ToolAliasNoInput, "List available tables",  list_guard),
            (ToolAliasNoInput, "List tables",            list_guard),
        ]
    if exec_guard is not None:
        alias_specs += [
            (ToolAliasPassthrough, "Execute SQL query to retrieve data.", exec_guard),
            (ToolAliasPassthrough, "Execute SQL query",                    exec_guard),
            (ToolAliasPassthrough, "Execute query",                        exec_guard),
            (ToolAliasPassthrough, "Execute query to get data.",           exec_guard),
            (ToolAliasPassthrough, "Execute query to get data",            exec_guard),
        ]
    if schema_guard is not None:
        alias_specs += [
            (ToolAliasPassthrough, "Get table schema",  schema_guard),
            (ToolAliasPassthrough, "Get schema",        schema_guard),
            (ToolAliasPassthrough, "Describe table",    schema_guard),
        ]
        available_tables: List[str] = []
        if list_guard is not None:
            try:
                available = await list_guard.arun({})
                if isinstance(available, str):
                    try:
                        available = json.loads(available)
                    except Exception:
                        available = []
                if isinstance(available, list):
                    available_tables = [str(t) for t in available]
            except Exception as e:
                print(f"⚠️ Could not fetch tables for schema aliases: {e}")
        patterns = [
            "Get table schema for '{t}'.",
            "Get table schema for '{t}'",
            "Get table schema for {t}.",
            "Get table schema for {t}",
            "Get schema for '{t}'.",
            "Get schema for '{t}'",
            "Describe table '{t}'.",
            "Describe table '{t}'",
            "Describe table {t}.",
            "Describe table {t}",
        ]
        for t in available_tables:
            for pat in patterns:
                alias_name = pat.format(t=t)
                alias_specs.append((ToolAliasPassthrough, alias_name, schema_guard))

    for cls, alias_name, target in alias_specs:
        try:
            alias_tool = cls(name=alias_name, target_tool=target)
            tools.append(alias_tool)
        except Exception as e:
            print(f"⚠️ Could not register alias '{alias_name}': {e}")

    print("🚀 Agent ready with tools: ", [t.name for t in tools])

    # --- System prompt (with required variables + doubled braces in JSON) ---
    system_text = '''
You are a SQL analysis agent with MCP tools:
- list_tables()
- get_table_schema(table_name)
- execute_query(sql)

RULES (CRITICAL):
- Do NOT include Python, Markdown, or code fences in your replies.
- NEVER write placeholders like "..." anywhere in outputs.
- The only allowed structures are exactly these lines, in this order:
  Thought: <brief reasoning> | Checklist: CAST=OK; AGG=OK; GBY=OK; EXTRA=NO
  Action: <one of: list_tables, get_table_schema, execute_query>
  Action Input: {{"...": "..."}}
  Observation: <tool result or ERROR: ...>
- After one or more cycles, finish with ONE final line beginning with "Final Answer:" followed by a single JSON object.
- The Checklist MUST appear ONLY in the Thought line and nowhere else.

SQL LINT CHECKLIST (MANDATORY in Thought line):
- CAST=OK → all non-aggregate SELECT expressions end with ::text.
- AGG=OK → no aggregate is cast to ::text.
- GBY=OK → every GROUP BY expression appears in SELECT and, if non-aggregate, ends with ::text; never use positional indices.
- EXTRA=NO → there are no auxiliary/duplicate queries for the same intent.
If any token would not be OK, FIX the SQL BEFORE calling execute_query or emitting Final Answer.

SELECT COLUMN POLICY (HUMAN-FRIENDLY OUTPUT) — HARD REQUIREMENT:
- Do NOT select raw primary key or foreign key columns (columns named "id" or ending with "_id") in the SELECT list.
- Instead, JOIN to the referenced table and select a human-readable field (typically `name`) with ::text and a clear alias.
  Mappings:
    * sales.person_id   → JOIN person         ON s.person_id   = p.id   → p.name::text        AS person
    * sales.product_id  → JOIN products       ON s.product_id  = pr.id  → pr.name::text       AS product
    * sales.payment_id  → JOIN payment_type   ON s.payment_id  = pt.id  → pt.name::text       AS payment_type
- If any *_id appears in SELECT, rewrite with the JOIN and name before execute_query.

CASTING POLICY (HARD REQUIREMENT):
- Every non-aggregate SELECT expression MUST include ::text.
- Do NOT cast aggregates (SUM/AVG/COUNT/MIN/MAX/etc.).
- Do NOT cast numeric/boolean expressions or literals.
- Cast date/time label expressions (e.g., DATE_TRUNC('year', ...)) to ::text.
- NEVER add extra queries just for casting — only cast the columns already needed in the SELECT list.

GROUP BY / ORDER BY POLICY:
- Do NOT use positional indices (e.g., "GROUP BY 1", "ORDER BY 1").
- Always GROUP BY and ORDER BY using explicit aliases or full expressions (e.g., GROUP BY year ORDER BY year).

TOOL CALL SHAPES (STRICT):
- Action: list_tables
  Action Input: {{}}
- Action: get_table_schema
  Action Input: {{"table_name":"<exact name>"}}
- Action: execute_query
  Action Input: {{"sql":"<full SQL>","chart_type":"barchart|line|piechart|table|stat|gauge","format":"table|time_series"}}
- The Action Input for execute_query MUST have "sql" as a raw SQL string only (not JSON). Never wrap the SQL string itself as JSON.

WORKFLOW:
- Determine the minimal set of chart intents in the user request (max 4), and handle them in order.
- For each intent:
  1) Call get_table_schema() ONCE for every table you will reference (e.g., sales, person, products, payment_type).
  2) Build ONE SQL that satisfies CAST/GBY and the SELECT COLUMN POLICY (no raw ids).
  3) Call execute_query once.
- Do NOT repeat list_tables or get_table_schema for the same table more than once in a run.
- If you receive an Observation starting with "ERROR: Schema for '<table>' already retrieved in this run", STOP calling get_table_schema('<table>') and move on.

SCHEMA-STRICT MODE:
- Only reference columns you personally saw via get_table_schema() this run.

COMMON PATTERNS (reference):
- Annual totals:
  SELECT DATE_TRUNC('year', s.sale_date)::text AS year, SUM(s.value) AS total
  FROM sales s
  GROUP BY year
  ORDER BY year
- Per person average sale:
  SELECT p.name::text AS person, AVG(s.value) AS avg_sales
  FROM sales s JOIN person p ON s.person_id=p.id
  GROUP BY person
  ORDER BY avg_sales DESC
- Per product totals:
  SELECT pr.name::text AS product, SUM(s.value) AS total
  FROM sales s JOIN products pr ON s.product_id=pr.id
  GROUP BY product
  ORDER BY total DESC
- Per payment type counts & totals:
  SELECT pt.name::text AS payment_type, COUNT(*) AS num_sales, SUM(s.value) AS total_value
  FROM sales s JOIN payment_type pt ON s.payment_id=pt.id
  GROUP BY payment_type
  ORDER BY total_value DESC

REQUIRED ARGUMENTS FOR execute_query (examples):
  {{"sql":"SELECT SUM(value) AS total FROM sales","chart_type":"stat","format":"table"}}
  {{"sql":"SELECT DATE_TRUNC('year', s.sale_date)::text AS year, SUM(s.value) AS total FROM sales s GROUP BY year ORDER BY year","chart_type":"barchart","format":"table"}}
  {{"sql":"SELECT pr.name::text AS product, AVG(s.value) AS avg_value FROM sales s JOIN products pr ON s.product_id=pr.id GROUP BY product ORDER BY avg_value DESC","chart_type":"barchart","format":"table"}}
  {{"sql":"SELECT pt.name::text AS payment_type, COUNT(*) AS num_sales, SUM(s.value) AS total_value FROM sales s JOIN payment_type pt ON s.payment_id=pt.id GROUP BY payment_type ORDER BY total_value DESC","chart_type":"barchart","format":"table"}}

ERROR BEHAVIOR:
- If an Observation starts with "ERROR:", correct the SQL once and try execute_query again for that intent, then move to the next intent.

FINAL OUTPUT (SUCCESS) — one line only:
Final Answer: {{"results":[
  {{"sql":"<SQL 1>","viz":{{"type":"<barchart|line|piechart|table|stat|gauge>","format":"<table|time_series>","title":"<short label>"}}}},
  {{"sql":"<SQL 2>","viz":{{"type":"<barchart|line|piechart|table|stat|gauge>","format":"<table|time_series>","title":"<short label>"}}}},
  ...
]}}

FINAL OUTPUT (ERROR) — one line only:
Final Answer: {{"error":"<clear message about what is missing or ambiguous>"}}

TOOLS:
{tools}

Valid tool names: {tool_names}
'''.strip()



    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{input}\n\n{agent_scratchpad}"),
    ])

    def _format_fix(e: Exception) -> str:
        return (
            "FORMAT ERROR: Your last message was not in the expected ReAct format.\n"
            "Respond again using EXACTLY this structure:\n"
            "Thought: <brief reasoning>\n"
            "Action: <one of: list_tables, get_table_schema, execute_query>\n"
            'Action Input: {"...": "..."}\n'
        )

    cbmgr = CallbackManager([DebugCallbackHandler()])
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        callback_manager=cbmgr,
        handle_parsing_errors=_format_fix,
        max_iterations=12,
        max_execution_time=45,
        early_stopping_method="force",
    )

    app.state.executor = executor
    app.state.tools = {t.name: t for t in tools}


# ----------------------------
# Helper: detect agent error JSON
# ----------------------------
def _looks_like_error_json(s: str) -> Optional[str]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "error" in obj and isinstance(obj["error"], str):
            return obj["error"]
    except Exception:
        pass
    return None


# ----------------------------
# Fast-lanes
# ----------------------------
DAILY_SALES_RE = re.compile(
    r"\b(daily|per\s*day)\b.*\b(sale|sales)\b.*\b(bar\s*chart|bar\s*graph)\b",
    re.IGNORECASE
)

# Multi-intent fast lane for common dashboards
AVG_PER_PERSON_RE = re.compile(r"\b(avg|average)\b.*\b(per|by)\b.*\bperson\b", re.IGNORECASE)
ANNUAL_RE = re.compile(r"\b(annual|annually|year|yearly)\b", re.IGNORECASE)
PER_PRODUCT_RE = re.compile(r"\b(product|products)\b", re.IGNORECASE)

async def _table_has(schema_tool, table: str, needed: List[str]) -> bool:
    try:
        sch_raw = await schema_tool.arun({"table_name": table})
        sch = json.loads(sch_raw) if isinstance(sch_raw, str) else sch_raw
        cols = set((sch or {}).get("columns", [])) if isinstance(sch, dict) else set()
        return set(needed).issubset(cols)
    except Exception:
        return False

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

    # Hoist stray fields the model sometimes puts at top level
    top_chart_type = item.get("chart_type")
    top_format = item.get("format")
    top_title = item.get("title")

    if "type" not in viz and isinstance(top_chart_type, str):
        viz["type"] = top_chart_type
    if "format" not in viz and isinstance(top_format, str):
        viz["format"] = top_format
    if "title" not in viz and isinstance(top_title, str):
        viz["title"] = top_title

    # Defaults
    viz.setdefault("type", "table")
    viz.setdefault("format", "table")
    viz.setdefault("title", "Result")

    return {"sql": sql.strip(), "viz": viz}

def _msg_text(msg: Any) -> str:
    """Extract text from LangChain AIMessage or plain string."""
    if isinstance(msg, str):
        return msg
    # LangChain's AIMessage has `.content`
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    return str(msg)

def _sanitize_text(s: Any) -> str:
    return (str(s or "")).replace("\n", " ").replace("\r", " ").strip()

def _extract_top_json_object(text: str) -> dict:
    """
    Extract the FIRST top-level JSON object from an arbitrary string.
    Strips code fences and optional 'Final Answer:' prefix.
    """
    s = (text or "").strip()
    # Strip code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        if "```" in s:
            s = s[:s.rfind("```")]
        s = s.strip()
    # Handle 'Final Answer:' prefix if the model included it
    idx = s.lower().find("final answer:")
    if idx != -1:
        s = s[idx + len("final answer:"):].lstrip()
    # Find first JSON object
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth, end, in_str, esc = 0, None, False, False
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
    if end is None:
        raise ValueError("Unterminated JSON object in model output.")
    candidate = s[start:end]
    try:
        return json.loads(candidate)
    except Exception:
        # last-ditch: common single-quote JSON
        return json.loads(candidate.replace("'", '"'))

# Prompt builder for Ollama to emit FINAL dashboard JSON only
def _prompt_for_grafana_dashboard(results: List[Dict[str, Any]], title_prefix: str, timestamp: str) -> str:
    """
    Build a strict prompt so Ollama returns ONE JSON object:
      {
        "dashboard": {
          "title": "<dynamic from panel titles (and optional prefix)> - <timestamp>",
          ...
          "panels": [...]
        },
        "overwrite": true
      }

    - Panels: 2-per-row layout; last odd panel full-width
    - type↔format mapping enforced
    - SQL used verbatim (escaped for prompt safety)
    """

    def _sanitize(s: str) -> str:
        return (s or "").replace("\n", " ").replace("\r", " ").replace('"', '\\"').strip()

    # Collect panel titles (preserve order, drop duplicates) for dynamic title
    seen = set()
    panel_titles: List[str] = []
    for r in results:
        viz = r.get("viz") or {}
        t = viz.get("title") or "Panel"
        t = " ".join(t.split())
        if t not in seen:
            seen.add(t)
            panel_titles.append(t)

    # Build a concise dynamic title from the panel titles
    # Example: "Sales per Product • Average Sales per Person • Annual Sales"
    dyn_from_titles = " • ".join(panel_titles) if panel_titles else "Dashboard"

    # If caller provided a prefix, prepend it; otherwise just use the titles
    # Resulting prefix kept to ~120 chars to avoid overly long titles
    base_prefix = (title_prefix or "").strip()
    if base_prefix:
        dynamic_prefix = f"{base_prefix}: {dyn_from_titles}"
    else:
        dynamic_prefix = dyn_from_titles
    if len(dynamic_prefix) > 120:
        dynamic_prefix = dynamic_prefix[:117] + "..."

    # Prepare compact panels description for the model
    lines = []
    for i, r in enumerate(results, 1):
        viz = r.get("viz") or {}
        p_title = _sanitize(viz.get("title", "Panel"))
        p_type  = _sanitize((viz.get("type") or "table").lower())
        p_sql   = _sanitize(r.get("sql", ""))
        lines.append(f'{i}) title="{p_title}", type="{p_type}", sql="{p_sql}"')
    panels_block = "\n".join(lines)

    # IMPORTANT: We tell the model the exact final title it must set.
    final_title_prefix = _sanitize(dynamic_prefix)
    EXACT_TITLE_LINE = f'- Set dashboard.title EXACTLY to "{final_title_prefix} - {timestamp}"'

    return f"""
    SYSTEM RULES:
    - Output ONE JSON object ONLY. No prose, no code fences.
    - Top-level keys: "dashboard", "overwrite".
    - {EXACT_TITLE_LINE}
    - dashboard.title MUST also match regex: ".* - {timestamp}"
    - dashboard.schemaVersion=36, version=1, refresh="5s"
    - dashboard.time={{"from":"now-5y","to":"now"}}
    - dashboard.timepicker with typical refresh_intervals and time_options
    - datasource={{"type":"postgres","uid":"desl46jinre9sb"}} for every panel
    - Panels layout: 24-column grid, h=8 per panel:
    * Index i (0-based), total N:
        - if i==N-1 and N is odd → gridPos={{"x":0,"y":(i//2)*8,"w":24,"h":8}}
        - else if i even → gridPos={{"x":0,"y":(i//2)*8,"w":12,"h":8}}
        - else (i odd)  → gridPos={{"x":12,"y":(i//2)*8,"w":12,"h":8}}
    - Each panel:
    * id = 1..N, title from input, type ∈ {{barchart,piechart,timeseries,table,stat}}
    * targets[0]={{"refId":"A","rawSql":<SQL>,"format":<see below>}}
    * FORMAT mapping: timeseries → "time_series"; others → "table"
    - Do NOT modify SQL. Use it exactly as provided.

    INPUT:
    title_prefix="{final_title_prefix}"
    timestamp="{timestamp}"
    PANELS:
    {panels_block}

    RETURN: The final dashboard JSON ONLY.
    """.strip()


# Call Ollama to generate dashboard JSON, validate, and push
async def _ollama_make_validate_push_dashboard_async(
    llm: ChatOllama,
    results: List[Dict[str, Any]],
    *,
    title_prefix: str,
    timestamp: str,
    gen_timeout: int = 90,         # <- configurable generation timeout
    post_timeout: int = 30,        # <- http post timeout
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ask Ollama to build the dashboard JSON, validate (if models present), optionally POST to Grafana,
    and optionally write to a file. Returns {"grafana_result": ..., "validated": ..., "raw_text": ...}.
    """
    prompt = _prompt_for_grafana_dashboard(results, title_prefix, timestamp)

    # ChatOllama.invoke(...) is blocking. Run it off the event loop thread and await with a timeout.
    try:
        ai_msg = await asyncio.wait_for(
            asyncio.to_thread(llm.invoke, prompt),
            timeout=gen_timeout
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Ollama generation timed out after {gen_timeout}s")

    # LangChain returns an AIMessage. Use .content.
    raw_text = getattr(ai_msg, "content", str(ai_msg))

    # Parse the JSON object from the model output
    try:
        dashboard_dict = _extract_top_json_object(raw_text)
    except Exception as e:
        snippet = raw_text[:400].replace("\n", " ")
        raise RuntimeError(f"Ollama did not return valid JSON: {e}. Got: {snippet}")

    # If you have Pydantic models/validators available, use them; otherwise pass-through.
    payload = dashboard_dict
    if "DashboardSpec" in globals():
        try:
            validated = DashboardSpec(**dashboard_dict)
            payload = validated.dict(by_alias=True, exclude_none=True)
        except Exception as e:
            raise RuntimeError(f"Final dashboard invalid (Pydantic): {e}")

        if "DashboardValidation" in globals():
            ok, err = DashboardValidation.validate_dashboard_python(
                dashboard_dict=payload,
                prompt=title_prefix,
                timestamp=timestamp,
                table_meta={}
            )
            if not ok:
                raise RuntimeError(f"Final dashboard invalid: {err}")

    # Save to file if requested
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed writing dashboard JSON to {save_path}: {e}")

    # Optional POST to Grafana when enabled
    grafana_result = None
    url = os.getenv("GRAFANA_MCP_URL")
    if url and os.getenv("ENABLE_GRAFANA_POST", "0") == "1":
        import requests  # local import so your app runs even if requests is omitted elsewhere
        try:
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=post_timeout
            )
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            grafana_result = resp.json() if "application/json" in ctype else resp.text
        except Exception as e:
            raise RuntimeError(f"Grafana POST failed: {e} | body={getattr(e, 'response', None) and getattr(e.response, 'text', '')[:400]}")
    return {"grafana_result": grafana_result, "validated": payload, "raw_text": raw_text}




def _remove_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        # strip leading fence
        s = s.split("```", 2)[1] if "```" in s[3:] else s[3:]
        # strip trailing fence if present
        if "```" in s:
            s = s[:s.rfind("```")]
    return s.strip()

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

    # System instructions kept lean and JSON-only
    system_msg = (
        "You are an expert in Grafana dashboard JSON (schemaVersion 36). "
        "Return EXACTLY one JSON object, no prose, no code fences. "
        "Use this structure:\n"
        "{\n"
        '  "dashboard": {\n'
        '    "schemaVersion": 36,\n'
        '    "title": "<dashboard_title>",\n'
        '    "time": {"from": "<time_from>", "to": "<time_to>"},\n'
        '    "refresh": "30s",\n'
        '    "panels": [\n'
        '      {"id": 0, "title": "Panel Title", "type": "panel_type",\n'
        '       "gridPos": {"x":0,"y":0,"w":12,"h":8},\n'
        '       "targets": [{"refId":"A","rawSql":"SQL_QUERY","format":"table_or_time_series"}]}\n'
        "    ]\n"
        "  },\n"
        '  "overwrite": true\n'
        "}\n"
        "Panel/layout requirements:\n"
        "- Panels are INSIDE dashboard.panels (not at root).\n"
        "- id = 0..N-1.\n"
        "- 24-col grid, h=8. Two-per-row: even index => x=0,y=(i//2)*8,w=12; odd => x=12,y=(i//2)*8,w=12.\n"
        "- If panel count is odd, last panel is full width: x=0,y=(i//2)*8,w=24.\n"
        "- Use the provided rawSql exactly; do not alter it.\n"
        "- For each panel, create targets[0] with refId A, rawSql, and the given format.\n"
        "- Do not invent extra keys. Only the fields shown above.\n"
        "Return ONLY the JSON."
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
            {"role": "user", "content": user_msg}
        ],
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 1024
        },
        "stream": False,   # <-- add this line
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
    except Exception:
        print(f"⚠️ Could not parse Ollama JSON: {cleaned[:400]}")       

    return dashboard

def validate_with_pydantic(data: dict) -> Tuple[bool,str,Optional[DashboardSpec]]:
    try:
        spec = DashboardSpec(**data)
        print("[DEBUG] Pydantic validation → OK")
        return True, "", spec
    except ValidationError as e:
        print(f"[ERROR] Pydantic validation → {e}")
        return False, str(e), None
    
# ----------------------------
# POST /ask: run the agent (MCP-only)
# ----------------------------
@app.post("/ask")
async def ask_sql(req: PromptRequest):
    if not hasattr(app.state, "executor"):
        raise HTTPException(status_code=503, detail="Agent not ready")

    ts = now_timestamp()
    start = time.perf_counter()

    raw_input = (req.prompt or "").strip()
    user_input = normalize_synonyms(raw_input)
    print(f"🧠 USER INPUT → {_short(user_input)}")

    # Clear ExecuteQueryGuard schema cache for this run (if present)
    exec_tool = app.state.tools.get("execute_query")
    if exec_tool and hasattr(exec_tool, "_schema_cache"):
        try:
            exec_tool._schema_cache = {}
        except Exception as e:
            print(f"⚠️ could not clear schema cache: {e}")

    # Clear per-run seen for schema guard so repeats are only flagged within a single request
    schema_tool = app.state.tools.get("get_table_schema")
    if schema_tool and hasattr(schema_tool, "_seen_this_run"):
        schema_tool._seen_this_run = set()

    # ---------- Agent invocation ----------
    try:
        result: dict = await asyncio.wait_for(
            app.state.executor.ainvoke({"input": user_input}),
            timeout=45
        )
    except asyncio.TimeoutError:
        record_metric(prompt=req.prompt, start_ts=start, success=False, error_msg="Timeout while processing your request.", iterations=0, function_times={})
        return {"status": "error", "timestamp": ts, "sql": None, "summary": "Timeout while processing your request."}
    except Exception as e:
        msg = str(e)
        record_metric(prompt=req.prompt, start_ts=start, success=False, error_msg=msg, iterations=0, function_times={})
        return {"status": "error", "timestamp": ts, "sql": None, "summary": msg}

    steps = result.get("intermediate_steps", [])
    results_payload: List[Dict[str, Any]] = []

    # Collect all successful execute_query calls (even if model stopped early)
    for action, observation in steps:
        if getattr(action, "tool", None) == "execute_query":
            ti = getattr(action, "tool_input", None)
            params = {}
            if isinstance(ti, dict):
                params = ti.get("kwargs", ti)
            elif isinstance(ti, str):
                s = ti.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        obj = json.loads(s)
                        params = obj.get("kwargs", obj) if isinstance(obj, dict) else {}
                    except Exception:
                        params = {}
            sql = params.get("query") or params.get("sql")
            if isinstance(sql, list):
                sql = " ".join(str(x) for x in sql)
            if isinstance(sql, str) and sql.strip():
                results_payload.append({
                    "sql": sql.strip(),
                    "viz": {
                        "type": (params.get("chart_type") or "table"),
                        "format": (params.get("format") or "table"),
                        "title": (params.get("title") or "Result"),
                    }
                })

    final_message = result.get("output")
    print(f"📝 FINAL MESSAGE → {_short(final_message)}")

    # Prefer explicit Final Answer with results[] if present
    fa_obj = _parse_final_answer_obj(final_message) if isinstance(final_message, str) else None
    existing_sqls = {r.get("sql") for r in results_payload if isinstance(r, dict) and isinstance(r.get("sql"), str)}

    if isinstance(fa_obj, dict) and isinstance(fa_obj.get("results"), list):
        for raw_item in fa_obj["results"]:
            norm = _normalize_result_item(raw_item)
            if norm and norm["sql"] not in existing_sqls:
                results_payload.append(norm)
                existing_sqls.add(norm["sql"])

    # If nothing collected, attempt single-SQL rescue
    if not results_payload:
        rescued_sqls: List[str] = []
        # ... your rescue extraction code ...
        if not rescued_sqls:
            msg = "No valid SQL was produced."
            record_metric(prompt=req.prompt, start_ts=start, success=False, error_msg=msg,
                          iterations=len(steps), function_times={})
            return {"status": "error", "timestamp": ts, "sql": None, "summary": msg}

        # Build a minimal results payload from rescued SQLs
        dedup_rescued: List[Dict[str, Any]] = []
        seen: set = set()
        for q in rescued_sqls:
            if q and q not in seen:
                seen.add(q)
                dedup_rescued.append({
                    "sql": q,
                    "viz": {"type": "table", "format": "table", "title": "Result"}
                })

        record_metric(prompt=req.prompt, start_ts=start, success=True, error_msg="",
                      iterations=len(steps), function_times={})
        return {
            "status": "success",
            "timestamp": ts,
            "sql": ";\n".join([r["sql"] for r in dedup_rescued]),
            "summary": "Final Answer: " + json.dumps({"results": dedup_rescued}),
        }

    # ---------- NORMAL PATH (results collected) ----------
    # Deduplicate, normalize viz, and cap to 4 panels
    dedup: List[Dict[str, Any]] = []
    seen_sql = set()
    for r in results_payload:
        s = r.get("sql")
        if isinstance(s, str) and s not in seen_sql:
            seen_sql.add(s)
            viz = r.get("viz") or {}
            if "title" not in viz:
                viz["title"] = "Result"
            r["viz"] = viz
            dedup.append(r)
    dedup = dedup[:4]

    # Always build the Final Answer JSON the client expects
    final_answer = {"results": dedup}
    sql_preview = ";\n".join([r["sql"] for r in dedup])

    print("📝 Final Answer SQL preview:")
    print(sql_preview[:200] + ("..." if len(sql_preview) > 200 else ""))

   
    print(f"📝 Building dashboard JSON with {len(dedup)} panels (Ollama) + validate + push...")
    extra = {}
    fn_times: Dict[str, int] = {}

    try:
        # 6) Generate dashboard JSON with Ollama
        sqls        = [r["sql"] for r in dedup]
        chart_types = [(r.get("viz") or {}).get("type", "table") for r in dedup]
        titles      = [(r.get("viz") or {}).get("title", "Panel") for r in dedup]

        dashboard_obj = time_call(
            fn_times, "ollama_generate",
            generate_dashboard_with_ollama,
            sqls=sqls,
            chart_types=chart_types,
            titles=titles,
            dashboard_title=f'{(req.prompt or "Dashboard")} - {ts}',
            time_from="now-5y",
            time_to="now",
        )
        print(f"✅ Dashboard JSON created, title: {dashboard_obj.get('dashboard', {}).get('title', 'No title')}")

        # 7) Pydantic validation (or pass-through)
        ok, err, validated = time_call(fn_times, "validate", validate_with_pydantic, dashboard_obj)
        if not ok:
            raise RuntimeError(f"Final dashboard invalid: {err}")

        # Build the outgoing JSON payload
        validated_json = validated.dict(by_alias=True) if hasattr(validated, "dict") else validated
        print(f"✅ Dashboard JSON validated, keys: {list(validated_json.keys())}")
        # 8) Push to Grafana MCP
        graf = time_call(
            fn_times, "post_grafana", requests.post,
            os.getenv("GRAFANA_MCP_URL", "http://grafana-mcp:8000/create_dashboard"),
            headers={"Content-Type": "application/json"},
            json=validated_json,
            timeout=30
        )
        graf.raise_for_status()
        print(f"✅ Dashboard JSON pushed to Grafana MCP, status: {graf.status_code}")
        # Some MCP servers return text rather than JSON
        try:
            graf_result = graf.json()
        except Exception:
            graf_result = {"text": graf.text}

        # Expose details in the response
        extra.update({
            "grafana_result": graf_result,
            "validated": (validated.dict(by_alias=True, exclude_none=True)
                        if hasattr(validated, "dict") else validated),
            "function_times_ms": fn_times
        })
        print(f"[DEBUG] dashboard keys: {list(dashboard_obj.keys())} "
            f"- validated keys: {list(validated_json.keys())}")

    except Exception as e:
        print(f"❌ Dashboard step failed: {e}")
        extra["dashboard_error"] = str(e)



    record_metric(prompt=req.prompt, start_ts=start, success=True, error_msg="",
                iterations=len(steps), function_times={})
    return {
        "status": "success",
        "timestamp": ts,
        "sql": sql_preview,
        "summary": "Final Answer: " + json.dumps(final_answer),
        **extra,
    }


