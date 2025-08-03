from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import requests, json, os, time, re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from src.models import DashboardSpec
from rag.context_retriever import get_table_schema, get_table_example
from src.metrics import record_metric, time_call

from fixation import (
    prompt_mentions_time,
    remove_time_grouping,
    fix_table_names,
    fix_join_conditions,
    fix_column_references,
    fix_group_by_clause,
    fix_syntax_issues,
    validate_and_correct_sql
)

app = FastAPI()

OLLAMA_TIMEOUT = 300  # 5 minutes
MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/v1/chat/completions")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# Will be populated at startup
valid_tables: List[str] = []

class PromptRequest(BaseModel):
    prompt: str

def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_control_chars(s: str) -> str:
    return "".join(ch for ch in s if ch in "\r\n\t" or 32 <= ord(ch) <= 126)

def validate_sql_with_ollama(
    raw_sql: str,
    schema_parts: list[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate a Grafana PostgreSQL SQL query via Ollama.

    Checks only for:
      1. Syntax errors (missing commas, parentheses, semicolons, etc.)
      2. Use of any alias not declared in FROM/JOIN clauses
      3. References to any table or column not listed in `schema_parts`
      4. Any JOIN missing an ON or USING clause

    Returns:
      (True, None)                   if the query is valid
      (False, "reason why invalid")  otherwise

    Example of schema_parts:
      [
        "Table: person (id, name, city)",
        "Table: sales  (id, person_id, value, sale_date)",
        "Table: products (id, name, price)"
      ]
    """

    # System prompt describes exactly what to validate—and what to ignore.
    system_prompt = """
    You are a Grafana PostgreSQL validator. Do not suggest best practices,
    stylistic issues, or semantic tips about LEFT vs INNER joins, ORDER BY alias usage, etc.

    Check only for:
    1. Syntax errors (missing commas, parentheses, semicolons, etc.)
    2. Use of any alias that was not declared in a FROM or JOIN clause
    3. References to any table or column not listed in the schema below
    4. Any JOIN missing an ON or USING clause

    If the query passes all these checks, reply with exactly:
    VALID

    Otherwise reply with exactly:
    INVALID: <one-sentence reason>

    Do not include anything else.

    Schema:
    """ + "\n".join(schema_parts)

    # A few-shot to teach the exact behavior
    few_shot = [
        {"role": "user", "content": (
            "-- Example 1 (VALID)\n"
            "SELECT p.name AS customer_name\n"
            "FROM person AS p\n"
            "JOIN sales  AS s ON p.id = s.person_id;\n"
        )},
        {"role": "assistant", "content": "VALID\n"},
        {"role": "user", "content": (
            "-- Example 2 (INVALID)\n"
            "SELECT p.name, city\n"
            "FROM person AS p;\n"
        )},
        {"role": "assistant", "content": (
            "INVALID: Unqualified column \"city\" (no alias p.city)\n"
        )}
    ]

    # The user-provided SQL to validate
    user_prompt = f"SQL to validate:\n```sql\n{raw_sql}\n```"

    # Build the message sequence
    messages = (
        [{"role": "system", "content": system_prompt}]
        + few_shot
        + [{"role": "user",   "content": user_prompt}]
    )

    # Call Ollama
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":       OLLAMA_MODEL,
            "messages":    messages,
            "temperature": 0.0,
            "stream":      False
        },
        timeout=OLLAMA_TIMEOUT
    )
    resp.raise_for_status()

    # Clean and parse the result
    content = clean_control_chars(
        resp.json()["choices"][0]["message"]["content"].strip()
    )
    print(f"[VALIDATION] Validator SQL:{raw_sql} . \n response: {content}")
    if content == "VALID":
        return True, None

    if content.startswith("INVALID:"):
        # Extract text after the colon
        return False, content[len("INVALID:"):].strip()

    # Fallback for unexpected replies
    return False, f"Unexpected validator response: {content}"

def fallback_table_match(prompt: str) -> List[str]:
    mappings = {
        "customer": "person",
        "user":     "person",
        "product":  "products",
        "item":     "products",
        "sale":     "sales",
        "order":     "orders",
        "payment method":    "payment_type",
    }
    return [tbl for kw, tbl in mappings.items() if kw in prompt.lower()]

def extract_table_name_with_llm(prompt: str, all_tables:[str] ) -> str:
    try:
        # tell the LLM which tables are allowed
        resp = requests.post(
            "http://ollama:11434/v1/chat/completions",
            json={
                "model": "llama3.1",
                "messages": [
                    {"role": "system",
                     "content": (
                        "You are a SQL assistant. Only extract table names from this "
                        "whitelist: " + 
                        ", ".join(all_tables) + ".\n"
                        "If the prompt says 'payment method', map it to 'payment_type'.\n"
                        "If the prompt says 'customer' or 'deliverer', map it to 'person'.\n"
                        "If the prompt says 'product', map it to 'products'.\n"
                        "If the prompt says 'order', map it to 'orders'.\n"
                        "If the prompt says 'city', map it to 'person'.\n"
                     )},
                    {"role": "user",
                     "content": f"Question: {prompt}\n\n"
                                "Return comma-separated table identifiers only."}
                ],
                "temperature": 0.0,
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        # strip backticks/spaces, ensure valid identifiers
        # tables = [t.strip(" `") for t in re.split(r"[,\n]+", raw) if t.strip(" `").isalnum()]
        # return ",".join(tables)
        tables = [t.strip(" `") for t in re.split(r"[,\n]+", raw) 
                 if t.strip(" `") and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', t.strip(" `"))]
        return ",".join(tables)
    except Exception as e:
        print(f"[ERROR] extract_table_name_with_llm: {e}")
        return ""


def get_panel_intents(prompt: str, schema_parts: List[str]) -> List[Dict[str, str]]:
    
    mapping_rules = """
    1. NEVER use "bar" – always "barchart"
    2. NEVER use "pie" – always "piechart"
    3. Only types: barchart, piechart, table, timeseries, stat
    4. bar/bar chart/bars → barchart
       pie/pie chart → piechart
       table/list/stat/gauge → table
       time series/over time → timeseries
       total/sum/count/statistic → stat
    """
    resp = requests.post(
        "http://ollama:11434/v1/chat/completions",
        json={
            "model": "llama3.1",
            "messages": [{
                "role": "user",
                "content": (
                    "You are a Grafana dashboard planner.\n"
                    f"Schema:\n{chr(10).join(schema_parts)}\n\n"
                    f"Request: {prompt}\n"
                    f"{mapping_rules}\n"
                    "Return ONLY a JSON array of objects with keys type+title."
                )
            }],
            "temperature": 0.1,
            "stream": False
        },
        timeout=OLLAMA_TIMEOUT
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in get_panel_intents")
    arr = json.loads(clean_control_chars(match.group(0)))
    # print(f"[DEBUG] get_panel_intents → {arr}")
    return arr


def call_llm(messages: List[Dict[str,str]]) -> str:
    """
    Generic LLM call – uses configured URL & model.
    Expects messages in [{role:..., content:...}] chat format.
    Returns the raw content string.
    """
    resp = requests.post(
        "http://ollama:11434/v1/chat/completions",
        json={"model": "llama3.1", "messages": messages, "temperature": 0.1},
        timeout=OLLAMA_TIMEOUT
    )
    resp.raise_for_status()
    choice = resp.json().get("choices", [])
    if not choice:
        raise RuntimeError("LLM returned no choices")
    return choice[0]["message"]["content"]

def remove_code_fences(output: str) -> str:
    """
    Remove leading/trailing markdown code fences (```) and optional language tags from the LLM output.
    """
    output_stripped = output.strip()
    if output_stripped.startswith("```"):
        # Remove the opening triple backticks
        content = output_stripped[3:]
        # Remove known language tags (e.g., "sql", "json") followed by whitespace or newline
        content = re.sub(r"^(?:sql|json)\s", "", content, flags=re.IGNORECASE)
        # Remove the closing triple backticks if present
        end_idx = content.rfind("```")
        if end_idx != -1:
            content = content[:end_idx]
        return content.strip()
    return output_stripped

def generate_sql_for_panel(
    intent: Dict[str, str],
    schema_parts: List[str],
    prompt: str,
    table_columns: Dict[str,List[str]],
    allowed_tables: List[str]
) -> Dict[str, str]:
    '''
    MCP is a specification (like a set of rules or instructions) for how to register and invoke tools in a way that LLMs can understand.
    LLMs are language models; they don’t have “native” database access or network rights.
    They can only use external data if you provide:
    A tool/function definition (like MCP, OpenAI function calling, or Anthropic’s Tool Use),
    And a “handler” (code) that actually performs the real action.
    If you mean no custom coding, but using a platform:
        Some LLM platforms let you connect tools visually (e.g., LangChain, LlamaIndex, Azure AI Studio, OpenAI “actions,” Zapier plugins).
        You still need to register your database connection and describe your tables/tools.
    LangChain or LlamaIndex (Visual or Config-Driven)
    These frameworks have “database agents” where you point to your DB, and the framework generates the tool wrappers.
    Still requires you to run/configure the agent, but almost no code.
    '''
    panel_type, title = intent["type"], intent["title"]
    
    # Build allowed tables schema as explicit as possible
 
    schema_prompt = (
        "ALLOWED TABLES AND COLUMNS AND EXAMPLE (use ONLY these):\n"
        f"{schema_parts}\n"
        "If a table/column is NOT in this list, DO NOT use it."
    )
    
    # print(f"[SQL] Generating SQL for schema_parts: {schema_parts}. schema prompt :{schema_prompt} \n")
    sql_rules_list = []
    sql_rules_list.append(
        "MUST NEVER select any column ending in `_id` or named `id` in the final SELECT clause. "
        "Instead: 1) Identify the referenced table (e.g., `person_id` → `person` table), "
        "2) JOIN that table with appropriate alias (e.g., `person AS pe`), "
        "3) Select the human-readable field (e.g., `pe.name AS person_name`)."
    )
    sql_rules_list.append("MUST assign a unique alias to every expression in the SELECT clause (e.g. EXTRACT(YEAR FROM <date_column>)::text AS year).")
    sql_rules_list.append("MUST alias every table in FROM/JOIN and use that alias for *all* column references (e.g. sales AS s → s.sale_date).")
    sql_rules_list.append("MUST cast grouping expressions in SELECT only (e.g. EXTRACT(YEAR FROM s.sale_date)::text AS year) and NEVER cast in GROUP BY or ORDER BY.")
    sql_rules_list.append("MUST include every non-aggregated SELECT column in the GROUP BY clause.")
    sql_rules_list.append("MUST only select or group by columns that appear in the provided schema.")
    
    
    sql_rules_list.append("MUST not include a semicolon at end of SQL.")
    
    if intent["type"] in ("barchart","piechart"):
        sql_rules_list.append(
            "For barchart and piechart, when grouping by EXTRACT(YEAR…), follow the cast rule above."
        )
    
    sql_rules_list.append(
        "For any chart or timeseries request involving date grouping, "
        "MUST use DATE_TRUNC('day', <date_column>) AS time for daily, "
        "DATE_TRUNC('month', <date_column>) AS time for monthly, "
        "DATE_TRUNC('year', <date_column>) AS time for yearly, etc. "
        "MUST NOT use EXTRACT to separate year/month/day for grouping, "
        "unless the prompt specifically requests it."
    )
    if intent["type"] == "timeseries" or "date" in prompt.lower():
        sql_rules_list.append(
            "For timeseries, alias the time column as time (e.g. DATE_TRUNC('day', <date_column>) AS time)."
        )
    sql_rules_list.append("Do NOT emit any explanatory text.")
    sql_rules_list.append("NEVER return explanations — only JSON like: {\"rawSql\": \"...\", \"format\": \"...\"}.")
    
 
    system_msg = (
        "You are a strict Postgres SQL generator for Grafana dashboards.\n"
        f"{schema_prompt}\n"
        "RULES:\n" + "\n".join(sql_rules_list) + "\n"
        
    )

    messages = [
        {"role":"system", "content": system_msg},
        {"role":"user",   "content": f"User request: {title}"}
    ]

    raw = call_llm(messages)
    # extract JSON blob…
    raw_stripped = clean_control_chars(raw)

    # Try to extract JSON object from the LLM output
    match = re.search(r"\{[\s\S]*?\}", raw_stripped)
    raw_sql = ""
    if match:
        try:
            spec = json.loads(clean_control_chars(match.group(0)))
            raw_sql = spec.get("rawSql", "")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON from LLM output: {e}")
            raw_sql = ""  # JSON parsing failed, will fallback to raw SQL handling

    # **Fallback:** If no JSON was found or parsing failed, handle the output as raw SQL (possibly in code fences)
    if not raw_sql:
        cleaned_output = remove_code_fences(raw_stripped)
        if cleaned_output.startswith("{"):
            # If the cleaned output is JSON, parse it
            try:
                spec = json.loads(clean_control_chars(cleaned_output))
                raw_sql = spec.get("rawSql", "")
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON in fenced content: {e}")
                raw_sql = cleaned_output
        else:
            # Otherwise, treat the cleaned output as the SQL query
            raw_sql = cleaned_output
    
    # Ensure raw_sql is a string and strip any trailing semicolon
    if not isinstance(raw_sql, str):
        raw_sql = str(raw_sql) if raw_sql is not None else ""
    raw_sql = raw_sql.rstrip().rstrip(";")
    # print(f"[DEBUG] Raw SQL extracted: {raw_sql}, type: {panel_type} ")
    # now validate/correct
    # final_sql = validate_and_correct_sql(
    #     raw_sql, prompt, allowed_tables, table_columns
    # )
    # ok, reason = validate_sql_with_ollama(raw_sql, schema_parts)
    
    # if ok:
    #     print("SQL is valid ✅")
    # else:
    #     print("SQL is invalid ❌")
    #     print("Reason:", reason)
    
    fmt_map = {
        "barchart":"table","piechart":"table","table":"table",
        "timeseries":"time_series","stat":"table", "gauge": "table"
    }
    # print(f"[DEBUG] Generated SQL for panel '{title}': {raw_sql}")
    return {"rawSql": raw_sql, "format": fmt_map[panel_type]}


def fetch_table_columns(table: str) -> List[str]:
    """
    Retrieve the list of column names for `table` by calling MCP’s /schema endpoint.
    """
    try:
        r = requests.get(f"{MCP_URL}/schema/{table}", timeout=10)
        r.raise_for_status()
        # MCP returns {"schema": ["col1 (type)", "col2 (type)", ...]}
        schema_list = r.json().get("schema", [])
        cols = [entry.split(" ", 1)[0] for entry in schema_list if entry]
        return cols
    except Exception as e:
        print(f"[ERROR] fetch_table_columns({table}): {e}")
        return []


def validate_with_pydantic(data: dict) -> Tuple[bool,str,Optional[DashboardSpec]]:
    try:
        spec = DashboardSpec(**data)
        print("[DEBUG] Pydantic validation → OK")
        return True, "", spec
    except ValidationError as e:
        print(f"[ERROR] Pydantic validation → {e}")
        return False, str(e), None

@app.on_event("startup")
def load_valid_tables():
    global valid_tables
    try:
        resp = requests.get(f"{MCP_URL}/tables", timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        valid_tables = resp.json().get("tables", [])
        print(f"[DEBUG] valid MCP tables → {valid_tables}")
    except Exception as e:
        print(f"[ERROR] loading valid_tables: {e}")
        valid_tables = []
        
def generate_dashboard_with_ollama(
    sqls: list[str],
    chart_types: list[str],
    titles: list[str],
    dashboard_title: str,
    time_from: str = "now-5y",
    time_to: str = "now"
) -> dict:
    """
    Given lists of SQLs, chart types, and panel titles, asks Ollama to assemble a full Grafana dashboard JSON.
    Returns the parsed dashboard dict.
    """
    # Assemble panel specifications
    panels = []
    for idx, (sql, chart_type, title) in enumerate(zip(sqls, chart_types, titles)):
        panels.append({
            "title": title,
            "type": chart_type,         # e.g. "bar chart", "pie chart", etc (map if needed)
            "format": "table" if chart_type in ("barchart", "piechart", "table", "stat", "gauge") else "time_series",
            "rawSql": sql
        })

    # Compose prompt for Ollama
    system_msg = (
        "You are an expert in Grafana dashboard JSON (schemaVersion 36). "
        "Given the panel specs below, generate a valid dashboard JSON with this EXACT structure:\n"
        "{\n"
        "  \"dashboard\": {\n"
        "    \"schemaVersion\": 36,\n"
        "    \"title\": \"<dashboard_title>\",\n"
        "    \"time\": {\n"
        "      \"from\": \"<time_from>\",\n"
        "      \"to\": \"<time_to>\"\n"
        "    },\n"
        "    \"refresh\": \"30s\",\n"
        "    \"panels\": [\n"
        "      {\n"
        "        \"id\": 0,\n"
        "        \"title\": \"Panel Title\",\n"
        "        \"type\": \"panel_type\",\n"
        "        \"gridPos\": {\"x\": 0, \"y\": 0, \"w\": 12, \"h\": 8},\n"
        "        \"rawSql\": \"SQL_QUERY_HERE\",\n"
        "        \"format\": \"table_or_time_series\"\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n"
        "Requirements for panels:\n"
        "- Each panel gets a unique integer id (starting from 0, incrementing by 1).\n"
        "- Each panel gets a gridPos field for 2-column layout: "
        "x=0, y=ROW*8, w=12 for left; x=12, y=ROW*8, w=12 for right; "
        "if there is an odd number of panels, the last panel is full width: x=0, w=24, y=(N//2)*8.\n"
        "- All panels have h=8 in gridPos.\n"
        "- Include rawSql and format fields directly in each panel (they will be transformed later).\n"
        "- Use the user-provided SQL, chart_type, and title for each panel as given.\n"
        "- IMPORTANT: panels must be INSIDE the dashboard object, not at the root level.\n"
        "Return ONLY the JSON, no explanation, no markdown fences."
    )

    user_msg = (
        f"dashboard_title: {dashboard_title}\n"
        f"time_from: {time_from}\n"
        f"time_to: {time_to}\n"
        f"panels: {json.dumps(panels, indent=2)}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "messages": messages, "temperature": 0.0},
        timeout=OLLAMA_TIMEOUT
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Extract JSON safely (handle possible code fences, etc)
    try:
        # Clean the response using the existing remove_code_fences function
        cleaned_raw = remove_code_fences(raw)
        
        # If it still looks like it might have language tags, handle them
        if cleaned_raw.startswith(('json', 'JSON')):
            # Remove language tag at the beginning
            lines = cleaned_raw.split('\n')
            if lines[0].strip().lower() in ('json', 'JSON'):
                cleaned_raw = '\n'.join(lines[1:]).strip()
        
        # Try to parse as JSON
        dashboard = json.loads(cleaned_raw)
        
        # Fix structure if panels are at root level instead of inside dashboard
        if "panels" in dashboard and "dashboard" in dashboard:
            if "panels" not in dashboard["dashboard"]:
                dashboard["dashboard"]["panels"] = dashboard.pop("panels")
        print(f"[DASHBOARD] Generated dashboard JSON: {json.dumps(dashboard, indent=2)}")
        return dashboard
        
    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract JSON object manually
        try:
            # Look for the opening brace and extract everything from there
            start_idx = raw.find('{')
            if start_idx != -1:
                # Find the matching closing brace
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(raw[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                json_str = raw[start_idx:end_idx]
                dashboard = json.loads(json_str)
                return dashboard
            else:
                raise RuntimeError(f"No JSON object found in response: {raw}")
                
        except Exception as inner_e:
            raise RuntimeError(f"Ollama did not return valid dashboard JSON: {e}\nInner error: {inner_e}\nRaw response: {raw}")
    
    except Exception as e:
        raise RuntimeError(f"Ollama did not return valid dashboard JSON: {e}\nRaw response: {raw}")
    
@app.post("/ask")
def ask_ollama(req: PromptRequest):
    start = time.perf_counter()
    fn_times: Dict[str,int] = {}
    prompt, ts = req.prompt, now_timestamp()
    success, err_msg = False, ""
    try:
        # 1) Infer tables
        resp = requests.get(f"{MCP_URL}/tables", timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        all_tables = resp.json().get("tables", [])
        print(f"[TABLE EXTRACTION] Valid tables from MCP: {all_tables}")
        extracted = time_call(fn_times, "extract_tables", extract_table_name_with_llm, prompt, all_tables)
        raw = [t for t in extracted.split(",") if t] or fallback_table_match(prompt)
        tables = [t.lower() for t in dict.fromkeys(raw)]
        # unknown = [t for t in tables if t not in valid_tables]
        # if unknown:
        #     raise HTTPException(400, detail=f"Unknown table(s): {unknown}")
        print(f"[DEBUG] Tables extracted: {tables}")    
        # 2) Fetch schema context & column lists
        schema_parts, table_columns = [], {}
        for t in tables:
            sch = time_call(fn_times, f"get_schema_{t}", get_table_schema, t)
            ex  = time_call(fn_times, f"get_example_{t}", get_table_example, t)
            schema_parts.append(f"Table: {t}\n{sch}\nExample Row: {json.dumps(ex)}")
            cols = time_call(fn_times, f"fetch_table_columns_{t}", fetch_table_columns, t)
            table_columns[t] = cols
        print(f"[DEBUG] Schema parts: {schema_parts}")
        # 3) Plan panels
        intents = time_call(fn_times, "get_panel_intents", get_panel_intents, prompt, schema_parts)

        # 4) Generate & validate SQL
        sqls, chart_types_for_mcp, titles = [], [], []
        # mcp_type_map = {"barchart":"bar chart","piechart":"pie chart","timeseries":"line chart","table":"table","stat":"table", "gauge":"table"}
        mcp_type_map = {
            "barchart": "barchart",
            "piechart": "piechart",
            "timeseries": "timeseries",
            "table": "table",
            "stat": "stat",
            "gauge": "gauge"
        }


        for intent in intents:
            # print(f"[DEBUG] Processing intent: {intent}")
            spec = time_call(
                fn_times, f"gen_sql_{intent['title']}",
                generate_sql_for_panel,
                intent, schema_parts, prompt,
                table_columns, tables
            )
            sqls.append(spec["rawSql"])
            chart_types_for_mcp.append(mcp_type_map[intent["type"]].lower())
            titles.append(intent["title"])

        # 5) Call MCP for Grafana JSON
        full_title = f"{', '.join(titles)} - {ts}"
        short_title = full_title[:10]
        # payload = {"sql": sqls, "chart_type": chart_types_for_mcp, "title": f"{short_title} - {ts}"}
        # mcp_resp = time_call(fn_times, "call_mcp", requests.post,
        #                      f"{MCP_URL}/grafana_json", json=payload, timeout=OLLAMA_TIMEOUT)
        # mcp_resp.raise_for_status()
        # dashboard = mcp_resp.json()
        
        # # 6) Inject panel IDs and proper layout
        # total_panels = len(dashboard["dashboard"]["panels"])
        # for i, p in enumerate(dashboard["dashboard"]["panels"]):
        #     p["id"] = i

        #     if i == total_panels - 1 and total_panels % 2 == 1:
        #         # Last panel in odd total - full width
        #         x, w = 0, 24
        #         y = (i // 2) * 8
        #     else:
        #         # Regular 2-panel layout
        #         x = 0 if i % 2 == 0 else 12
        #         w = 12
        #         y = (i // 2) * 8

        #     p["gridPos"] = {"x": x, "y": y, "w": w, "h": 8}
        # Call Ollama to assemble Grafana dashboard JSON
        dashboard = generate_dashboard_with_ollama(
            sqls=sqls,
            chart_types=chart_types_for_mcp,
            titles=titles,
            dashboard_title=short_title,
            time_from="now-5y",
            time_to="now"
        )
        
        # ---- PATCH TARGETS IN PANELS HERE ----
        for idx, panel in enumerate(dashboard["dashboard"]["panels"]):
            if "targets" not in panel:
                panel["targets"] = [{
                    "rawSql": panel.get("rawSql", ""),
                    "format": panel.get("format", "table"),
                    "refId": chr(ord('A') + idx)
                }]
            # PATCH: Inject minimal fieldConfig if missing
            if "fieldConfig" not in panel:
                panel["fieldConfig"] = {"defaults": {}, "overrides": []}



        # 7) Pydantic validation
        ok, err, validated = time_call(fn_times, "validate", validate_with_pydantic, dashboard)
        if not ok:
            raise RuntimeError(f"Final dashboard invalid: {err}")
        success = True       
            
        # 8) Push to Grafana
        graf = time_call(fn_times, "post_grafana", requests.post,
                         os.getenv("GRAFANA_MCP_URL", "http://grafana-mcp:8000/create_dashboard"),
                         headers={"Content-Type":"application/json"},
                         json=validated.dict(by_alias=True),
                         timeout=30)
        graf.raise_for_status()
        print(f"[DEBUG] dashboard: {dashboard} - \n validated: {validated.dict(by_alias=True, exclude_none=True)}")
        return {"status":"success","timestamp":ts,
                "function_times_ms":fn_times,
                "grafana_result":graf.json(),
                "validated":validated.dict(by_alias=True,exclude_none=True)}
    except Exception as exc:
        print(f"[ERROR] ask_ollama failed: {exc}")
        return {"status":"error","error":str(exc),"function_times_ms":fn_times}
    finally:
        record_metric(
            prompt=prompt, start_ts=start,
            success=success, error_msg=err_msg,
            iterations=1, function_times=fn_times
        )
