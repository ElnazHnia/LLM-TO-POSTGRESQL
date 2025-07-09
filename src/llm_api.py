from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import requests
import json
from datetime import datetime
from typing import List, Dict
import time
# Import your models (use the fixed models from above)
from src.models import DashboardSpec, Panel, Target, GridPos, PanelOptions, ReduceOptions
from rag.context_retriever import get_table_schema, get_table_example 
from src.metrics import record_metric, time_call
from collections import Counter, defaultdict
import re
import json

app = FastAPI()

OLLAMA_TIMEOUT = 300  # 5 minutes timeout for Ollama requests         

class PromptRequest(BaseModel):
    prompt: str
    
# Create a global timestamp when the request starts
def now_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fallback_table_match(prompt: str) -> List[str]:
    mappings = {
        "customer": "person",
        "user": "person",
        "client": "person",
        "product": "products",
        "item": "products",
        "sales": "sales",
        "transaction": "sales",
    }
    found = set()
    for keyword, table in mappings.items():
        if keyword in prompt.lower():
            found.add(table)
    return list(found)

def extract_table_name_with_llm(prompt: str) -> str:
    """Use LLM to extract table names from the prompt."""
    try:
        # Try the OpenAI-compatible API first
        response = requests.post(
            "http://ollama:11434/v1/chat/completions",
            json={
                "model": "llama3",
                "messages": [
                    {
                        "role": "user",
                        "content": f"You are a SQL assistant.\nGiven this question, extract the table name(s) mentioned.\n\nQuestion: {prompt}\n\nReturn only the table names, comma-separated if more than one. The sales table contains person_id, product_id, sale_date is a DATE, and value. The person table stores id and name of customers or deliverer. The products table has id and name."
                    }
                ],
                "temperature": 0.1,
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            # OpenAI-compatible API response
            result = response.json()
            raw_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().lower()
        else:
            # Fallback to native Ollama API
            response = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": f"You are a SQL assistant.\nGiven this question, extract the table name(s) mentioned.\n\nQuestion: {prompt}\n\nReturn only the table names, comma-separated if more than one. The sales table contains person_id, product_id, sale_date is a DATE, and value. The person table stores id and name of customers. The products table has id and name.",
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=OLLAMA_TIMEOUT
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            raw_response = result.get("response", "").strip().lower()
        
        print("Raw response from table extractor:", raw_response)
        
        # Extract just the table names
        table_line = raw_response.split("\n")[-1]  # get the last line (which should contain only the names)
        clean_tables = [name.strip() for name in table_line.split(",") if name.strip().isidentifier()]

        return ",".join(clean_tables)
        
    except Exception as e:
        print(f"Error in extract_table_name_with_llm: {e}")
        # Fallback to basic table extraction
        prompt_lower = prompt.lower()
        tables = []
        if "sales" in prompt_lower or "sale" in prompt_lower:
            tables.append("sales")
        if "person" in prompt_lower or "customer" in prompt_lower:
            tables.append("person")
        if "product" in prompt_lower:
            tables.append("products")
        return ",".join(tables) if tables else "sales"


def create_structured_prompt(prompt: str, schema_parts: List[str], timestamp: str) -> str:
    """Create a structured prompt that requests JSON output matching our Pydantic schema"""
    
    # Create the JSON schema for the LLM to follow
    json_schema = {
        "dashboard": {
            "title": f"String with timestamp - {timestamp}",
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
                    "datasource": {"type": "postgres", "uid": "cer8a03ztsu0wf"},
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
    
    return f"""
{chr(10).join(schema_parts)}

User question: {prompt}

CRITICAL: The dashboard title MUST EXACTLY be: "[Your descriptive title] - {timestamp}"

CRITICAL PANEL TYPE RULES - MUST BE FOLLOWED EXACTLY:
1. NEVER use "bar" - always use "barchart"
2. NEVER use "pie" - always use "piechart" 
3. Valid panel types are ONLY: "barchart", "piechart", "table", "timeseries", "stat"
4. Chart Type Mapping:
   - If user mentions "bar chart", "bar", "bars" → use type: "barchart" with format: "table"
   - If user mentions "pie chart", "pie" → use type: "piechart" with format: "table"
   - If user mentions "table", "list" → use type: "table" with format: "table"
   - If user mentions "time series", "over time" → use type: "timeseries" with format: "time_series"
   - If user mentions "total", "sum", "count", "statistic" → use type: "stat" with format: "table"

5. SQL Rules:
   - For time-based queries (format: "time_series"), alias timestamp as "time": DATE_TRUNC('day', sale_date) AS time
   - Cast numeric grouping fields to text: EXTRACT(YEAR FROM sale_date)::text AS year
   - No semicolons at end of SQL
   - Always prefer displaying human-readable fields (like names) over IDs:
    - person.name AS customer_name
    - products.name AS product_name

6. GRID POSITIONS - CRITICAL LAYOUT RULES:
   ALWAYS arrange panels in pairs (2 per row). Only the final panel gets full width if total is odd.
   
   MATHEMATICAL FORMULA - Follow this EXACTLY:
   ```
   for i in range(total_panels):
       if i == total_panels - 1 and total_panels % 2 == 1:
           # Last panel in odd total - full width
           x = 0
           w = 24  
           y = (i // 2) * 8
       else:
           # Regular 2-panel layout
           x = 0 if i % 2 == 0 else 12
           w = 12
           y = (i // 2) * 8
       
       panels[i].gridPos = {{"x": x, "y": y, "w": w, "h": 8}}
   ```
   
   EXAMPLES:
   - 1 panel: Panel 0 (x=0, y=0, w=24, h=8) - FULL WIDTH
   - 2 panels: Panel 0 (x=0, y=0, w=12, h=8), Panel 1 (x=12, y=0, w=12, h=8) - SIDE BY SIDE
   - 3 panels: Panel 0 (x=0, y=0, w=12, h=8), Panel 1 (x=12, y=0, w=12, h=8), Panel 2 (x=0, y=8, w=24, h=8) - LAST ONE FULL WIDTH
   - 4 panels: All panels w=12, arranged in 2 rows of 2 panels each
   - 5 panels: First 4 panels w=12 (2 rows), last panel w=24 (full width)
   
   CRITICAL: Never put a single panel with w=12 alone on a row - it must be either paired (w=12) or full width (w=24)

7. Dashboard title MUST include the exact timestamp: "[Description] - {timestamp}"

8. IMPORTANT: Single panels or last panel in odd count should use FULL WIDTH (w=24, x=0)

9. Do **not** invent aliases.  Use table names directly
   (sales.value, person.name, products.name) or declare an alias in FROM.
   
REMEMBER: 
- The panel type must be exactly "barchart" (not "bar"), "piechart" (not "pie"), etc.
- The title MUST be formatted as: "[Your Description] - {timestamp}"

JSON Schema to follow:
{json.dumps(json_schema, indent=2)}

Return ONLY valid JSON matching this exact structure. Do not include any explanatory text before or after the JSON.
"""


def create_correction_prompt(original_prompt: str, schema_parts: List[str], json_output: str, error_message: str, timestamp: str) -> str:
    """Create a simplified prompt for Ollama to fix JSON based on validation error"""
    
    return f"""
You are a JSON validator and fixer. Your task is to analyze the JSON and fix the specific errors mentioned.

ORIGINAL USER REQUEST: {original_prompt}

DATABASE SCHEMA:
{chr(10).join(schema_parts)}

YOUR PREVIOUS JSON OUTPUT:
{json_output}

VALIDATION ERRORS TO FIX:
{error_message}

CRITICAL SQL FIXES NEEDED:
1. For time-based queries (format: "time_series"), alias timestamp as "time": DATE_TRUNC('day', sale_date) AS time

2. Table names must be PLURAL: sales, person, products (not sale, product)

3. Don't use column aliases in GROUP BY/ORDER BY unless they're simple column names

4. Title must be: "[Description] - {timestamp}"

5. Panel type must be exactly "barchart" (not "bar")

6. Format must be "table" for barchart panels

7. GRID POSITIONS - CRITICAL LAYOUT RULES:
   Follow these steps EXACTLY in order:
   
   Step 1: Count total panels: total_panels = len(panels)
   Step 2: Apply grid positioning rules:
   
8. GRID POSITIONS - CRITICAL LAYOUT RULES:
   ALWAYS arrange panels in pairs (2 per row). Only the final panel gets full width if total is odd.
   
   MATHEMATICAL FORMULA - Follow this EXACTLY:
   ```
   for i in range(total_panels):
       if i == total_panels - 1 and total_panels % 2 == 1:
           # Last panel in odd total - full width
           x = 0
           w = 24  
           y = (i // 2) * 8
       else:
           # Regular 2-panel layout
           x = 0 if i % 2 == 0 else 12
           w = 12
           y = (i // 2) * 8
       
       panels[i].gridPos = {{"x": x, "y": y, "w": w, "h": 8}}
   ```
   
   EXAMPLES:
   - 1 panel: Panel 0 (x=0, y=0, w=24, h=8) - FULL WIDTH
   - 2 panels: Panel 0 (x=0, y=0, w=12, h=8), Panel 1 (x=12, y=0, w=12, h=8) - SIDE BY SIDE
   - 3 panels: Panel 0 (x=0, y=0, w=12, h=8), Panel 1 (x=12, y=0, w=12, h=8), Panel 2 (x=0, y=8, w=24, h=8) - LAST ONE FULL WIDTH
   - 4 panels: All panels w=12, arranged in 2 rows of 2 panels each
   - 5 panels: First 4 panels w=12 (2 rows), last panel w=24 (full width)
   
   CRITICAL: Never put a single panel with w=12 alone on a row - it must be either paired (w=12) or full width (w=24)

SPECIFIC FIXES FOR YOUR SQL:
- Change: GROUP BY day ORDER BY day
- To: GROUP BY DATE_TRUNC('day', sale_date) ORDER BY DATE_TRUNC('day', sale_date)
- Or: GROUP BY 1 ORDER BY 1

Return ONLY the corrected JSON. No explanations.
"""


def call_ollama_structured(prompt: str) -> dict:
    """Call Ollama with a structured prompt and parse JSON response."""
    try:
        # Try OpenAI-compatible API first
        response = requests.post(
            "http://ollama:11434/v1/chat/completions",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.9,
                "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )

        if response.status_code == 200:
            result = response.json()
            message = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        

        # Attempt to parse it as JSON
        try:
            return json.loads(message)
        except json.JSONDecodeError as e:
            print("❌ JSON decode error:", e)
            print("🔎 Raw output from model:\n", message)
        
    
           
            # Extract the largest JSON block in the response
            # json_match = re.search(r'{[\s\S]*}', message)
            json_match = re.search(r'{[\s\S]+}', message)

            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError as inner_e:
                    print("⚠️ Still failed parsing extracted JSON:", inner_e)

            raise Exception(f"Invalid JSON from LLM: {str(e)}")

    except Exception as e:
        print(f"🔥 Error calling Ollama: {e}")
        raise


def validate_with_pydantic(dashboard_data: dict) -> tuple[bool, str, DashboardSpec]:
    """Validate dashboard data with Pydantic and return success status, error message, and object"""
    try:
        # Validate the dashboard data against the Pydantic model
        spec = DashboardSpec(**dashboard_data)
        return True, "", spec
    except ValidationError as e:
        return False, str(e), None


def ask_ollama_if_valid(
        prompt: str,
        schema: List[str],
        dashboard_json: str,      # dashboard you want inspected
        timestamp: str
) -> tuple[bool, str]:
    """
    Ask the LLM to (1) dump every SQL query it sees and (2) run the
    validation checklist – *including that each panel has the correct
    `"format"` value*.  The function returns (is_valid, first_error_msg).
    """

    check_prompt = f"""
You are a **PostgreSQL-and-Grafana inspector**.
Return **plain text** only – no Markdown.

STEP 1 – Dump SQL  
  For every panel in the JSON output  
    [panel {{id}}] {{title}}  
    SQL: {{rawSql}}

STEP 2 – Validation   (STOP at the first failure)
───────────────────────────────────────────────
USER REQUEST
{prompt}

DB SCHEMA
{chr(10).join(schema)}

DASHBOARD JSON
{dashboard_json}
───────────────────────────────────────────────
VALIDATION CHECK-LIST

1 Panel **type ⇄ format** mapping *(both must be present)*  
   • barchart   → format = table  
   • piechart   → format = table  
   • table      → format = table  
   • stat       → format = table  
   • timeseries → format = time_series **AND** SQL must alias the timestamp as `time`

2 Alias sanity  
   • Build *DeclaredAlias* = every alias/table in FROM/JOIN  
   • Build *UsedAlias*     = every prefix before "." in SELECT/WHERE/GROUP/ORDER  
   • If UsedAlias ∉ DeclaredAlias → "Alias <x> not declared."

3 Allowed tables: **sales, person, products** (plural) only.

4 No semicolon as the very last SQL character.

5 Bar/Pie numeric group fields must end with `::text`.

6 Every alias used in SELECT/WHERE/GROUP/ORDER
   must be declared in a FROM/JOIN clause.

7 Singular table names (sale, product, …) are invalid.

OUTPUT  
• List every failure, one per line  
• If **all** rules pass → **OK**
───────────────────────────────────────────────
""".strip()

    try:
        # Try OpenAI-compatible API first
        response = requests.post(
            "http://ollama:11434/v1/chat/completions",
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": check_prompt}],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        
        if response.status_code == 200:
            result = response.json()
            result_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
       
        
        print("\n── LLM output ──────────────────────────────\n"
            + result_text +
            "\n────────────────────────────────────────────")

        # grab every line that begins with "ERROR"
        error_lines = [
            line.strip()
            for line in result_text.splitlines()
            if line.strip().lower().startswith("error")
        ]
        
        try:
            parsed_json = json.loads(dashboard_json)
            panels = parsed_json["dashboard"]["panels"]

            sql_to_types = defaultdict(list)
            for p in panels:
                raw_sql = p['targets'][0]['rawSql'].strip().lower()
                panel_type = p['type']
                sql_to_types[raw_sql].append(panel_type)

            for raw_sql, types in sql_to_types.items():
                # Extract common keywords from SQL
                keywords = re.findall(
                    r'\b(?:sum|avg|average|total|count|customer|product|sales|value|date)\b',
                    raw_sql
                )
                if not keywords:
                    continue

                keyword_set = set(keywords)
                if not keyword_set:
                    continue

                prompt_lower = prompt.lower()

                # Count overlapping conceptual matches in prompt
                match_count = 0
                for line in re.split(r'[.,;\n]', prompt_lower):
                    line_words = set(re.findall(r'\b\w+\b', line))
                    overlap = keyword_set & line_words
                    if len(overlap) >= max(2, len(keyword_set) // 2):  # 50%+ overlap
                        match_count += 1

                unique_types = set(types)

                if match_count > len(unique_types):
                    return False, (
                        f"User requested the concept '{' '.join(sorted(keyword_set))}' approximately {match_count} times, "
                        f"but only {len(unique_types)} unique panel types were generated for it."
                    )

        except Exception as e:
            print(f"⚠️ Additional logic validation skipped due to parsing issue: {e}")

        
        if not error_lines:                # ⇐ the model answered "OK"
            return True, ""

        # otherwise return the whole list (joined with new-lines)
        return False, "\n".join(error_lines)

    except Exception as e:
        return False, f"Ollama error: {e}"


@app.post("/ask")
def ask_ollama(prompt_request: PromptRequest):
    wall_start = time.perf_counter()
    fn_times: Dict[str, int] = {}
    iterations_used = 0
    success = False
    err_msg = ""
    prompt = prompt_request.prompt

    try:
        ts = now_timestamp()
        schema_parts: List[str] = [
            "Table: sales\nColumns: id, person_id, product_id, sale_date, value\n"
            "Example Row: {\"id\":1,\"person_id\":1,...}"
        ]

        # 1)  Table extraction
        tables = time_call(fn_times, "extract_table_names", extract_table_name_with_llm, prompt)
        for tbl in [t.strip() for t in tables.split(",") if t.strip()]:
            schema_parts.append(
                f"Table: {tbl}\n"
                f"{time_call(fn_times, f'get_schema_{tbl}', get_table_schema, tbl)}\n"
                f"Example Row: {json.dumps(time_call(fn_times, f'get_example_{tbl}', get_table_example, tbl))}"
            )

        # 2)  Generate dashboard JSON
        s_prompt = time_call(fn_times, "create_structured_prompt", create_structured_prompt, prompt, schema_parts, ts)
        dashboard = time_call(fn_times, "ollama_generate", call_ollama_structured, s_prompt)

        # 3)  Validate / correct loop
        max_iter = 1
        for i in range(1, max_iter + 1):
            iterations_used = i
            ok_struct, err_struct, validated = time_call(fn_times, "pydantic_validate", validate_with_pydantic, dashboard)
            
            if not ok_struct:
                corr_prompt = time_call(fn_times, "create_corr_struct", create_correction_prompt, prompt, schema_parts, json.dumps(dashboard), err_struct, ts)
                dashboard = time_call(fn_times, "ollama_fix_struct", call_ollama_structured, corr_prompt)
                continue
            
            validated_dict = validated.dict(by_alias=True, exclude_none=True)
            validated_slim = validated.dict(
                by_alias=True,
                include={
                    "dashboard": {               # ─┐ field "dashboard"
                        "title": True,           #   ├─ keep the whole title field
                        "panels": {              #   └─ keep only some sub-fields of every panel
                            "__all__": {"id", "type", "targets"}
                        },
                    }
                },
            )
            # targets → keep only refId/format/rawSql
            for p in validated_slim["dashboard"]["panels"]:
                p["targets"] = [
                    {k: t[k] for k in ("refId", "format", "rawSql")} for t in p["targets"]
                ]
            print(f"validated slim: {validated_slim}")
            ok_logic, err_logic = time_call(fn_times, "llm_logic_validate", ask_ollama_if_valid, prompt, schema_parts, json.dumps(validated_slim), ts)
            print(f"validated slim: {validated_slim}, ok_logic: {ok_logic}, err_logic: {err_logic}")
            
            if not ok_logic:
                corr_prompt = time_call(fn_times, "create_corr_logic", create_correction_prompt, prompt, schema_parts, json.dumps(dashboard), err_logic, ts)
                dashboard = time_call(fn_times, "ollama_fix_logic", call_ollama_structured, corr_prompt)
                continue

            success = True
            break

        if not success:
            raise RuntimeError("Max validation iterations reached; dashboard still invalid")

        # 4)  Push to Grafana - Enhanced error handling
        print(f"Pushing dashboard to Grafana...")
        print(f"Dashboard data: {json.dumps(validated.dict(by_alias=True), indent=2)}")
        
        try:
            grafana_resp = time_call(
                fn_times, 
                "post_grafana", 
                requests.post, 
                "http://grafana-mcp:8000/create_dashboard", 
                headers={"Content-Type": "application/json"}, 
                json=validated.dict(by_alias=True),
                timeout=30  # Add timeout for Grafana request
            )
            
            print(f"Grafana response status: {grafana_resp.status_code}")
            print(f"Grafana response headers: {grafana_resp.headers}")
            print(f"Grafana response text: {grafana_resp.text}")
            
            # Check if response is successful
            if grafana_resp.status_code != 200:
                raise Exception(f"Grafana API error: {grafana_resp.status_code} - {grafana_resp.text}")
            
            # Check if response has content
            if not grafana_resp.text.strip():
                raise Exception("Empty response from Grafana API")
            
            # Try to parse JSON
            try:
                grafana_json = grafana_resp.json()
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid JSON from Grafana API: {e}. Response: {grafana_resp.text}")
            
        except requests.exceptions.Timeout:
            raise Exception("Timeout connecting to Grafana service")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error to Grafana service - is it running?")
        except Exception as e:
            raise Exception(f"Grafana API error: {str(e)}")

        return {
            "status": "success",
            "iterations": iterations_used,
            "timestamp": ts,
            "function_times_ms": fn_times,
            "grafana_result": grafana_json,
            "validated": validated.dict(by_alias=True, exclude_none=True),
        }

    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        print(f"❌ ask_ollama failed: {err_msg}")
        return {"status": "error", "error": err_msg, "function_times_ms": fn_times}

    finally:
        record_metric(
            prompt=prompt,
            start_ts=wall_start,
            iterations=iterations_used,
            success=success,
            error_msg=err_msg,
            function_times=fn_times,
        )