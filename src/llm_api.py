from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import requests
import json
from datetime import datetime
from typing import List

# Import your models (use the fixed models from above)
from src.models import DashboardSpec, Panel, Target, GridPos, PanelOptions, ReduceOptions
from rag.context_retriever import get_table_schema, get_table_example 

app = FastAPI()

OLLAMA_TIMEOUT = 300  # 5 minutes timeout for Ollama requests         

class PromptRequest(BaseModel):
    prompt: str

# Create a global timestamp when the request starts
def get_current_timestamp() -> str:
    """Get current timestamp in consistent format"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def extract_table_name_with_llm(prompt: str) -> str:
    res = requests.post("http://ollama:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"You are a SQL assistant.\nGiven this question, extract the table name(s) mentioned.\n\nQuestion: {prompt}\n\nReturn only the table names, comma-separated if more than one. The sales table contains person_id, product_id, sale_date is a DATE, and value. The person table stores id and name of customers. The products table has id and name.  " 
    },
    timeout=OLLAMA_TIMEOUT
    )
   
    raw_response = "".join(
        json.loads(line).get("response", "")
        for line in res.text.strip().split("\n")
        if line.strip().startswith("{")
    ).strip().lower()
    print("Raw response from table extractor:", raw_response)
    #  Extract just the table names
    table_line = raw_response.split("\n")[-1]  # get the last line (which should contain only the names)
    clean_tables = [name.strip() for name in table_line.split(",") if name.strip().isidentifier()]

    return ",".join(clean_tables)


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
                    "datasource": {"type": "postgres", "uid": "aeo8prusu1i4gc"},
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
    """Call Ollama with structured output request"""
    try:
        response = requests.post(            # ← 8 spaces (aligned with the rest)
            "http://ollama:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,      # Lower temperature for more consistent JSON
                    "top_p": 0.9
                }
            },
            timeout=OLLAMA_TIMEOUT           # 90 s (or whatever you set)
        )

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")

        result = response.json()
        llm_output = result.get("response", "")

        # Try to parse as JSON
        try:
            return json.loads(llm_output)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw LLM output: {llm_output}")
            raise Exception(f"Invalid JSON from LLM: {str(e)}")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        raise


def validate_with_pydantic(dashboard_data: dict) -> tuple[bool, str, DashboardSpec]:
    """Validate dashboard data with Pydantic and return success status, error message, and object"""
    try:
        dashboard = DashboardSpec(**dashboard_data)
        return True, "", dashboard
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
1 Title must equal “[Description] – {timestamp}”

2 Panel **type ⇄ format** mapping *(both must be present)*  
   • barchart   → format = table  
   • piechart   → format = table  
   • table      → format = table  
   • stat       → format = table  
   • timeseries → format = time_series **AND** SQL must alias the timestamp as `time`

3 Alias sanity  
   • Build *DeclaredAlias* = every alias/table in FROM/JOIN  
   • Build *UsedAlias*     = every prefix before “.” in SELECT/WHERE/GROUP/ORDER  
   • If UsedAlias ∉ DeclaredAlias → “Alias <x> not declared.”

4 Allowed tables: **sales, person, products** (plural) only.

5 No semicolon as the very last SQL character.

6 Bar/Pie numeric group fields must end with `::text`.

7 Every alias used in SELECT/WHERE/GROUP/ORDER
   must be declared in a FROM/JOIN clause.

8 Singular table names (sale, product, …) are invalid.

OUTPUT  
• List every failure, one per line  
• If **all** rules pass → **OK**
───────────────────────────────────────────────
""".strip()

    try:
        rsp = requests.post(
            "http://ollama:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": check_prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        if rsp.status_code != 200:
            raise RuntimeError(f"Ollama API error: {rsp.status_code}")

        result_text = rsp.json().get("response", "").strip()
        print("\n── LLM output ──────────────────────────────\n"
            + result_text +
            "\n────────────────────────────────────────────")

        # grab every line that begins with “ERROR”
        error_lines = [
            line.strip()
            for line in result_text.splitlines()
            if line.strip().lower().startswith("error")
        ]

        if not error_lines:                # ⇐ the model answered “OK”
            return True, ""

        # otherwise return the whole list (joined with new-lines)
        return False, "\n".join(error_lines)

    except Exception as e:
        return False, f"Ollama error: {e}"





@app.post("/ask")
def ask_ollama(prompt_request: PromptRequest):
    try:
        prompt = prompt_request.prompt
        print(f"Received prompt: {prompt}")

        # Create a single timestamp for this entire request
        current_timestamp = get_current_timestamp()
        print(f"Generated timestamp: {current_timestamp}")

        # Database schema (example — adjust as needed)
        schema_parts = [
            "Table: sales\nColumns: id (integer), person_id (integer), product_id (integer), sale_date (date), value (numeric), payment_id (integer), order_id (integer)\nExample Row: {\"id\": 1, \"person_id\": 1, \"product_id\": 1, \"value\": 999.99, \"payment_id\": 1, \"order_id\": 1, \"sale_date\": \"2024-03-01\"}"
        ]
        
        # Extract table names from the prompt using LLM
        table_names_str = extract_table_name_with_llm(prompt)
        table_names = [name.strip() for name in table_names_str.split(",")]
        print(f"Extracted table names: {table_names}")
        
        for table in table_names:
            schema = get_table_schema(table)
            example = get_table_example(table)
            schema_parts.append(f"Table: {table}\n{schema}\nExample Row: {json.dumps(example)}")
        print(f"Extracted schema parts: {schema_parts}")
        
        # STEP 1: Generate initial JSON
        print("🧠 Step 1: Generating initial JSON from LLM...")
        structured_prompt = create_structured_prompt(prompt, schema_parts, current_timestamp)
        dashboard_data = call_ollama_structured(structured_prompt)
        print(f"🧠 LLM generated initial JSON")

        # STEP 2-N: Iterative validation and correction loop
        max_iterations = 1
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration}: Validating JSON...")
            
            # Check structure with Pydantic
            # Pydantic is an automatic checklist we run after the model runs
            is_structure_valid, structure_error, validated_dashboard = validate_with_pydantic(dashboard_data)
            
            if not is_structure_valid:
                print(f"❌ Structure validation failed: {structure_error}")
                
                # Fix structure issues
                correction_prompt = create_correction_prompt(
                    prompt, schema_parts, 
                    json.dumps(dashboard_data, indent=2), 
                    f"Pydantic structure error: {structure_error}",
                    current_timestamp
                )
                
                print("🔧 Asking LLM to fix structure issues...")
                dashboard_data = call_ollama_structured(correction_prompt)
                continue  # Go to next iteration
            
            print("✅ Structure validation passed")
            
            # Check logic with Ollama
            print("🧠 Checking logic validation with LLM...")
            original_json_str = json.dumps(dashboard_data, indent=2)
            # original_json_str = """
            # {
            # "dashboard": {
            #     "title": "Bad Dashboard - 2025-07-03 10:00:00",
            #     "schemaVersion": 36,
            #     "version": 1,
            #     "refresh": "5s",
            #     "time": {"from":"now-1d","to":"now"},
            #     "timepicker": {"refresh_intervals":["5s"],"time_options":["5m"]},
            #     "panels": [
            #     {
            #         "id": 0,
            #         "type": "barchart",
            #         "title": "Bad Alias Example",
            #         "datasource": {"type":"postgres","uid":"aeo8prusu1i4gc"},
            #         "targets": [
            #         {
            #             "refId": "A",
            #             "format": "table",
            #             //  ← ERROR 1:   wrong alias “day”, never declared in FROM
            #             "rawSql": "SELECT DATE_TRUNC('day', sale_date) AS day, SUM(value) AS value FROM sales GROUP BY day ORDER BY day"
            #         }
            #         ],
            #         "gridPos": {"x":0,"y":0,"w":12,"h":8}
            #     }
            #     ]
            # },
            # "overwrite": false
            # }
            # """ 
            is_logic_valid, logic_error = ask_ollama_if_valid(prompt, schema_parts, original_json_str, current_timestamp)
            # OPTIONAL: get all SQLs
                        
            print(f"🧠 LLM logic validation result: {is_logic_valid}, error: {logic_error}")
            
            if not is_logic_valid:
                print(f"❌ Logic validation failed: {logic_error}")
                
                # Fix logic issues
                correction_prompt = create_correction_prompt(
                    prompt, schema_parts, 
                    original_json_str, 
                    f"Logic validation error: {logic_error}",
                    current_timestamp
                )
                
                print(f"🔧 Asking LLM to fix logic issues, correction_prompt: {correction_prompt}...")
                dashboard_data = call_ollama_structured(correction_prompt)
                print(f"🧠 after correction :{dashboard_data}")
                is_logic_valid, logic_error = ask_ollama_if_valid(correction_prompt, schema_parts, dashboard_data, current_timestamp)
                print(f"🧠 after correction  is_logic_valid :{is_logic_valid}")
                continue  # Go to next iteration
            
            print("✅ Logic validation passed")
            print(f"🎉 JSON is valid after {iteration} iteration(s)!")
            break
        
        # else:
        #     # Max iterations reached
        #     print(f"❌ Max iterations ({max_iterations}) reached. JSON may still have issues.")
        #     return {
        #         "status": "error", 
        #         "error": "Max validation iterations reached", 
        #         "detail": f"Could not validate JSON after {max_iterations} attempts"
        #     }

        # STEP FINAL: Send to Grafana
        print("📊 Sending validated dashboard to Grafana...")
        dashboard_dict = validated_dashboard.dict(by_alias=True)
        
        try:
            grafana_response = requests.post(
                "http://grafana-mcp:8000/create_dashboard",
                headers={"Content-Type": "application/json"},
                json=dashboard_dict
            )
            grafana_result = grafana_response.json()
            print(f"📊 Grafana result: {grafana_result}")
        except Exception as e:
            print(f"❌ Grafana error: {e}")
            grafana_result = {"error": str(e)}

        return {
            "dashboard": dashboard_dict,
            "grafana_result": grafana_result,
            "status": "success",
            "iterations_used": iteration,
            "timestamp_used": current_timestamp
        }

    except Exception as e:
        print(f"❌ Error in ask_ollama: {e}")
        return {
            "error": str(e),
            "status": "error"
        }