from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
import requests
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import time
# Import your models (use the fixed models from above)
from src.models import DashboardSpec, Panel, Target, GridPos, PanelOptions, ReduceOptions
from rag.context_retriever import get_table_schema, get_table_example 
from src.metrics import record_metric, time_call
from collections import Counter, defaultdict
import re
import os
import json

app = FastAPI()

OLLAMA_TIMEOUT = 300  # 5 minutes timeout for Ollama requests         
MCP_URL = os.getenv("MCP_URL", "http://mcp_server:8000")

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
                "model": "llama3.1",
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
                    "model": "llama3.1",
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
                    "datasource": {"type": "postgres", "uid": "aesl3yvmthfy8e"},
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


def call_ollama_structured(prompt: str) -> dict:
    """Call Ollama with a structured prompt and parse JSON response."""
    try:
        # Try OpenAI-compatible API first
        response = requests.post(
            "http://ollama:11434/v1/chat/completions",
            json={
                "model": "llama3.1",
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

def build_full_dashboard_with_mcp(prompt: str, schema_parts: List[str], timestamp: str) -> dict:
    """
    Full pipeline to support ANY database + Grafana dashboard generation:
    1. Uses LLM to extract panel intent from prompt.
    2. Uses `generate_panel_sql_with_tools()` for precise SQL via tool-calling and MCP.
    3. Constructs final Grafana dashboard JSON.
    """

    # STEP 1 — Use LLM to break prompt into panel intents (titles + types)
    intent_extraction_prompt = f"""
    You are a Grafana dashboard planner.
    Available schema: {schema_parts}
    User request: "{prompt}"

    Return _only_ a JSON array of objects with exactly these two keys:
    [
    {{ "type": "barchart", "title": "..." }},
    {{ "type": "timeseries", "title": "..." }},
    …
    ]
    Valid panel types: barchart, piechart, table, timeseries, stat.
    (No additional fields, no markdown fences, no explanatory text.)
    """

    try:
        r = requests.post(
            "http://ollama:11434/v1/chat/completions",
            json={
                "model": "llama3.1",
                "messages": [{"role": "user", "content": intent_extraction_prompt}],
                "temperature": 0.1, "top_p": 0.9, "stream": False
            },
            timeout=OLLAMA_TIMEOUT
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        print("📤 Raw panel intent response:\n", raw)

        # extract the first JSON array
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON array found in response: {raw}")
        arr = json.loads(m.group(0))

        # normalize to only type/title
        panel_intents = []
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            t = obj.get("type")
            ti = obj.get("title")
            if not t or not ti:
                raise ValueError(f"Panel entry missing type or title: {obj}")
            panel_intents.append({"type": t, "title": ti})

        if not panel_intents:
            raise ValueError("LLM returned empty or invalid panel list")

    except Exception as e:
        raise RuntimeError(f"Failed to extract panel intents: {e}")

    # STEP 2 — Generate SQL for each panel using tools + MCP (unchanged) …
    panel_specs = []
    for i, panel in enumerate(panel_intents):
        # …
        result = generate_panel_sql_with_tools(
            panel_id=i,
            panel_type=panel["type"],
            panel_title=panel["title"],
            schema_parts=[p for p in schema_parts if 'Example Row' not in p and '...' not in p],
            timestamp=timestamp,
            model="llama3.1",
            max_iterations=3
        )
        panel_specs.append({
            "id": i,
            "type": panel["type"],
            "title": panel["title"],
            "rawSql": result["rawSql"],
            "format": result.get("format", "table")
        })

    # STEP 3 — Build final dashboard JSON with grid positioning (unchanged) …
    panels = []
    total = len(panel_specs)
    for i, spec in enumerate(panel_specs):
        if i == total - 1 and total % 2 == 1:
            x, w = 0, 24
        else:
            x, w = (0, 12) if i % 2 == 0 else (12, 12)
        y = (i // 2) * 8
        panels.append({
            "id": spec["id"],
            "type": spec["type"],
            "title": f"{spec['title']} - {timestamp}",
            "datasource": {"type": "postgres", "uid": "aesl3yvmthfy8e"},
            "targets": [{
                "refId": "A",
                "format": spec["format"],
                "rawSql": spec["rawSql"]
            }],
            "gridPos": {"x": x, "y": y, "w": w, "h": 8},
            "options": {
                "tooltip": {"mode": "single"},
                "legend": {"displayMode": "list", "placement": "bottom"},
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": True}
            }
        })

    return {
        "dashboard": {
            "title": f"Dashboard for: {prompt} - {timestamp}",
            "schemaVersion": 36,
            "version": 1,
            "refresh": "5s",
            "time": {"from": "now-5y", "to": "now"},
            "timepicker": {
                "refresh_intervals": ["5s","10s","30s","1m","5m","15m","30m","1h","2h","1d"],
                "time_options":    ["5m","15m","1h","6h","12h","24h","2d","7d","30d","90d","6M","1y","5y"]
            },
            "panels": panels
        },
        "overwrite": False
    }



def call_ollama_with_tools(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    endpoint: str = "http://ollama:11434/v1/chat/completions",
    timeout: Tuple[int, Optional[int]] = OLLAMA_TIMEOUT,
    max_iterations: int = 3,
) -> str:
    """Fixed tool-enabled conversation handler for Ollama."""
    import requests, json
    history = messages.copy()

    # IMPORTANT: Format tools correctly for Ollama
    # Ollama expects the tool format without the nested "function" wrapper
    tool_defs = []
    for t in tools:
        tool_defs.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"]
            }
        })

    for iteration in range(max_iterations):
        print(f"🔄 Tool iteration {iteration + 1}/{max_iterations}")
        
        payload = {
            "model": model,
            "messages": history,
            "tools": tool_defs,
            "stream": False,
            "temperature": 0.1,
            "tool_choice": "auto"  # Explicitly enable tool calling
        }
        
        print(f"📤 Sending payload: {json.dumps(payload, indent=2)}")

        try:
            resp = requests.post(endpoint, json=payload, timeout=timeout)
            resp.raise_for_status()
            response_data = resp.json()
            
            print(f"📥 Full API response: {json.dumps(response_data, indent=2)}")
            
            msg = response_data["choices"][0]["message"]
            
            # Check for tool calls in the response
            if "tool_calls" in msg and msg["tool_calls"]:
                print(f"🔧 Model made {len(msg['tool_calls'])} tool calls")
                
                # Add the assistant's message with tool calls to history
                history.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                    "tool_calls": msg["tool_calls"]
                })
                
                # Execute each tool call
                for call in msg["tool_calls"]:
                    function_name = call["function"]["name"]
                    function_args = call["function"]["arguments"]
                    
                    print(f"🛠️ Executing tool: {function_name}")
                    print(f"📋 Arguments: {function_args}")
                    
                    # Parse arguments (they might be a string or dict)
                    if isinstance(function_args, str):
                        try:
                            args = json.loads(function_args)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = function_args or {}
                    
                    # Execute the actual tool
                    try:
                        if function_name == "list_tables":
                            result = requests.get(f"{MCP_URL}/tables", timeout=10).json()
                        elif function_name == "get_schema":
                            table_name = args.get("table_name")
                            if not table_name:
                                raise ValueError("table_name is required for get_schema")
                            result = requests.get(f"{MCP_URL}/schema/{table_name}", timeout=10).json()
                            print(f"📊 Schema for {table_name}: {result}")
                        elif function_name == "get_example":
                            table_name = args.get("table_name")
                            if not table_name:
                                raise ValueError("table_name is required for get_example")
                            result = requests.get(f"{MCP_URL}/example/{table_name}", timeout=10).json()
                            print(f"📋 Examples for {table_name}: {result}")
                        elif function_name == "run_sql":
                            sql_string = args.get("sql_string")
                            if not sql_string:
                                raise ValueError("sql_string is required for run_sql")
                            result = requests.post(
                                f"{MCP_URL}/execute", 
                                json={
                                    "sql": [sql_string], 
                                    "chart_type": [args.get("chart_type", "table")]
                                },
                                timeout=30
                            ).json()
                            print(f"🔍 SQL execution result: {result}")
                        else:
                            result = {"error": f"Unknown tool: {function_name}"}
                            
                    except requests.exceptions.RequestException as e:
                        result = {"error": f"MCP service error for {function_name}: {str(e)}"}
                        print(f"❌ MCP error: {result}")
                    except Exception as e:
                        result = {"error": f"Tool execution error for {function_name}: {str(e)}"}
                        print(f"❌ Tool error: {result}")

                    # Add tool result to history
                    # IMPORTANT: Ollama expects tool results in this specific format
                    history.append({
                        "role": "tool",
                        "content": json.dumps(result),
                        "tool_call_id": call.get("id", f"call_{iteration}")
                    })
                    
            else:
                # No tool calls, this should be the final response
                content = msg.get("content", "")
                if not content.strip():
                    raise ValueError(f"LLM returned empty content on iteration {iteration + 1}")
                
                print(f"📤 Final response: {content}")
                return content

        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama API error on iteration {iteration + 1}: {e}")
            raise RuntimeError(f"Ollama API error on iteration {iteration + 1}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error on iteration {iteration + 1}: {e}")
            raise RuntimeError(f"Unexpected error on iteration {iteration + 1}: {e}")

    raise RuntimeError(f"Maximum iterations ({max_iterations}) reached without final response")


def generate_panel_sql_with_tools(
    panel_id: int,
    panel_type: str,
    panel_title: str,
    schema_parts: List[str],
    timestamp: str,
    model: str = "llama3.1",
    max_iterations: int = 5  # Increased for tool calling
) -> Dict[str, Any]:
    """Enhanced SQL generator using proper Ollama tool calling."""
    
    tools = [
        {
            "name": "list_tables", 
            "description": "List all available tables in the database", 
            "parameters": {
                "type": "object", 
                "properties": {}, 
                "required": []
            }
        },
        {
            "name": "get_schema", 
            "description": "Get detailed schema for a specific table including column names and types", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "table_name": {
                        "type": "string", 
                        "description": "Name of the table to get schema for"
                    }
                }, 
                "required": ["table_name"]
            }
        },
        {
            "name": "get_example", 
            "description": "Get example rows from a table to understand the data format", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "table_name": {
                        "type": "string", 
                        "description": "Name of the table to get examples from"
                    }
                }, 
                "required": ["table_name"]
            }
        },
        {
            "name": "run_sql", 
            "description": "Test execute SQL query to validate it works", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "sql_string": {
                        "type": "string", 
                        "description": "SQL query to execute"
                    },
                    "chart_type": {
                        "type": "string", 
                        "default": "table", 
                        "description": "Type of chart this SQL is for"
                    }
                }, 
                "required": ["sql_string"]
            }
        }
    ]

    # More directive system prompt that forces tool usage
    system_message = """You are a SQL expert assistant. You MUST use the provided tools to explore the database and generate accurate SQL.

    MANDATORY PROCESS:
    1. FIRST: Call list_tables to see what tables exist
    2. SECOND: Call get_schema on the most relevant table (likely 'sales' for sales data)
    3. THIRD: Call get_example on that table to see sample data
    4. FOURTH: Create SQL using the EXACT column names you discovered
    5. FIFTH: Call run_sql to test your query
    6. FINALLY: Return ONLY a JSON object like: {"rawSql": "your SQL here", "format": "table"}

    CRITICAL RULES:
    - You MUST call the tools, don't just describe what you would do
    - Use ONLY the actual table and column names from the tools
    - Never use placeholder names like "sales_data", "table_name", "column1" etc.
    - Your final response must be ONLY the JSON object, nothing else"""

    user_message = f"""Generate SQL for a Grafana {panel_type} panel: "{panel_title}"

    Panel requirements:
    - Type: {panel_type}
    - Title: {panel_title}
    - Show daily sales totals grouped by date

    You MUST use the tools to discover the real database schema first. Start by calling list_tables now."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    try:
        reply = call_ollama_with_tools(model, messages, tools, max_iterations=max_iterations)

        if not reply.strip():
            raise ValueError("LLM returned an empty response")
        
        print("📥 Raw LLM reply:", reply)
        
        # Extract JSON from response
        result = extract_json_from_response(reply)
        
        # Validate the result
        if "rawSql" not in result:
            raise ValueError("Response missing required 'rawSql' field")
        
        raw_sql = result["rawSql"].strip()
        
        # Check for placeholder/hardcoded values that indicate failure
        forbidden_patterns = [
            "table_name", "column_name", "your_table", "placeholder", 
            "example_table", "column1", "column2", "timestamp_column",
            "sales_data", "orders"  # Add these specific wrong table names
        ]
        
        for pattern in forbidden_patterns:
            if pattern.lower() in raw_sql.lower():
                raise ValueError(f"Generated SQL contains placeholder '{pattern}' - must use actual table/column names")
        
        # Ensure SQL is not just an error message
        if raw_sql.upper().startswith("SELECT 'ERROR") or "error" in raw_sql.lower():
            raise ValueError(f"Generated SQL is an error message: {raw_sql}")
        
        # Set default format if missing
        if "format" not in result:
            result["format"] = "table"
            
        print("✅ Successfully generated SQL with real tables:", result)
        return result

    except Exception as e:
        error_msg = f"SQL generation failed for panel '{panel_title}': {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Enhanced JSON extraction with better error reporting."""
    import json
    import re
    
    if not response.strip():
        raise ValueError("Empty response received")
    
    # First, try to find JSON blocks marked with ```json
    json_code_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL | re.IGNORECASE)
    for block in json_code_blocks:
        try:
            parsed = json.loads(block.strip())
            if "rawSql" in parsed:
                return parsed
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON code block: {e}")
            continue
    
    # Try to find any JSON object with rawSql
    json_matches = re.findall(r'\{[^{}]*?"rawSql"[^{}]*?\}', response, re.DOTALL)
    for match in json_matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON match: {e}")
            continue
    
    # Look for any complete JSON object
    json_objects = re.findall(r'\{(?:[^{}]|{[^{}]*})*\}', response, re.DOTALL)
    for obj in json_objects:
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict) and ("rawSql" in parsed or "sql_string" in parsed):
                # Convert sql_string to rawSql if needed
                if "sql_string" in parsed and "rawSql" not in parsed:
                    parsed["rawSql"] = parsed["sql_string"]
                return parsed
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON object: {e}")
            continue
    
    # Last resort: try to parse the entire response
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as e:
        print(f"Failed to parse entire response as JSON: {e}")
    
    raise ValueError(f"No valid JSON found in response. Response was: {response[:500]}...")
    
    
def validate_with_pydantic(dashboard_data: dict) -> tuple[bool, str, DashboardSpec]:
    """Validate dashboard data with Pydantic and return success status, error message, and object"""
    try:
        # Validate the dashboard data against the Pydantic model
        spec = DashboardSpec(**dashboard_data)
        return True, "", spec
    except ValidationError as e:
        return False, str(e), None




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
        # s_prompt = time_call(fn_times, "create_structured_prompt", create_structured_prompt, prompt, schema_parts, ts)
        # dashboard = time_call(fn_times, "ollama_generate", call_ollama_structured, s_prompt)
        ## mcp_ tools calling
        # dashboard = time_call(fn_times, "build_full_dashboard_with_mcp", build_full_dashboard_with_mcp, prompt, schema_parts, ts)
        ## mcp tools calling with LLM
        payload = {
            "prompt": prompt,
            "schema_description": schema_parts,
            "timestamp": ts
        }
        
        mcp_response = time_call(fn_times, "call_mcp_ask", requests.post, f"{MCP_URL}/ask", json=payload,timeout=OLLAMA_TIMEOUT)
        if mcp_response.status_code != 200:
            raise Exception(f"MCP error: {mcp_response.text}")
        dashboard = mcp_response.json()
        print(f"📥 MCP response: {json.dumps(dashboard, indent=2)}")
        # 3)  Validate / correct loop
        max_iter = 1
        for i in range(1, max_iter + 1):
            iterations_used = i
            ok_struct, err_struct, validated = time_call(fn_times, "pydantic_validate", validate_with_pydantic, dashboard)
            
            if not ok_struct:
                corr_prompt = time_call(fn_times, "create_corr_struct", create_correction_prompt, prompt, schema_parts, json.dumps(dashboard), err_struct, ts)
                dashboard = time_call(fn_times, "ollama_fix_struct", call_ollama_structured, corr_prompt)
                continue
            
            # validated_dict = validated.dict(by_alias=True, exclude_none=True)
            # validated_slim = validated.dict(
            #     by_alias=True,
            #     include={
            #         "dashboard": {               # ─┐ field "dashboard"
            #             "title": True,           #   ├─ keep the whole title field
            #             "panels": {              #   └─ keep only some sub-fields of every panel
            #                 "__all__": {"id", "type", "targets"}
            #             },
            #         }
            #     },
            # )
            # # targets → keep only refId/format/rawSql
            # for p in validated_slim["dashboard"]["panels"]:
            #     p["targets"] = [
            #         {k: t[k] for k in ("refId", "format", "rawSql")} for t in p["targets"]
            #     ]
            # print(f"validated slim: {validated_slim}")
            # ok_logic, err_logic = time_call(fn_times, "llm_logic_validate", ask_ollama_if_valid, prompt, schema_parts, json.dumps(validated_slim), ts)
            # print(f"validated slim: {validated_slim}, ok_logic: {ok_logic}, err_logic: {err_logic}")
            
            # if not ok_logic:
            #     corr_prompt = time_call(fn_times, "create_corr_logic", create_correction_prompt, prompt, schema_parts, json.dumps(dashboard), err_logic, ts)
            #     dashboard = time_call(fn_times, "ollama_fix_logic", call_ollama_structured, corr_prompt)
            #     continue

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