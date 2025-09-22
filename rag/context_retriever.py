import requests
import re, json, requests
from typing import List, Dict
# Base URL of your MCP API (adjust if needed)

MCP_URL = "http://mcp_server:8000"


# 🔍 Fetch the schema (columns and types) of a table via MCP
def get_table_schema(table_name: str) -> str:
    try:
        response = requests.get(f"{MCP_URL}/schema/{table_name}")
        data = response.json()
        if "schema" in data:
            schema = [col if col != "sale_date" else "time (from sale_date)" for col in data["schema"]]
            return "Columns: " + ", ".join(schema)
        else:
            return data.get("error", "Unknown error")
    except Exception as e:
        return f"Error fetching schema: {e}"


# 🧪 Fetch one example row from a table via MCP
def get_table_example(table_name: str) -> str:
    try:
        response = requests.get(f"{MCP_URL}/example/{table_name}")
        data = response.json()
        example = data.get("example", {})
        if "sale_date" in example:
            example["time"] = example.pop("sale_date")
        return example
    except Exception as e:
        return f"Error fetching example: {e}"


def get_grafana_json_template() -> str:
    return """
    Example Grafana Dashboard JSON structure:

    {
    "dashboard": {
        "title": "<Your Dashboard Title>",
        "schemaVersion": 36,
        "version": 1,
        "refresh": "5s",
        "time": { "from": "now-5y", "to": "now" },
        "timepicker": {
         "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"],
         "time_options": ["5m", "15m", "1h", "6h", "12h", "24h", "2d", "7d", "30d", "90d", "6M", "1y", "5y"]
        },
        "panels": [
        {
            "id": 1,
            "type": "barchart", 
            "title": "<Panel Title>",
            "datasource": {
                "type": "postgres",
                "uid": "feyd4obe5zb40b"
            },
            "targets": [
            {
                "refId": "A",
                "format": "time_series",
                "rawSql": "<Generated SQL query here, as a valid JSON string. Do not use triple quotes. Escape newlines if needed.>"
            }
            ],
            "gridPos": {
                "x": <dynamic_x>, "y": <dynamic_y>, "w": <dynamic_w>, "h": 8
            },

            "options": {
                "tooltip": { "mode": "single" },
                "legend": { "displayMode": "list", "placement": "bottom" }
            }
        }
        ]
    },
    "overwrite": false
    }

    Instructions:
    - Only return JSON with this structure.
    - The "rawSql" should match the user request based on provided table schema.
    - Use valid panel types: "barchart", "timeseries", "table", "piechart", etc.
    - The "options" field is required and must include "tooltip" and "legend".
    - If the panel type is "piechart", you must also include "reduceOptions" inside "options" to ensure all values are displayed individually. Use:
        "reduceOptions": {
            "calcs": ["lastNotNull"],
            "fields": "",
            "values": true
        }
    - Dynamically assign `gridPos`: Stack large panels vertically. Place smaller ones side-by-side (e.g., w:12). x should be 0 or 12, y should increment by 8 for each new row.
    - Do not include unsupported fields like "query.load", "p", or others not shown above.
    - Use the correct "uid" for the PostgreSQL datasource.
    """


# bring in or define clean_control_chars
def clean_control_chars(s: str) -> str:
    """
    Strip out any non-printable control characters except newline, carriage return, and tab.
    """
    return "".join(ch for ch in s if ch in "\r\n\t" or 32 <= ord(ch) <= 126)

def get_panel_intents(
    prompt: str,
    ollama_url: str = "http://ollama:11434/v1/chat/completions",
    model: str = "llama3.1",
    timeout: int = 300
) -> List[Dict[str, str]]:
    """
    Split a compound user request into individual Grafana panel intents.
    
    Args:
      prompt:        The user's multi-part dashboard request.
      ollama_url:    Ollama chat completions endpoint.
      model:         Ollama model name.
      timeout:       Request timeout in seconds.
    
    Returns:
      A list of {"type": <chart_type>, "title": <panel_title>} objects.
    """
    # 1) Few-shot / mapping rules reminder
    mapping_rules = (
        "1. NEVER use “bar” – always “barchart”\n"
        "2. NEVER use “pie” – always “piechart”\n"
        "3. Only types: barchart, piechart, table, timeseries, stat\n"
        "4. bar/bar chart/bars → barchart; pie/pie chart → piechart; "
        "table/list/stat/gauge → table; time series/over time → timeseries; "
        "total/sum/count/statistic → stat\n"
    )

    # 2) Construct messages
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Grafana dashboard planner.  When asked to split a request, "
                "respond *only* with a JSON array of objects with keys `type` and `title`, "
                "with no extra text."
            )
        },
        {
            "role": "user",
            "content": (
                f"Request: {prompt}\n\n"
                f"{mapping_rules}"
            )
        }
    ]
    print("[get_panel_intents] Sending to Ollama:", messages)
    # 3) Call the Ollama endpoint
    resp = requests.post(
        ollama_url,
        json={
            "model":       model,
            "messages":    messages,
            "temperature": 0.1,
            "stream":      False
        },
        timeout=timeout
    )
    resp.raise_for_status()
    print("[get_panel_intents] Ollama raw response:", resp.text)
    raw_content = resp.json()["choices"][0]["message"]["content"].strip()

    # 4) Extract the first [...] JSON array (non-greedy)
    m = re.search(r"\[.*?\]", raw_content, re.DOTALL)
    if not m:
        raise ValueError(f"Failed to parse JSON array from LLM response: {raw_content!r}")

    raw_array = m.group(0)
    # 5) Clean control chars
    cleaned = clean_control_chars(raw_array)

    # 6) Parse JSON
    try:
        panels = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {cleaned!r}") from e

    # 7) Validate shape
    if not isinstance(panels, list):
        raise ValueError(f"Expected a list, got {type(panels).__name__}: {panels!r}")
    for i, obj in enumerate(panels):
        if not isinstance(obj, dict) or "type" not in obj or "title" not in obj:
            raise ValueError(f"Malformed panel intent at index {i}: {obj!r}")
    print("[get_panel_intents] Parsed intents:", panels)
    return panels
