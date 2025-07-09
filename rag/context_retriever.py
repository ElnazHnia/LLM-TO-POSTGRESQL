import requests

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
                "uid": "cer8a03ztsu0wf"
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


