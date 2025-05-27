import requests

# Base URL of your MCP API (adjust if needed)

MCP_URL = "http://mcp_server:8000"


# 🔍 Fetch the schema (columns and types) of a table via MCP
def get_table_schema(table_name: str) -> str:
    try:
        response = requests.get(f"{MCP_URL}/schema/{table_name}")
        data = response.json()
        return "Columns: " + ", ".join(data["schema"]) if "schema" in data else data.get("error", "Unknown error")
    except Exception as e:
        return f"Error fetching schema: {e}"

# 🧪 Fetch one example row from a table via MCP
def get_table_example(table_name: str) -> str:
    try:
        response = requests.get(f"{MCP_URL}/example/{table_name}")
        data = response.json()
        return data.get("example", {})
    except Exception as e:
        return f"Error fetching example: {e}"
