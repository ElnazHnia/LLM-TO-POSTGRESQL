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

