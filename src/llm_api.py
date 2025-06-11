from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import re
import dateparser
from datetime import datetime, timedelta
from typing import List

# from mlflow_logger.logger import log_to_mlflow  # Custom MLflow logging module
from rag.context_retriever import get_table_schema, get_table_example  # MCP helpers for schema & example
'''
# llm_api.py
Handles user prompts, talks to LLM, talks to MCP, assembles results

'''
# Initialize FastAPI app
app = FastAPI()

# Define request model: expects JSON body with a "prompt" string
class PromptRequest(BaseModel):
    prompt: str


# ===  Convert MySQL-style SQL to PostgreSQL-style ===
def force_order_by_time_if_column_exists(sql: str) -> str:
    # Look for time in SELECT columns
    select_match = re.search(r"(?i)select\s+(.*?)\s+from", sql, re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        if re.search(r"\btime\b", select_clause, re.IGNORECASE):
            # Replace entire ORDER BY ... with ORDER BY time
            sql = re.sub(
                r"(?i)ORDER BY\s+[^;]*(?=(\s+LIMIT|\s+OFFSET|;|$))",
                "ORDER BY time",
                sql
            )
    return sql

def ensure_aliases_in_select(sql: str) -> str:
    match = re.search(r"(?is)\bSELECT\s+(.*?)\s+FROM\b", sql)
    if not match:
        return sql

    select_clause = match.group(1)

    def split_columns(clause):
        parts = []
        current = ''
        depth = 0
        for char in clause:
            if char == ',' and depth == 0:
                parts.append(current.strip())
                current = ''
            else:
                current += char
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
        if current:
            parts.append(current.strip())
        return parts

    columns = split_columns(select_clause)
    updated_columns = []

    for col in columns:
        # Skip if already aliased
        if re.search(r"\s+AS\s+\w+$", col, re.IGNORECASE):
            updated_columns.append(col)
            continue

        # Skip if it's a function call (starts with NAME(...) or EXTRACT(...))
        if re.match(r"(?i)^\w+\s*\(", col):
            updated_columns.append(col)
            continue

        # Generate alias by removing table prefix
        alias = col.strip()
        if '.' in alias:
            alias = alias.split('.')[-1]
        updated_columns.append(f"{col} AS {alias}")
    
    new_select_clause = "SELECT " + ', '.join(updated_columns) + " FROM"
    sql = re.sub(r"(?is)\bSELECT\s+.*?\s+FROM\b", new_select_clause, sql)
    print(f"ALIACE Updated SELECT clause: {new_select_clause}")
    return sql



def cast_labels_to_text(sql): 
    # More precise regex to avoid matching FROM inside functions
    # Look for FROM that's at the start of a line or after whitespace (not inside parentheses)
    match = re.search(r"\bSELECT\s+(.*?)\s+^FROM\b", sql, re.IGNORECASE | re.DOTALL | re.MULTILINE) 
    if not match:
        # Fallback: try to match FROM at word boundary with more context
        match = re.search(r"\bSELECT\s+(.*?)\s+FROM\s+", sql, re.IGNORECASE | re.DOTALL)
        if match:
            select_clause = match.group(1).strip()
        else:
            return sql
    else:
        select_clause = match.group(1).strip()
    
     
    # Simplified pattern - removed possessive quantifiers which aren't supported in Python
    pattern = re.compile(r"""(?ix)
        (                                     # Start capture group 1 (expression)
            (?:
                # Handle nested function calls like EXTRACT(YEAR FROM DATE_TRUNC(...))
                \b\w+\s*\(                   # Function name followed by opening paren
                (?:[^()]*|\([^()]*\))*        # Handle nested parentheses (simplified)
                \)                            # Closing paren
            |
                # Handle simple column references like s.id or region_id
                (?:\w+\.\w+|\w+)
            )
        )                                     # End capture group 1
        \s+AS\s+(\w+)                         # AS alias (capture group 2)
    """) 
    
    print(f"Testing pattern on: '{select_clause}'")
    matches = pattern.findall(select_clause)
    print(f"Found matches: {matches}")
    
    def replace_expression(m): 
        expr, alias = m.groups() 
        print(f"CAST expr:{expr}, alias: {alias}") 
        # Skip if it's an aggregate function
        if (re.search(r'\b(SUM|AVG|COUNT|MIN|MAX)\s*\(', expr, re.IGNORECASE)  or alias.lower() == "time"): 
            return f"{expr} AS {alias}" 
        # Skip if already casted 
        if '::text' in expr.lower(): 
            return f"{expr} AS {alias}" 
        # Add ::TEXT cast
        return f"{expr}::TEXT AS {alias}" 
 
    # Apply replacement for all matched expressions 
    updated_select_clause = pattern.sub(replace_expression, select_clause) 
    print(f"CAST Updated SELECT clause: {updated_select_clause}")
    
    # Find the original SELECT clause and replace it entirely
    # Use the same pattern we used to extract it
    original_select_part = match.group(0)  # The entire matched part
    new_select_part = f"SELECT {updated_select_clause} FROM"
    
    
    sql = sql.replace(select_clause, updated_select_clause, 1)
    
    print(f"CAST Updated SQL: {sql}")
    
    return sql


def normalize_sql_for_postgres(sql: str, chart_type: str) -> str:
    
    sql = ensure_aliases_in_select(sql)
    # ✅ MySQL-style to PostgreSQL conversion
    sql = re.sub(r'\bMONTH\((.*?)\)', r'EXTRACT(MONTH FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bYEAR\((.*?)\)', r'EXTRACT(YEAR FROM \1)', sql, flags=re.IGNORECASE)

    # ✅ Fix direct EXTRACT(day FROM sale_date) patterns that cause type ambiguity
    sql = re.sub(
        r"(?i)EXTRACT\(\s*day\s+FROM\s+(\w+\.)?sale_date\s*\)",
        r"DATE_TRUNC('day', \1sale_date)",
        sql
    )
    print("SQL after initial normalization:", sql)
    # ✅ Clean up problematic EXTRACT patterns with DATE_TRUNC
    sql = re.sub(
        r"(?i)EXTRACT\(\s*day\s+FROM\s+DATE_TRUNC\('day',\s*(.*?)\)\s*\)",
        r"DATE_TRUNC('day', \1)",
        sql
    )

    # ✅ Normalize DATE(...) to DATE_TRUNC for time axis
    sql = re.sub(
        r"(?i)\bDATE\(\s*(\w+\.)?sale_date\s*\)",
        r"DATE_TRUNC('day', \1sale_date)",
        sql
    )
    print("SQL after 1 normalization:", sql)
    # ✅ Convert bare sale_date to DATE_TRUNC, but only in specific contexts
    # In SELECT clauses (but not WHERE/JOIN conditions)
    sql = re.sub(
        r"(?i)(SELECT\s+(?:(?!WHERE|FROM|JOIN).)*?)(\b(\w+\.)?sale_date\b)(?![^,]*\))",
        r"\1DATE_TRUNC('day', \3sale_date)",
        sql,
        flags=re.DOTALL
    )
    print("SQL after 2 normalization:", sql)
    
    # # ✅ Normalize sale_date in SELECT clause (only if not already inside DATE_TRUNC)
    # sql = re.sub(
    #     r"(?i)(SELECT\s+.*?)(\b(\w+\.)?sale_date\b)(\s+AS\s+\w+)?(?=,|\s+FROM)",
    #     lambda m: m.group(1) + f"DATE_TRUNC('day', {m.group(3) or ''}sale_date) AS time",
    #     sql,
    #     flags=re.DOTALL
    # )
    # ✅ Wrap EXTRACT(... FROM sale_date) with DATE_TRUNC for Grafana compatibility
    sql = re.sub(
        r"(?i)EXTRACT\s*\(\s*(\w+)\s+FROM\s+((\w+\.)?sale_date)\s*\)",
        r"EXTRACT(\1 FROM DATE_TRUNC('day', \2))",
        sql
    )
    # In GROUP BY clauses
    # sql = re.sub(
    #     r"(?i)(GROUP BY\s+(?:(?!ORDER|HAVING).)*?)(\b(\w+\.)?sale_date\b)",
    #     r"\1DATE_TRUNC('day', \3sale_date)",
    #     sql,
    #     flags=re.DOTALL
    # )
    
    # In ORDER BY clauses  
    sql = re.sub(
        r"(?i)(ORDER BY\s+)(.*?\b)(\w+\.)?sale_date\b",
        r"\1\2time",
        sql
    )
    

    print("SQL after 3 normalization:", sql)
    # ✅ Avoid double-nested DATE_TRUNC
    sql = re.sub(
        r"(?i)DATE_TRUNC\('day',\s*DATE_TRUNC\('day',\s*(.*?)\)\)",
        r"DATE_TRUNC('day', \1)",
        sql
    )
    
    # ✅ Clean up malformed aliases (remove extra content after AS time)
    sql = re.sub(
        r"(?i)(DATE_TRUNC\('day',\s*(\w+\.)?sale_date\))\s+AS\s+time[^,\n]*",
        r"\1 AS time",
        sql
    )
      # ✅ Clean up any other malformed aliases
    sql = re.sub(
        r"(?i)(DATE_TRUNC\('day',\s*(\w+\.)?sale_date\))\s+AS\s+[^,\n\s]*\([^)]*\)",
        r"\1 AS time",
        sql
    )
    # ✅ Standardize DATE_TRUNC aliases to 'time' 
    sql = re.sub(
        r"(?i)(DATE_TRUNC\('day',\s*(\w+\.)?sale_date\))\s+AS\s+(?!time\b)\w+",
        r"\1 AS time",
        sql
    )
    print("SQL after 4 normalization:", sql)
    # ✅ Add AS time alias if missing in SELECT
    
   
    re.sub(
        r"""(?ix)                            # Case-insensitive, verbose mode
        (SELECT\s+.*?)                       # Match SELECT clause up to DATE_TRUNC
        \bDATE_TRUNC\(\s*'day',\s*((\w+\.)?sale_date)\s*\)  # Match DATE_TRUNC('day', sale_date)
        
        ([^;]*?\bFROM\b)                     # Match up to FROM
        """,
        r"\1DATE_TRUNC('day', \2) AS time\4",
        sql,
        flags=re.DOTALL
    )
    

        
    print("SQL after 5 normalization:", sql)
    # ✅ Replace DATE_TRUNC in GROUP BY with 'time' if alias exists
    if "AS time" in sql:
        sql = re.sub(
            r"(?i)(GROUP BY[^;]*?)DATE_TRUNC\('day',\s*(\w+\.)?sale_date\)",
            r"\1time",
            sql
        )
        
        sql = re.sub(
            r"(?i)(ORDER BY[^;]*?)DATE_TRUNC\('day',\s*(\w+\.)?sale_date\)",
            r"\1time",
            sql
        )
    print("SQL after 6 normalization:", sql)
    # 🔁 Keep only 'time' in ORDER BY clause, discard others
    sql = force_order_by_time_if_column_exists(sql)
    print("Chart_type after 7 normalization:", chart_type)
    if chart_type == "bar chart":
       sql = cast_labels_to_text(sql)
    print("SQL after 8 normalization:", sql)
    # ✅ Fix WHERE clause EXTRACT comparisons
    sql = re.sub(
        r"(?i)WHERE\s+DATE_TRUNC\('day',\s*(\w+\.)?sale_date\)\s*=\s*EXTRACT\(\s*day\s+FROM\s+'([^']+)'\s*\)",
        r"WHERE DATE_TRUNC('day', \1sale_date) = '\2'::date",
        sql
    )
    
    
    return sql

# === Check if the prompt is related to SQL/data analysis ===
def is_sql_related(question: str):
    keywords = ["sales", "total", "amount", "date", "report", "sold", "data"]
    return any(word in question.lower() for word in keywords)

# === Extract SQL from LLM response ===
# def extract_sql(text: str) -> str:
#     # Prefer SQL block inside triple backticks
#     match = re.search(r"```(?:sql)?\s*(SELECT.*?);?\s*```", text, re.DOTALL | re.IGNORECASE)
#     if match:
#         return match.group(1).strip()
#     # Fallback: Match raw SELECT...FROM line
#     match = re.search(r"(SELECT\s.+?\sFROM\s.+?);", text, re.IGNORECASE | re.DOTALL)
#     return match.group(1).strip() if match else ""

def extract_sql(text: str) -> List[str]:
    # Match all SQL blocks in triple backticks
    sql_blocks = re.findall(r"```(?:sql)?\s*(SELECT.*?);?\s*```", text, re.DOTALL | re.IGNORECASE)
    
    # Fallback: try to find all inline SELECT...FROM queries
    if not sql_blocks:
        sql_blocks = re.findall(r"(SELECT\s.+?\sFROM\s.+?);", text, re.IGNORECASE | re.DOTALL)
    
    return [sql.strip() for sql in sql_blocks]


# ===  Ask LLM to extract table name(s) ===
def extract_table_name_with_llm(prompt: str) -> str:
    res = requests.post("http://ollama:11434/api/generate", json={
        "model": "llama3",
        "prompt": f"You are a SQL assistant.\nGiven this question, extract the table name(s) mentioned.\n\nQuestion: {prompt}\n\nReturn only the table names, comma-separated if more than one. The sales table contains person_id, product_id, sale_date is a DATE, and value. The person table stores id and name of customers. The products table has id and name.  " 
    })
   
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



def extract_chart_type_with_llama(prompt: str) -> str:
    """
    Uses Ollama with LLaMA model to extract the chart type from a natural language prompt.
    Returns one of: 'bar chart', 'line chart', 'pie chart', 'table', 'scatter plot', 'area chart'.
    Defaults to 'line chart' if no match is found.
    """
    system_instruction = (
        "You are a data visualization assistant. "
        "Given a user's prompt, identify the intended type of chart to use. "
        "Return only one of: 'bar chart', 'line chart', 'pie chart', 'table', 'scatter plot', 'area chart'. "
        "Be concise. Output only the chart type."
    )

    full_prompt = f"{system_instruction}\n\nUser prompt: {prompt}\n\nChart type:"

    try:
        response = requests.post("http://ollama:11434/api/generate", json={
            "model": "llama3",
            "prompt": full_prompt
        })
       
        raw_text = "".join(
            json.loads(line).get("response", "")
            for line in response.text.strip().split("\n")
            if line.strip().startswith("{")
        ).strip().lower()

        known_types = [
            "bar chart",
            "line chart",
            "pie chart",
            "table",
            "scatter plot",
            "area chart"
        ]
        
        for chart_type in known_types:
            if chart_type in raw_text:
                return chart_type

        return "line chart"  # default fallback

    except Exception as e:
        print("Error extracting chart type:", e)
        return "line chart"  # fallback in case of API or parsing failure

# === MAIN ENTRYPOINT: POST /ask ===
@app.post("/ask")
def ask_ollama(prompt_request: PromptRequest):
    # Step 1: Replace natural-language time expressions
   
    prompt = prompt_request.prompt
    print(f"Received prompt: {prompt}")
    # Step 2: Check if it's a SQL/data-related query
    if is_sql_related(prompt):
        # Step 3: Ask LLM to find relevant table names
        table_names_str = extract_table_name_with_llm(prompt)
        table_names = [name.strip() for name in table_names_str.split(",")]
        print(f"Extracted table names: {table_names}")
        # Step 4: Get schema and example data for each table via MCP
        schema_parts = []
        for table in table_names:
            schema = get_table_schema(table)
            example = get_table_example(table)
            schema_parts.append(f"Table: {table}\n{schema}\nExample Row: {json.dumps(example)}")
        
        # Combine context + user question into final LLM prompt
        full_prompt = "\n\n".join(schema_parts) + f"\n\nUser question: {prompt}"
    else:
        # If not SQL-related, send question directly to LLM
        full_prompt = prompt
        
    print(f"Full prompt sent to LLM: {full_prompt}")
    # Step 5: Send the structured prompt to Ollama (running LLaMA3 model)
    response = requests.post("http://ollama:11434/api/generate", json={
        "model": "llama3",
        "prompt": full_prompt
    })

    # Step 6: Parse the streamed LLM response
    llm_answer = "".join(
        json.loads(line).get("response", "") 
        for line in response.text.strip().split("\n") 
        if line.strip().startswith("{")
    )
    
    # Extract multiple SQLs and chart types
    sql_list = extract_sql(llm_answer)
    chart_types = []
    for part in prompt.split(" and "):
        chart_types.append(extract_chart_type_with_llama(part.strip()))

    # Normalize each SQL
    normalized_sqls = [
        normalize_sql_for_postgres(sql, chart)
        for sql, chart in zip(sql_list, chart_types)
    ]
    
    print("🧪 Normalized SQLs:", normalized_sqls)
    print("🧪 Chart Types:", chart_types)

    try:
        # Execute all SQLs
        db_response = requests.post("http://mcp_server:8000/execute", json={
            "sql": normalized_sqls,
            "chart_type": chart_types
        })
        result = db_response.json()
        print(f"SQL execution result: {result}")

        # Create multi-panel dashboard
        creds = requests.get("http://grafana-mcp:8000/credentials")
        dashboard_json = requests.post("http://mcp_server:8000/grafana_json", json={
            "sql": normalized_sqls,
            "chart_type": chart_types,
            "title": "LLM: Multi-panel Dashboard"
        }).json()

        headers = {"Content-Type": "application/json"}
        create_dash = requests.post("http://grafana-mcp:8000/create_dashboard", headers=headers, json=dashboard_json)
        grafana_result = create_dash.json()

    except Exception as e:
        result = {"error": str(e)}
        grafana_result = {"error": str(e)}

    

    # Step 7: If response contains SQL, try executing it
    # if "select" in llm_answer.lower() and "from" in llm_answer.lower():
    #     try:
    #         chart_type = extract_chart_type_with_llama(prompt)
    #         sql_query = normalize_sql_for_postgres(extract_sql(llm_answer),chart_type)
    #         # Send SQL to the MCP server to execute
    #         print(f"Executing SQL: {sql_query}")
    #         db_response = requests.post("http://mcp_server:8000/execute", json={"sql": sql_query})
    #         result = db_response.json()
    #         print(f"SQL execution result: {result}")
    #         # === Grafana Integration ===
    #         creds = requests.get("http://grafana-mcp:8000/credentials")
    #         dashboard_json = requests.post("http://mcp_server:8000/grafana_json", json={"sql": sql_query, "chart_type": chart_type}).json()
    #         headers = {"Content-Type": "application/json"}
            
    #         # Create Grafana dashboard using the MCP server
    #         create_dash = requests.post("http://grafana-mcp:8000/create_dashboard", headers=headers, json=dashboard_json)
    #         grafana_result = create_dash.json()
            
            
    #     except Exception as e:
    #         result = {"error": str(e)}
    #         grafana_result = {"error": str(e)}
    # else:
    #     # Otherwise just return the plain answer from LLM
    #     result = {"answer": llm_answer}
    #     grafana_result = {}

    # Step 8: Log prompt + LLM result + SQL result to MLflow
    # log_to_mlflow(prompt_request.prompt, llm_answer, result)

    # Step 9: Return both the LLM's response and final result
    # return {"query": llm_answer, "result": result, "grafana": grafana_result}
    return {"query": llm_answer, "result": result}
