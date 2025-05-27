from fastapi import FastAPI
from pydantic import BaseModel
import requests
import json
import re
import dateparser
from datetime import datetime, timedelta

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
def normalize_sql_for_postgres(sql: str) -> str:
    sql = re.sub(r'\bMONTH\((.*?)\)', r'EXTRACT(MONTH FROM \1)', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bYEAR\((.*?)\)', r'EXTRACT(YEAR FROM \1)', sql, flags=re.IGNORECASE)
    return sql

# === Check if the prompt is related to SQL/data analysis ===
def is_sql_related(question: str):
    keywords = ["sales", "total", "amount", "date", "report", "sold", "data"]
    return any(word in question.lower() for word in keywords)

# === Extract SQL from LLM response ===
def extract_sql(text: str) -> str:
    # Prefer SQL block inside triple backticks
    match = re.search(r"```(?:sql)?\s*(SELECT.*?);?\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: Match raw SELECT...FROM line
    match = re.search(r"(SELECT\s.+?\sFROM\s.+?);", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

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

    #  Extract just the table names
    table_line = raw_response.split("\n")[-1]  # get the last line (which should contain only the names)
    clean_tables = [name.strip() for name in table_line.split(",") if name.strip().isidentifier()]

    return ",".join(clean_tables)


# === MAIN ENTRYPOINT: POST /ask ===
@app.post("/ask")
def ask_ollama(prompt_request: PromptRequest):
    # Step 1: Replace natural-language time expressions
   
    prompt = prompt_request.prompt
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
        print(f"Schema parts: {schema_parts}")
        # Combine context + user question into final LLM prompt
        full_prompt = "\n\n".join(schema_parts) + f"\n\nUser question: {prompt}"
    else:
        # If not SQL-related, send question directly to LLM
        full_prompt = prompt

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

    # Step 7: If response contains SQL, try executing it
    if "select" in llm_answer.lower() and "from" in llm_answer.lower():
        try:
            sql_query = normalize_sql_for_postgres(extract_sql(llm_answer))
            # Send SQL to the MCP server to execute
            print(f"Executing SQL: {sql_query}")
            db_response = requests.post("http://mcp_server:8000/execute", json={"sql": sql_query})
            result = db_response.json()
        except Exception as e:
            result = {"error": str(e)}
    else:
        # Otherwise just return the plain answer from LLM
        result = {"answer": llm_answer}

    # Step 8: Log prompt + LLM result + SQL result to MLflow
    # log_to_mlflow(prompt_request.prompt, llm_answer, result)

    # Step 9: Return both the LLM's response and final result
    return {"query": llm_answer, "result": result}
