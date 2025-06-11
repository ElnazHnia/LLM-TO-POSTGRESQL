from fastapi import FastAPI, Request
import requests
from dotenv import load_dotenv
import os
import json

app = FastAPI()

load_dotenv()  # Loads .env file

GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY")
GRAFANA_URL = os.getenv("GRAFANA_URL")


@app.get("/credentials")
def get_credentials():
    return {
        "host": "postgres-server-mcp",
        "port": 5432,
        "user": "admin",
        "password": "password",
        "database": "llm_data_db"
    }

# @app.post("/create_dashboard")
# async def create_dashboard(request: Request):
#     json_data = await request.json()
#     print("GRAFANA_API_KEY ", GRAFANA_API_KEY)	
#     print("GRAFANA_URL ", GRAFANA_URL)
#     headers = {"Authorization": f"Bearer {GRAFANA_API_KEY}", "Content-Type": "application/json"}

#     response = requests.post(f"{GRAFANA_URL}/api/dashboards/db", headers=headers, json=json_data)
#     return response.json()


@app.post("/create_dashboard")
async def create_dashboard(request: Request):
    json_data = await request.json()

    headers = {
        "Authorization": f"Bearer {GRAFANA_API_KEY}",
        "Content-Type": "application/json"
    }

    # Get folder UID dynamically (optional)
    # folders = requests.get(f"{GRAFANA_URL}/api/folders", headers=headers).json()
    response = requests.get(f"{GRAFANA_URL}/api/folders", headers=headers)
    print("📦 grafana-mcp Folders raw response:", response.text)
    print("📦 grafana-mcp Folders status code:", response.status_code)
    folders = response.json()
    folder_uid = next((f["uid"] for f in folders if f["title"] == "LLM_To_POSTGRESQL_FOLDER"), None)

    # Clean any fields that override folderUid
    for key in ["uid", "id", "folderId"]:
        json_data["dashboard"].pop(key, None)
    
    # Inject folder info and overwrite flag
    json_data["folderUid"] = folder_uid
    json_data["overwrite"] = False

    # Debug print final JSON
    
    print("✅ Final JSON to POST:", json.dumps(json_data, indent=2))
    
    
   
    response = requests.post(f"{GRAFANA_URL}/api/dashboards/db", headers=headers, json=json_data)
    
     # NEW: print actual result from Grafana
    print("📩 Grafana API Response Status:", response.status_code)
    print("📩 Grafana API Response Body:", response.text)
    
    return response.json()


# @app.post("/create_dashboard")
# def create_dashboard(payload: dict):
#     return {"status": "ok"}  # Simplified placeholder