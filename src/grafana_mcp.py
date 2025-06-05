from fastapi import FastAPI, Request
import requests
from dotenv import load_dotenv
import os

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
    
    folder_uid = next((f["uid"] for f in folders if f["title"] == "fastapi-dashboards"), None)
    print(f"Using folder UID: {folder_uid}")
    print("GRAFANA_API_KEY ", GRAFANA_API_KEY)	
    print("GRAFANA_URL ", GRAFANA_URL)
    # Inject folderUid into payload
    json_data["folderUid"] = folder_uid or ""
    json_data["overwrite"] = True
   
    response = requests.post(f"{GRAFANA_URL}/api/dashboards/db", headers=headers, json=json_data)
    return response.json()


# @app.post("/create_dashboard")
# def create_dashboard(payload: dict):
#     return {"status": "ok"}  # Simplified placeholder