import json
import re
import yaml
import requests
from typing import List, Dict, Any
from datetime import datetime
from src.metrics import time_call


def build_schema_parts(prompt: str) -> List[str]:
    """
    Extract table schema parts based on prompt context.
    You can customize to use your existing extract_table_name_with_llm and
    context_retriever.get_table_schema, get_table_example.
    """
    # Example stub; replace with your actual logic
    tables = prompt.lower().split()
    schema_parts = []
    for tbl in tables:
        if tbl in ("sales","person","products","payment_type","orders"):
            schema_parts.append(f"Table: {tbl}")
    return schema_parts


def calculate_gridpos(panel_id: int, total_panels: int) -> Dict[str, int]:
    """
    Calculate gridPos for a panel index given total number of panels.
    """
    i = panel_id
    if i == total_panels - 1 and total_panels % 2 == 1:
        x = 0
        w = 24
        y = (i // 2) * 8
    else:
        x = 0 if i % 2 == 0 else 12
        w = 12
        y = (i // 2) * 8
    return {"x": x, "y": y, "w": w, "h": 8}


def default_panel_options() -> Dict[str, Any]:
    """
    Return default options block for panels.
    """
    return {
        "tooltip": {"mode": "single"},
        "legend": {"displayMode": "list", "placement": "bottom"},
        # include reduceOptions if needed
    }


def default_timepicker_options() -> Dict[str, Any]:
    """
    Return default timepicker configuration.
    """
    return {
        "refresh_intervals": ["5s","10s","30s","1m","5m","15m","30m","1h","2h","1d"],
        "time_options": ["5m","15m","1h","6h","12h","24h","2d","7d","30d","90d","6M","1y","5y"],
    }


def call_ollama_structured(prompt: str, model: str) -> Any:
    """
    Wrapper around requests.post to Ollama, stripping non-JSON and returning parsed JSON.
    """
    resp = requests.post(
        "http://ollama:11434/v1/chat/completions",
        json={"model":model, "messages":[{"role":"user","content":prompt}], "temperature":0.1},
        timeout=(5, None)
    )
    
    resp.raise_for_status()
    print(f"🔥 Response from Ollama: {resp.text}")
    content = resp.json()["choices"][0]["message"]["content"]
    if content.startswith("["):
        
        try:
            return  json.loads(content)  # Validate JSON structure
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON structure: {e}")
            raise ValueError("Invalid JSON structure in Ollama response")   
    else:
        # fall back to extracting a single object
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end == -1:
            print("❌ No JSON object found in Ollama response")
            raise ValueError("No JSON object found in Ollama response")
        return json.loads(content[start:end])
 

def load_validation_rules(yaml_file_path: str = "src/validation_tags.yaml") -> Dict[str, Any]:
    """
    Load the YAML file containing validation tag rules.
    """
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)
