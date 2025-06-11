# LLM-to-PostgreSQL

This project integrates a Large Language Model (LLM) with PostgreSQL to convert natural language queries into SQL and visualize the results using Grafana. It enables users to interact with a database using simple English prompts, lowering the barrier to data analysis.

## 🔍 Features

- 🧠 Natural language to SQL translation using LLMs (e.g., OpenAI)
- 🗃️ Querying a PostgreSQL database
- 📊 Visualizing SQL results in Grafana dashboards
- 🔁 Easily extendable and configurable

## 📦 Technologies Used

- **Python** for backend logic
- **PostgreSQL** as the database
- **OpenAI API** (or similar LLM)
- **Grafana** for real-time data visualization
- **SQLAlchemy**, **psycopg2** for database interactions

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL instance running
- Grafana set up and accessible
- OpenAI API Key (if using GPT)

### Installation

1. **Clone the repository:**

```bash  
git clone https://github.com/ElnazHnia/LLM-TO-POSTGRESQL.git
cd LLM-TO-POSTGRESQL
```

2. **Build and run the project using Docker:**

Make sure Docker and Docker Compose are installed, then run:

```bash
docker compose up --build -d
```

This will start the following services:

- FastAPI app (on port 8000)
- MCP server (on port 8001)
- Grafana-MCP (on port 8002)
- PostgreSQL (port 5432)
- Ollama LLM runtime (port 11434)
- Grafana (on port 3000)
- Qdrant (on port 6333)

You can view the running containers with:

```bash
docker ps
```

Grafana will be available at: http://localhost:3000  
FastAPI docs (if enabled) at: http://localhost:8000/docs



3. **Configure environment variables:**

Create a `.env` file in the project root with the following content:

```bash
# --- PostgreSQL connection details ---
POSTGRES_URL=postgres
POSTGRES_PORT=5432
POSTGRES_USER=admin
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DATABASE=llm_data_db

# --- MinIO configuration ---
MINIO_HOST=minio
MINIO_ROOT_USER=your_minio_user
MINIO_ROOT_PASSWORD=your_minio_password
MINIO_PORT_API=9000
MINIO_PORT_UI=9001
MINIO_DATALAKE_BUCKET=datalake

# --- MLflow configuration ---
MLFLOW_SERVER_URL=http://mlflow:5000
MLFLOW_BUCKET_NAME=mlflow-artifacts
MLFLOW_PORT=5000

# --- Grafana configuration ---
GRAFANA_API_KEY=your_grafana_api_key
GRAFANA_URL=http://grafana:3000
```


4. **Start all services with Docker**  
*(already covered above)*  
Run the following if you haven’t already:

```bash
docker compose up --build -d
```
5.  **Connect Grafana to PostgreSQL and Upload Dashboards (via Terminal)**

To automate Grafana configuration without using the browser UI, follow these steps using the terminal. 
This avoids using deprecated API keys and instead leverages **Grafana service accounts**,
which are the new standard.

### 🛠 Step 1: Create a Service Account

Run the following command to create a service account (role must be `"Admin"`):

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"name": "fastapi-dashboard-bot-elnaz", "role": "Admin"}' \
     http://admin:admin@localhost:3000/api/serviceaccounts
```

### 🔐 Step 2: Generate a Token for the Service Account

Replace `2` with the correct service account ID (from Step 1 response):

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"name": "fastapi-dashboard-bot-elnaz-token"}' \
     http://admin:admin@localhost:3000/api/serviceaccounts/2/tokens
```

Save the token (`"key": "..."`) from the response — this replaces old API keys.

### 📁 Step 3: Create a Folder to Store Dashboards

Use the token from Step 2 in the Authorization header:

```bash
curl -X POST http://localhost:3000/api/folders \
     -H "Authorization: Bearer YOUR_SERVICE_ACCOUNT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{ "title": "LLM_To_POSTGRESQL_FOLDER" }'
```

### 🧭 Step 4: Get Your PostgreSQL Datasource UID

```bash
curl -X GET http://localhost:3000/api/datasources \
     -H "Authorization: Bearer YOUR_SERVICE_ACCOUNT_TOKEN"
```

From the response, find your PostgreSQL datasource and note its `"uid"` value.

Example to insert in your dashboard JSON:

```json
"datasource": {
  "type": "grafana-postgresql-datasource",
  "uid": "aeo8prusu1i4gc"
}
```

Make sure your dashboard JSON includes this datasource UID before uploading it.

---

### ✅ PostgreSQL Datasource Configuration in Grafana

In the Grafana UI or API, configure your PostgreSQL connection like this:

| Grafana Field     | Value          | Source (.env)               |
|------------------|----------------|-----------------------------|
| Host             | `postgres:5432`| `POSTGRES_URL` + port       |
| Database Name    | `llm_data_db`  | `POSTGRES_DATABASE`         |
| Username         | `admin`        | `POSTGRES_USER`             |
| Password         | `your_password`| `POSTGRES_PASSWORD`         |
| TLS/SSL Mode     | `disable`      | (default unless needed)     |

---

### 🔐 Why Use Service Accounts?

Grafana **deprecated API keys** in favor of **service accounts**, which offer better security and permission control. A service account is a secure, role-based identity meant for automation or bots — ideal for uploading dashboards, managing data sources, or creating folders programmatically.

To assign folder access:
1. Go to the folder in Grafana UI.
2. Click the "folder action" → "manage permission".
3. Click "Add Permission", select "Service Account", and assign access.



6. **View the results in Grafana:**

- Connect Grafana to your PostgreSQL instance
- Create dashboards manually or import them from the `grafana/` folder if available

## 📁 Project Structure

```bash
ELNAZ_THESIS_LLM/
├── .env                         # Environment variables (do not commit)
├── .gitignore                   # Git ignore file
├── docker-compose.yml           # Docker orchestration
├── Dockerfile                   # Main FastAPI service
├── Dockerfile.grafana-mcp       # Grafana MCP-specific container
├── Dockerfile.mcp               # MCP API container
├── Dockerfile.mlflow            # MLflow container
├── mlflow_requirements.txt      # MLflow dependencies
├── requirements.txt             # Python requirements
├── src/                         # Source code
│   ├── grafana_mcp.py           # Grafana management via MCP
│   ├── llm_api.py               # FastAPI LLM endpoint
│   ├── pg_mcp.py                # PostgreSQL MCP logic
│   └── Dockerfile               # Possibly linked to src-specific service
├── rag/
│   └── context_retriever.py     # RAG context fetching logic
├── postgres_data/               # Volume for PostgreSQL
├── mlflow_logger/               # MLflow logging code
├── minio_data/                  # MinIO volume
├── ollama_models/               # LLM models directory
└── __pycache__/                 # Python bytecode cache
```

