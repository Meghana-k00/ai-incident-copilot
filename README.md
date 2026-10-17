# AI Incident Copilot

An AI-powered incident investigation assistant built with Python, FastAPI, SQLite, ChromaDB, and a locally hosted Ollama language model.

## Overview

AI Incident Copilot is a portfolio project designed to help users record, manage, and investigate service incidents through a REST API.

The application combines structured incident management with AI-assisted troubleshooting. It uses an Ollama-hosted language model to generate possible root causes, diagnostic checks, and remediation suggestions. A ChromaDB-based retrieval component provides relevant indexed knowledge to support context-aware investigations.

The project also includes automated tests for API endpoints, incident validation, retrieval behavior, investigation logic, and error handling.

**Important:** AI-generated findings are hypotheses, not verified diagnoses. Human review is required before applying any remediation.

## Features

- Create, list, retrieve, update the status of, and delete incident records.
- Validate incident details and severity using request models.
- Store incident records in SQLite.
- Generate AI-assisted incident investigation findings using Ollama.
- Use the Qwen 2.5 3B model for local language generation.
- Retrieve relevant indexed incident knowledge using ChromaDB.
- Handle missing incidents, invalid inputs, unavailable Ollama services, and invalid model responses.
- Expose REST API endpoints through FastAPI.
- Provide interactive API documentation through Swagger UI.
- Include automated tests for API behavior, retrieval, investigation quality, and failure scenarios.

## Technology Stack

- **Language:** Python
- **API framework:** FastAPI
- **ASGI server:** Uvicorn
- **Database:** SQLite
- **Vector database:** ChromaDB
- **Local LLM runtime:** Ollama
- **Language model:** Qwen 2.5 3B (`qwen2.5:3b`)
- **Testing:** Pytest and HTTPX
- **Version control:** Git

## Project Structure

```text
ai-incident-copilot/
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── ingest_knowledge.py
│   ├── investigation.py
│   ├── main.py
│   ├── models.py
│   └── retrieval.py
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   ├── test_incident_endpoints.py
│   ├── test_incidents.py
│   ├── test_investigation.py
│   ├── test_main.py
│   └── test_retrieval.py
├── evaluation/
│   ├── run_evaluation.py
│   └── test_investigation_quality.py
├── docs/
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

The structure above describes the main project components. Additional files may exist in the local working directory.

Local databases, vector-store data, virtual environments, cache files, and secrets should generally be excluded from version control. Include only sanitized sample data when it is intentionally required.

## Prerequisites

Before running the project, install:

- Python 3.11 or a compatible version supported by the dependencies.
- Ollama.
- Git, if cloning the repository.

The project uses a locally hosted model, so a paid LLM API key is not required for the configured local inference workflow.

## Setup and Installation

### 1. Clone the repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <your-repository-url>
cd ai-incident-copilot
```

Alternatively, open your existing project directory in VS Code.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Install Ollama from its official website:

https://ollama.com/

Download the configured model:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running locally and the model is available before requesting an AI investigation.

### 5. Start the application

From the project root, run:

```bash
python -m uvicorn src.main:app --reload
```

The API server should be available at:

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **Health endpoint:** http://127.0.0.1:8000/health

The application initializes its database and incident retriever during startup.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Returns the API welcome message and status |
| GET | `/health` | Returns the service health response |
| POST | `/incidents` | Creates an incident |
| GET | `/incidents` | Lists incidents |
| GET | `/incidents/{incident_id}` | Retrieves an incident by ID |
| PATCH | `/incidents/{incident_id}/status` | Updates an incident's status |
| DELETE | `/incidents/{incident_id}` | Deletes an incident |
| POST | `/incidents/{incident_id}/investigate` | Generates AI-assisted investigation findings |

### Example: Create an incident

Send a `POST` request to `/incidents` with a JSON body such as:

```json
{
  "title": "Database connection timeout",
  "description": "The application is unable to connect to the database. Connection attempts are timing out.",
  "severity": "high",
  "service_name": "order-service",
  "status": "open"
}
```

The accepted request fields and validation rules are defined by the application's request models.

A successful creation returns HTTP `201` and the created incident record.

### Investigation behavior

Send a `POST` request to `/incidents/{incident_id}/investigate` to request an investigation of an existing incident.

The endpoint retrieves the incident, passes it to the investigation logic, and supplies the configured retriever when available.

The endpoint handles the following error scenarios:

- `404 Not Found` — the requested incident does not exist.
- `422 Unprocessable Entity` — the investigation input is invalid.
- `503 Service Unavailable` — an investigation runtime failure occurs, such as an unavailable Ollama service.

Successful investigation responses include the generated findings and associated response fields.

## Testing

Run the complete automated test suite from the project root:

```bash
python -m pytest -v
```

### Investigation-quality evaluation

Run the dedicated investigation-quality tests:

```bash
python -m pytest evaluation/test_investigation_quality.py -v
```

The evaluation suite checks behaviors such as retrieved-knowledge usage, handling missing incident information, and rejection of invalid model responses.

A convenience runner is also available:

```bash
python -m pytest evaluation/run_evaluation.py -v
```

### Verified development test result

**70 tests passed with no failures** in the complete test suite during development. The dedicated investigation-quality suite also passed its 10 tests.

Run the commands above to reproduce the results in your own environment after cloning the repository and installing its dependencies.

Tests use mocks for external model calls where appropriate, allowing failure scenarios to be tested without depending on live AI inference for every test.

## Limitations

- AI-generated root causes and remediation suggestions require human verification.
- Investigation generation depends on a running Ollama service and an available local model.
- Retrieval quality depends on the indexed knowledge, embedding configuration, and relevance of the source documents.
- The health endpoint reports the API's health response; it should not be interpreted as a comprehensive check of every external dependency.
- The project is intended as a development and portfolio demonstration, not as a production-ready incident-management platform.
- Authentication, authorization, production deployment, and operational monitoring may require additional implementation.

## Future Improvements

Potential enhancements include:

- Expand integration tests for complete incident lifecycle workflows.
- Evaluate retrieval and generated findings against a representative incident dataset.
- Add structured investigation results with evidence references.
- Introduce authentication and role-based access control.
- Add production deployment configuration and operational observability.
- Improve dependency health checks and service-level monitoring.

## Disclaimer

AI Incident Copilot is intended for educational and portfolio demonstration purposes. It assists incident investigation but does not independently verify root causes, guarantee remediation outcomes, or replace qualified human judgment.