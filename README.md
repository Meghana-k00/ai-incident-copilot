# AI Incident Copilot

An AI-powered incident investigation and human approval workflow application built with Python, FastAPI, SQLite, ChromaDB, Ollama, and LangGraph.

## Overview

AI Incident Copilot is a portfolio project designed to help users record, manage, and investigate service incidents through a REST API.

The application combines structured incident management with AI-assisted troubleshooting. It uses a locally hosted Ollama language model to generate possible root causes, diagnostic checks, and remediation suggestions. A ChromaDB-based retrieval component provides relevant indexed knowledge to support context-aware investigations.

The application also includes a human approval workflow built with LangGraph. A proposed incident-related action can be submitted for review, a reviewer can record a decision, and the workflow can resume from its saved checkpoint. Approval decisions are persisted, while decision claims and resume leases help coordinate concurrent requests and reduce duplicate workflow processing.

The project includes automated tests for API behavior, incident validation, investigation logic, retrieval, approval workflows, and failure-recovery scenarios.

**Important:** AI-generated findings are hypotheses, not verified diagnoses. Human review is required before applying any remediation.

## Features

### Incident management

- Create, list, retrieve, update the status of, and delete incident records.
- Validate incident details and severity using request models.
- Store incident records in SQLite.
- Handle missing incidents and invalid requests through API error responses.

### AI-assisted investigation

- Generate investigation findings using a locally hosted Ollama language model.
- Use the Qwen 2.5 3B model (`qwen2.5:3b`) for local language generation.
- Retrieve relevant indexed incident knowledge using ChromaDB.
- Generate possible root causes, diagnostic checks, and remediation suggestions.
- Handle unavailable model services, invalid model responses, and retrieval failures.

### Human approval workflow

- Create approval requests for proposed incident-related actions.
- Use LangGraph to manage a workflow that pauses for a human decision and resumes after review.
- Persist workflow checkpoints using SQLite-backed checkpoint storage.
- Protect approval review and decision endpoints with reviewer authentication.
- Persist approval decisions and reviewer information.
- Use durable decision claims to coordinate repeated or conflicting decision requests.
- Use resume leases to coordinate concurrent attempts to resume an approval workflow.
- Return appropriate conflict or error responses when an approval cannot be processed.

**Scope clarification:** The application supports a reviewer decision workflow. Automatic assignment to a particular team lead or agent, organizational escalation policies, and a complete enterprise role-based approval system are not claimed as implemented features.

### API and testing

- Expose REST API endpoints through FastAPI.
- Provide interactive API documentation through Swagger UI and ReDoc.
- Include automated tests for API behavior, investigation, retrieval, approval processing, and failure scenarios.

## Technology Stack

- **Language:** Python
- **API framework:** FastAPI
- **ASGI server:** Uvicorn
- **Relational database:** SQLite
- **Vector database:** ChromaDB
- **Local LLM runtime:** Ollama
- **Language model:** Qwen 2.5 3B (`qwen2.5:3b`)
- **Workflow orchestration:** LangGraph
- **Workflow checkpointing:** SQLite-backed LangGraph checkpoint storage
- **Testing:** Pytest and HTTPX
- **Version control:** Git

The configured local inference workflow does not require a paid LLM API key.

## Project Structure

The following is a high-level view of the main project components. The exact repository may contain additional files.

```text
ai-incident-copilot/
├── src/
│   ├── database.py
│   ├── auth.py
│   ├── approval_workflow.py
│   ├── investigation.py
│   ├── ingest_knowledge.py
│   ├── main.py
│   ├── models.py
│   └── retrieval.py
├── tests/
│   ├── test_health.py
│   ├── test_incident_endpoints.py
│   ├── test_incidents.py
│   ├── test_investigation.py
│   ├── test_main.py
│   ├── test_retrieval.py
│   └── additional approval workflow tests
├── evaluation/
│   ├── run_evaluation.py
│   └── test_investigation_quality.py
├── docs/
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

Local databases, workflow checkpoint files, vector-store data, virtual environments, cache files, and secrets should generally be excluded from version control. Commit sample data only when it is sanitized and intentionally required.

## Prerequisites

Install the following before running the application:

- Python 3.11 or a compatible version supported by the dependencies.
- Ollama.
- Git, if cloning the repository.

The configured local model must be available before requesting an AI investigation. The approval workflow also requires its configured checkpoint storage to be accessible.

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

The application initializes its database, incident retriever, and approval workflow checkpointing during startup.

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
| POST | `/incidents/{incident_id}/approval-request` | Creates an approval request for a proposed action |
| GET | `/approvals/pending` | Lists pending approval requests; reviewer authentication required |
| GET | `/approvals/{approval_id}` | Retrieves an approval request; reviewer authentication required |
| POST | `/approvals/{approval_id}/decision` | Records a reviewer decision and resumes the workflow; reviewer authentication required |

The incident-management endpoints and approval-request creation endpoint do not currently have equivalent reviewer protection. See the Limitations section before considering deployment.

## Example: Create an Incident

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

## AI-Assisted Investigation

Send a `POST` request to:

```text
/incidents/{incident_id}/investigate
```

The endpoint retrieves the incident, passes it to the investigation logic, and supplies the configured retriever when available.

The investigation uses the local model to produce findings informed by the incident details and retrieved knowledge.

Expected error scenarios include:

- `404 Not Found` — the requested incident does not exist.
- `422 Unprocessable Entity` — the investigation input is invalid.
- `503 Service Unavailable` — an investigation runtime failure occurs, such as an unavailable Ollama service.

Successful investigation responses include the generated findings and associated response fields.

AI-generated recommendations must be verified before any operational action is taken.

## Human Approval Workflow

The application supports a reviewer decision workflow for proposed incident-related actions.

### Workflow lifecycle

1. An approval request is created for an existing incident.
2. The application stores the approval request and associates it with a workflow thread.
3. LangGraph executes the workflow until it reaches the human review step.
4. An authenticated reviewer retrieves pending requests or a specific approval request.
5. The reviewer submits a decision through the decision endpoint.
6. The application claims the decision and coordinates workflow resumption using a resume lease.
7. The workflow resumes from its checkpoint, and the approval decision is recorded.

Decision claims help prevent conflicting decisions from being processed for the same approval. Resume leases coordinate concurrent attempts to resume a workflow.

### Important workflow boundaries

- Reviewer authentication is required for listing pending approvals, retrieving an approval, and submitting a decision.
- Creating an approval request is not currently protected by the same reviewer dependency.
- The workflow supports human review, but it does not automatically assign approvals to named team leads or agents.
- Resume leases have a configured expiry. Long-running workflow execution beyond that duration requires further consideration.
- This implementation is a portfolio demonstration, not a complete enterprise approval or change-management platform.

Use the interactive Swagger documentation at `/docs` to inspect request schemas and try the available endpoints.

## Testing

Run the complete automated test suite from the project root:

```bash
python -m pytest -q
```

For verbose output:

```bash
python -m pytest -v
```

### Investigation-quality evaluation

Run the dedicated investigation-quality tests:

```bash
python -m pytest evaluation/test_investigation_quality.py -v
```

The evaluation tests check behaviors such as retrieved-knowledge usage, handling missing incident information, and rejection of invalid model responses.

A convenience runner is also available:

```bash
python -m pytest evaluation/run_evaluation.py -v
```

### Verified development test result

**112 tests passed** in the latest confirmed full test-suite run during development.

The test suite covers API behavior, incident validation, investigation logic, retrieval behavior, approval workflows, and failure-recovery scenarios.

Tests use mocks for external model calls where appropriate. Passing automated tests therefore does not guarantee that every live model response will be accurate or that all external integrations will behave identically in production.

Run the test commands above to verify the current checkout in your own environment.

## Security and Data Handling

- Keep `.env` files, credentials, tokens, and other secrets out of version control.
- Keep local SQLite databases, approval checkpoint files, and generated vector-store data out of version control unless sanitized test fixtures are intentionally required.
- The `.gitignore` file includes rules for local databases, ChromaDB data, Python caches, and virtual environments.
- Reviewer authentication is currently applied to the approval review and decision endpoints, not uniformly across the entire API.
- Do not expose this development application publicly without reviewing authentication, authorization, configuration, data protection, and operational security.

Ignoring a file in Git does not remove it if it was already committed. Review repository history separately if there is reason to believe a secret was committed.

## Limitations

- AI-generated root causes, diagnostic findings, and remediation suggestions are hypotheses and require human verification.
- Investigation generation depends on a running Ollama service and an available local model.
- Retrieval quality depends on the indexed knowledge, embedding configuration, and relevance of the source documents.
- The health endpoint reports the API's health response; it is not a comprehensive check of every external dependency.
- Reviewer authentication is applied to approval review and decision endpoints. Incident-management endpoints and approval-request creation do not currently have equivalent reviewer protection.
- A complete organizational approval system with automatic team-lead or agent assignment, role-specific permissions, and escalation policies is not implemented.
- Approval resume leases have a configured expiry. Long-running workflow resumption beyond that lease duration requires further consideration.
- The application is intended as a development and portfolio demonstration, not a production-ready incident-management platform.
- Production deployment, comprehensive authorization, operational monitoring, and broader integration testing require further work.

## Future Improvements

Potential enhancements include:

- Extend authentication and authorization across incident-management and approval-request creation endpoints.
- Introduce explicit role-based permissions and team-lead or agent assignment where required.
- Expand integration tests for complete incident lifecycle and approval workflows.
- Evaluate retrieval and generated findings against a representative incident dataset.
- Add structured investigation results with evidence references.
- Improve dependency health checks, operational monitoring, and production deployment configuration.
- Further strengthen recovery behavior for long-running approval workflows and expired resume leases.

## Disclaimer

AI Incident Copilot is intended for educational and portfolio demonstration purposes. It assists incident investigation but does not independently verify root causes, guarantee remediation outcomes, or replace qualified human judgment. It should not be treated as a production-ready incident-management or change-approval system.
