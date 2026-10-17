# AI Incident Copilot

An AI-powered incident investigation assistant built with Python, FastAPI, SQLite, and a locally hosted Ollama language model.

## Overview

AI Incident Copilot helps users record and manage service incidents and generate AI-assisted investigation findings. It provides structured incident context to a local language model to suggest possible root causes, diagnostic checks, and remediation steps.

The generated findings are suggestions for investigation, not verified diagnoses.

## Features

- Create and manage incident records.
- Validate required incident details and severity.
- Store incident data in SQLite.
- Generate AI-assisted investigation findings using Ollama.
- Handle unavailable Ollama services and invalid model responses.
- Test incident validation and investigation error handling.

## Technology Stack

- Python
- FastAPI
- SQLite
- Ollama
- Qwen 2.5 3B
- Pytest

## Project Structure

```text
ai-incident-copilot/
├── src/
│   ├── database.py
│   ├── investigation.py
│   ├── main.py
│   └── models.py
├── tests/
│   ├── test_incidents.py
│   └── test_investigation.py
├── data/
│   └── incidents.db
└── README.md
```

## Setup

1. Create and activate a Python virtual environment.
2. Install the required dependencies.
3. Install and start Ollama.
4. Download the configured model:

```bash
ollama pull qwen2.5:3b
```

5. Run the automated tests:

```bash
python -m pytest -v
```

## Testing

The project includes automated tests for incident validation, investigation responses, and error handling. Tests that simulate Ollama responses can run without making actual model requests.

## Limitations

- AI-generated root causes are hypotheses and require human verification.
- Investigation generation requires a running Ollama service and an available model.
- This project is a development portfolio project and is not presented as a production incident-management platform.
