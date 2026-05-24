# RAG Context Engine — Phase 1 Foundation

This project contains the Phase 1 foundation setup for an enterprise FastAPI-based context engine.

## Features

- FastAPI bootstrap with application factory
- Structured logging configuration
- Pydantic settings loader with environment support
- Health endpoints for liveness and readiness
- Dockerfile and Compose support for container validation
- Pytest configuration with coverage reporting

## Local Development

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   pip install -r requirements.txt
   ```

2. Run the application locally:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

3. Verify health endpoints:

   - `GET /health/live`
   - `GET /health/ready`

## Docker Validation

Build and run the container:

```bash
docker build -t rag-context-engine .
docker run --rm -p 8000:8000 rag-context-engine
```

## Testing

```bash
pytest
```
