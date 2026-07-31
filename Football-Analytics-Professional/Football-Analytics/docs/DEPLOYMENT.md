# Deployment guide

## Local

Use `pip install -e ".[dev]"`, then run Streamlit and FastAPI separately.

## Docker

`docker compose up --build`

## Club environment

Recommended components:

- object storage for raw provider files
- scheduled ingestion and quality jobs
- curated warehouse tables
- container registry and CI/CD
- identity provider and SSO
- reverse proxy with TLS
- central logs and metrics
- model registry and approval workflow
- separate development, test and production environments

## Operational checks

- health endpoint responds
- latest matches are present
- event counts are within expected ranges
- model metadata matches the deployed binary
- dashboard definitions match the analyst glossary
- failed quality gates block publication
