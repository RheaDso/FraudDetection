.PHONY: setup data pipeline test lint api app smoke

setup:
	uv sync

data:
	uv run python -m fraud.pipeline check-data

pipeline:
	uv run python -m fraud.pipeline run

test:
	uv run pytest --cov --cov-report=term-missing --cov-report=json:build/coverage.json --junitxml=build/test-results.xml

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

api:
	uv run uvicorn fraud.api.main:app --port 8000

app:
	uv run streamlit run app/investigator.py

smoke:
	uv run python -m fraud.pipeline smoke
