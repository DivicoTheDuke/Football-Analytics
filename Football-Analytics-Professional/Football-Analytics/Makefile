.PHONY: install test lint demo validate train dashboard api report clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=football_analytics --cov-report=term-missing

lint:
	ruff check .

demo:
	football-analytics generate-demo --matches 28 --seed 42

validate:
	football-analytics validate-data

train:
	football-analytics train-xg

dashboard:
	streamlit run app.py

api:
	uvicorn api:app --reload

report:
	football-analytics build-report --match-id M0001

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
