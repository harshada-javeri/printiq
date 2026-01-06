.PHONY: help install data train evaluate test api docker-build docker-run clean

help:
	@echo "PrintIQ - ML Capstone Commands"
	@echo "================================"
	@echo "install          Install dependencies"
	@echo "data             Generate synthetic training data"
	@echo "train            Train failure and quality models"
	@echo "evaluate         Evaluate model performance"
	@echo "api              Run FastAPI server locally"
	@echo "test             Run test suite"
	@echo "docker-build     Build Docker image"
	@echo "docker-run       Run Docker container"
	@echo "clean            Clean generated files"

install:
	pip install -r requirements.txt

data:
	python data/generate_data.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=html

docker-build:
	docker build -t printiq:latest .

docker-run:
	docker run -p 8000:8000 printiq:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f data/raw/*.csv data/processed/*.csv
	rm -f models/*.pkl models/*.joblib
