.PHONY: help dev worker all test lint fmt train export-onnx bench api-test clean

# Default target
help:
	@echo "FastAPI Inference Service - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev          - Start app + Redis with docker-compose"
	@echo "  worker       - Start Celery worker"
	@echo "  all          - Start all services (app + worker + redis + prometheus + grafana)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test         - Run pytest with coverage"
	@echo "  lint         - Run ruff, black, mypy"
	@echo "  fmt          - Format code with black and ruff"
	@echo ""
	@echo "Model Management:"
	@echo "  train        - Train sklearn model to models/model.joblib"
	@echo "  export-onnx  - Export ONNX model"
	@echo ""
	@echo "Performance:"
	@echo "  bench        - Run k6 load test"
	@echo "  api-test     - Test API endpoints with curl"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean up containers and volumes"
	@echo "  logs         - View application logs"
	@echo "  shell        - Access app container shell"

# Development commands
dev:
	@echo "Starting development environment (app + redis)..."
	docker-compose up app redis -d
	@echo "Services started. Access:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Docs: http://localhost:8000/docs"
	@echo "  - Health: http://localhost:8000/health"

worker:
	@echo "Starting Celery worker..."
	docker-compose up worker -d
	@echo "Worker started. Check logs with: make logs"

all:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "All services started. Access:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Docs: http://localhost:8000/docs"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"

# Testing commands
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html
	@echo "Coverage report generated in htmlcov/"

lint:
	@echo "Running linting..."
	ruff check .
	black --check .
	mypy app/

fmt:
	@echo "Formatting code..."
	black .
	ruff check --fix .

# Model management
train:
	@echo "Training sklearn model..."
	python scripts/train_model.py
	@echo "Model training completed!"

export-onnx:
	@echo "Exporting model to ONNX..."
	python scripts/export_onnx.py
	@echo "ONNX export completed!"

# Performance testing
bench:
	@echo "Running load test with k6..."
	@if command -v k6 >/dev/null 2>&1; then \
		k6 run scripts/load_test.js; \
	else \
		echo "k6 not found. Install from https://k6.io/docs/getting-started/installation/"; \
		echo "Or use: make api-test for basic API testing"; \
	fi

api-test:
	@echo "Testing API endpoints..."
	@echo "Testing health endpoint..."
	curl -s http://localhost:8000/health | jq .
	@echo ""
	@echo "Testing model info..."
	curl -s http://localhost:8000/model/info | jq .
	@echo ""
	@echo "Testing single prediction..."
	curl -s -X POST http://localhost:8000/api/v1/predict \
		-H "Content-Type: application/json" \
		-d '{"inputs": [5.1, 3.5, 1.4, 0.2]}' | jq .
	@echo ""
	@echo "Testing batch prediction..."
	curl -s -X POST http://localhost:8000/api/v1/predict/batch \
		-H "Content-Type: application/json" \
		-d '{"batches": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]}' | jq .
	@echo ""
	@echo "Testing async prediction..."
	RESPONSE=$$(curl -s -X POST http://localhost:8000/api/v1/predict/async \
		-H "Content-Type: application/json" \
		-d '{"inputs": [5.1, 3.5, 1.4, 0.2]}'); \
	echo $$RESPONSE | jq .; \
	TASK_ID=$$(echo $$RESPONSE | jq -r '.task_id'); \
	echo "Task ID: $$TASK_ID"; \
	sleep 2; \
	curl -s http://localhost:8000/api/v1/predict/async/$$TASK_ID | jq .

# Utility commands
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	docker system prune -f
	@echo "Cleanup completed!"

logs:
	@echo "Viewing application logs..."
	docker-compose logs -f app

shell:
	@echo "Accessing app container shell..."
	docker-compose exec app bash

# Additional utility commands
redis-cli:
	@echo "Accessing Redis CLI..."
	docker-compose exec redis redis-cli

queue-status:
	@echo "Checking Celery queue status..."
	docker-compose exec app celery -A app.services.tasks.celery_app inspect active

queue-purge:
	@echo "Purging Celery queue..."
	docker-compose exec app celery -A app.services.tasks.celery_app purge

metrics:
	@echo "Viewing Prometheus metrics..."
	curl -s http://localhost:8000/metrics | head -20

# Development setup
setup:
	@echo "Setting up development environment..."
	cp env.example .env
	@echo "Environment file created. Edit .env as needed."
	@echo "Run 'make train' to create a model, then 'make dev' to start services."
