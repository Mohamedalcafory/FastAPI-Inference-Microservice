# Create project structure for FastAPI Inference Microservice
project_structure = {
    "fastapi-inference-service": {
        "app": {
            "main.py": "FastAPI application entry point",
            "api": ["__init__.py", "routes.py", "deps.py"],
            "core": ["__init__.py", "config.py", "logging.py", "middleware.py", "metrics.py"],
            "models": ["__init__.py", "loader.py", "predict.py", "onnx_utils.py"],
            "schemas": ["__init__.py", "request.py", "response.py"],
            "services": ["__init__.py", "tasks.py", "preprocessing.py", "postprocessing.py"]
        },
        "worker": ["__init__.py", "celery_app.py"],
        "tests": ["__init__.py", "test_sync_inference.py", "test_batch_inference.py", "test_async_tasks.py", "conftest.py"],
        "infra": {
            "docker-compose.yml": "Development environment",
            "grafana": ["dashboards"]
        },
        "scripts": ["train_model.py", "export_onnx.py", "load_test.js"],
        "models": ["README.md"],
        "root_files": ["pyproject.toml", "Dockerfile", "Dockerfile.worker", ".env.example", "Makefile", "README.md", ".gitignore"]
    }
}

def print_structure(structure, indent=0):
    for key, value in structure.items():
        if isinstance(value, dict):
            print("  " * indent + key + "/")
            print_structure(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + key + "/")
            for item in value:
                if isinstance(item, str):
                    print("  " * (indent + 1) + item)
        else:
            print("  " * indent + key + " - " + value)

print("FastAPI Inference Service Structure:")
print_structure(project_structure)