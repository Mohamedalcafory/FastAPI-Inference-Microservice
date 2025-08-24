# FastAPI ML Inference Service

Production-ready FastAPI microservice for machine learning inference with async job processing, Prometheus monitoring, and ONNX optimization support.

## 🚀 Features

### Core Capabilities
- **Sync Inference**: Single and batch predictions with <200ms P95 latency
- **Async Jobs**: Background processing with Celery + Redis
- **Model Management**: Hot-swappable sklearn/ONNX models with warmup
- **Monitoring**: Comprehensive Prometheus metrics + Grafana dashboards
- **Autoscaling Ready**: Docker deployment with health checks

### Performance & Reliability
- **Cold/Warm Start Handling**: Automatic model warmup on startup
- **Request Validation**: Pydantic schemas with size limits
- **Error Handling**: Structured error responses with correlation IDs
- **Graceful Degradation**: Circuit breakers and timeouts

### Developer Experience
- **OpenAPI Docs**: Auto-generated API documentation
- **Type Safety**: Full mypy compliance
- **Testing**: 85%+ coverage with performance regression tests
- **CI/CD Ready**: GitHub Actions with quality gates

## 📊 Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │────│   FastAPI    │────│   Model     │
│  Requests   │    │     App      │    │  Registry   │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │             │
              ┌─────────┐   ┌──────────┐
              │  Redis  │   │  Celery  │
              │ Broker  │   │ Workers  │
              └─────────┘   └──────────┘
                   │
              ┌─────────┐   ┌──────────┐
              │Prometheus│   │ Grafana  │
              │Metrics  │   │Dashboard │
              └─────────┘   └──────────┘
```

## 🛠 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- k6 (for load testing)

### 1. Clone and Setup
```bash
git clone <repo-url>
cd fastapi-inference-service
cp .env.example .env
```

### 2. Start Development Environment
```bash
# Start app + Redis
make dev

# Or start all services (includes Prometheus/Grafana)
make all
```

### 3. Train Model and Test
```bash
# Train a simple sklearn model
make train

# Test the API
make api-test
```

### 4. Access Services
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### Sync Inference
```bash
# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [5.1, 3.5, 1.4, 0.2]}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"batches": [[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]]}'
```

### Async Jobs
```bash
# Submit async job
curl -X POST http://localhost:8000/api/v1/predict/async \
  -H "Content-Type: application/json" \
  -d '{"inputs": [5.1, 3.5, 1.4, 0.2]}'

# Check status
curl http://localhost:8000/api/v1/predict/async/{task_id}
```

### Monitoring
```bash
# Metrics
curl http://localhost:8000/metrics

# Model info
curl http://localhost:8000/model/info
```

## ⚡ Performance

### Benchmarks (Local)
- **P95 Latency**: <200ms (sync inference)
- **Throughput**: 500+ RPS (batch size 10)
- **Cold Start**: <2s with warmup
- **Memory Usage**: <512MB baseline

### Load Testing
```bash
# Run k6 load test
make bench

# Performance regression test
make bench-local
```

## 🔧 Configuration

### Environment Variables
```bash
# Model Configuration
MODEL_BACKEND=sklearn          # sklearn|onnx
MODEL_PATH=models/model.joblib
ONNX_PATH=models/model.onnx
WARMUP_ENABLED=true

# Performance
MAX_BATCH_SIZE=100
INFERENCE_TIMEOUT=30.0
UVICORN_WORKERS=4

# Redis/Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

### Model Backends
Switch between sklearn and ONNX:
```bash
# Use sklearn (default)
export MODEL_BACKEND=sklearn

# Use ONNX (requires export)
make export-onnx
export MODEL_BACKEND=onnx
```

## 📈 Monitoring & Observability

### Prometheus Metrics
- `inference_requests_total` - Request counts by endpoint
- `model_inference_duration_seconds` - Inference latency
- `celery_queue_depth` - Async job queue size
- `model_load_duration_seconds` - Model loading time

### Structured Logging
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "event": "prediction_completed",
  "request_id": "req_123",
  "inference_duration_ms": 45.2,
  "backend": "sklearn"
}
```

### Grafana Dashboards
- **Service Overview**: Request rates, latency, errors
- **Model Performance**: Inference metrics, queue depth
- **System Health**: Memory, CPU, cache hit rates

## 🧪 Testing

### Run Tests
```bash
# Full test suite with coverage
make test

# Performance tests
make bench-local

# Linting and formatting
make lint
make fmt
```

### Test Coverage
- Unit tests: Model loading, prediction logic
- Integration tests: API endpoints, async tasks
- Performance tests: Latency regression checks
- Target: 85%+ coverage

## 🚀 Deployment

### Docker Production
```bash
# Build production image
docker build -t inference-service .

# Run with resource limits
docker run -d \
  --name inference-service \
  -p 8000:8000 \
  --memory=1g \
  --cpus=2 \
  -e MODEL_BACKEND=onnx \
  inference-service
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: inference-service:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
```

## 🔄 Development Workflow

### Daily Development
```bash
# Start development environment
make dev

# Make changes, then test
make test
make lint

# Performance check
make bench-local
```

### Model Updates
```bash
# Retrain model
make train

# Export to ONNX
make export-onnx

# Compare performance
python scripts/benchmark_backends.py
```

## 📝 API Schema

### Request/Response Models
```python
# Single prediction
class PredictionRequest(BaseModel):
    inputs: List[float]
    model_version: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: Union[float, List[float]]
    model_version: str
    backend: str
    inference_duration_ms: float
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test: `make test lint`
4. Commit changes: `git commit -m "Add new feature"`
5. Push and create PR

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Check model file exists
ls -la models/
make train  # Retrain if missing
```

**Redis connection errors:**
```bash
# Check Redis status
make redis-cli
docker-compose logs redis
```

**High latency:**
```bash
# Check model backend
curl http://localhost:8000/model/info

# Profile with different batch sizes
make bench
```

**Queue backed up:**
```bash
# Check queue status
make queue-status

# Purge if needed
make queue-purge
```

### Debug Commands
```bash
# Access app container
make shell

# View logs
make logs

# Check metrics
make metrics

# Test API manually
make api-test
```

---
