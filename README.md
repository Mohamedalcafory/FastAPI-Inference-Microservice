# Identity Service

A Django REST Framework-based identity and authentication service with role-based access control and audit logging.

## 🚀 Features

- User authentication and authorization
- Role-based access control (RBAC)
- Audit logging for security events
- PostgreSQL database integration
- Redis caching for performance
- Docker containerized deployment
- API versioning and documentation
- Comprehensive test coverage

## 🛠️ Prerequisites

- Docker & Docker Compose
- Python 3.11+
- PostgreSQL
- Redis

## 🏃‍♂️ Quick Start

1. Clone and setup environment:
```bash
git clone <repository-url>
cd identity-service
cp .env.example .env
```

2. Start development environment:
```bash 
make dev
```

3. Run migrations and create admin user:
```bash
make migrate
make superuser
```

## 🔧 Configuration

Configure using environment variables in `.env`:

```bash
# Django
SECRET_KEY=change-me-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database
DB_NAME=identity_db
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_URL=redis://localhost:6379/1
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/login/` | User login |
| POST | `/api/auth/register/` | User registration |
| POST | `/api/auth/refresh/` | Refresh JWT token |
| GET | `/api/users/` | List users |
| GET | `/api/roles/` | List roles |
| GET | `/api/audit-logs/` | View audit logs |

## 🛠️ Development Commands

```bash
# Start development server
make dev

# Run database migrations
make migrate

# Create superuser
make superuser

# Run tests with coverage
make test

# Run linting
make lint

# Format code
make fmt

# View logs
make logs
```

## 🧪 Testing

Run the test suite:

```bash
# Full test suite
make test

# Continuous testing
make test-watch

# Run linting checks
make lint
```

## 📦 Project Structure

```
identity-service/
├── auth/                    # Authentication views and serializers
├── users/                   # User management and roles
├── audit/                   # Audit logging functionality
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── manage.py               # Django management script
├── settings.py             # Project settings
├── urls.py                 # URL routing
└── docker-compose.yml      # Service orchestration
```

## 🔒 Security Features

- Strong password validation
- Role-based access control (RBAC)
- JWT authentication
- Comprehensive audit logging
- Request rate limiting
- Secure password hashing
- Session management

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Run tests: `make test lint`
4. Commit changes: `git commit -m "Add new feature"`
5. Push and create pull request

## 📄 License

[MIT License] - See LICENSE file for details.

---

Built with ❤️ using Django REST Framework and PostgreSQL