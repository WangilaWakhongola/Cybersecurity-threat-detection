# Setup Guide - Cybersecurity Threat Detection

## Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend)

## Quick Start

```bash
# Clone repository
git clone <repo-url>
cd cybersecurity-threat-detection

# Setup environment
cp .env.example .env

# Start services
docker-compose up --build

# Create superuser (new terminal)
docker-compose exec backend python manage.py createsuperuser
```

## Access Points

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:3000 |
| API | http://localhost:8000/api/ |
| Admin | http://localhost:8000/admin/ |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

## Development Setup

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Agent
```bash
cd agent
pip install -r requirements.txt
python main.py
```

## Commands

```bash
# View logs
docker-compose logs -f backend

# Run tests
docker-compose exec backend python manage.py test

# Database migrations
docker-compose exec backend python manage.py migrate

# Create superuser
docker-compose exec backend python manage.py createsuperuser
```

See full documentation in `docs/` directory.
