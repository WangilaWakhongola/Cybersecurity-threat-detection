.PHONY: help build up down logs test lint migrate clean

help:
	@echo "Cybersecurity Threat Detection - Commands"
	@echo "=========================================="
	@echo ""
	@echo "Docker:"
	@echo "  make build              Build images"
	@echo "  make up                 Start services"
	@echo "  make down               Stop services"
	@echo "  make logs               View logs"
	@echo ""
	@echo "Development:"
	@echo "  make test               Run tests"
	@echo "  make lint               Check code quality"
	@echo "  make format             Format code"
	@echo ""
	@echo "Database:"
	@echo "  make migrate            Run migrations"
	@echo "  make db-reset           Reset database"
	@echo ""
	@echo "Setup:"
	@echo "  make setup              Initial setup"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f backend

test:
	docker-compose exec backend python manage.py test
	docker-compose exec frontend npm test

lint:
	docker-compose exec backend flake8 .
	docker-compose exec frontend npm run lint

format:
	docker-compose exec backend black .
	docker-compose exec frontend npm run format

migrate:
	docker-compose exec backend python manage.py migrate

migrations:
	docker-compose exec backend python manage.py makemigrations

db-reset:
	docker-compose exec backend python manage.py flush --no-input
	docker-compose exec backend python manage.py migrate

setup: build up migrate
	@echo "Setup complete!"
	@echo "Visit http://localhost:3000"

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	docker system prune -f

.DEFAULT_GOAL := help
