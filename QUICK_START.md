# 🔒 Quick Start - Cybersecurity Threat Detection

## 30-Second Setup

```bash
# 1. Clone
git clone <repo-url>
cd cybersecurity-threat-detection

# 2. Setup environment
cp .env.example .env

# 3. Start
docker-compose up --build

# 4. Create superuser (new terminal)
docker-compose exec backend python manage.py createsuperuser

# Done! 🎉
```

## Access Points

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| **API** | http://localhost:8000/api/ |
| **API Docs** | http://localhost:8000/api/schema/swagger/ |
| **Admin** | http://localhost:8000/admin/ |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3001 |

## Key Features

✅ **Intrusion Detection (IDS/IPS)**  
✅ **Malware Analysis**  
✅ **Vulnerability Scanning**  
✅ **Network Traffic Analysis**  
✅ **Log Anomaly Detection**  
✅ **Threat Intelligence**  
✅ **Real-time Alerts**  
✅ **Distributed Agents**  

## Essential Commands

```bash
# Start
docker-compose up -d

# Logs
docker-compose logs -f backend

# Tests
docker-compose exec backend python manage.py test

# Migrate
docker-compose exec backend python manage.py migrate

# Reset DB
docker-compose exec backend python manage.py flush --no-input
```

## Documentation

- [Setup Guide](./docs/SETUP.md)
- [API Documentation](./docs/API.md)
- [Deployment Guide](./docs/DEPLOYMENT.md)

## Support

- 📧 support@threathunter.io
- 🐛 GitHub Issues
- 📖 Full Documentation

---

**Enterprise-Grade Threat Detection** 🔒
