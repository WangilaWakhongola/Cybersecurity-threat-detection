# API Documentation

## Base URL
`http://localhost:8000/api`

## Authentication
```
Authorization: Bearer <token>
```

## Endpoints

### Threats
- `GET /threats/` - List threats
- `POST /threats/` - Report threat
- `GET /threats/{id}/` - Get threat
- `PUT /threats/{id}/` - Update threat
- `DELETE /threats/{id}/` - Delete threat
- `POST /threats/{id}/mitigate/` - Take action

### Alerts
- `GET /alerts/` - List alerts
- `POST /alerts/` - Create alert
- `GET /alerts/{id}/` - Get alert
- `PUT /alerts/{id}/` - Update alert

### Agents
- `GET /agents/` - List agents
- `POST /agents/register/` - Register new agent
- `GET /agents/{id}/` - Get agent
- `POST /agents/{id}/push-rules/` - Push rules to agent

### Intelligence
- `GET /intelligence/feeds/` - Threat feeds
- `GET /intelligence/iocs/` - Indicators of compromise
- `GET /intelligence/reputation/{ip}/` - IP reputation

### Vulnerabilities
- `GET /vulnerabilities/` - List vulnerabilities
- `POST /vulnerabilities/scan/` - Start vulnerability scan
- `GET /vulnerabilities/{id}/` - Get vulnerability

## Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Server Error
