# FinSight-OCR Docker Setup Guide

Docker containerization setup for the FinSight-OCR Korean financial document OCR system.

## 📦 Container Architecture

```
┌─────────────────────────────────────────────────┐
│                Load Balancer/Proxy              │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│              Frontend (nginx)                   │
│          React + Vite Production Build          │
│                 Port: 80                        │
└─────────────────┬───────────────────────────────┘
                  │ API Proxy
┌─────────────────▼───────────────────────────────┐
│            Backend (FastAPI)                    │
│      PyTorch OCR Pipeline + API Server          │
│                Port: 8000                       │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Production Deployment

```bash
# 1. Clone the repository
git clone <repository-url>
cd FinSight-OCR

# 2. Ensure model files are in place
# Copy your trained models to backend/modelrun/saved_models/
# Required files:
# - dbnet_best.pth
# - korean_recognition_best.pth

# 3. Build and start all services
docker-compose up -d

# 4. Check service status
docker-compose ps

# 5. View logs
docker-compose logs -f

# Access the application at http://localhost
```

### Development Setup

```bash
# Start development environment with hot reload
docker-compose -f docker-compose.dev.yml up -d

# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

## 📁 Required Model Files

Before running the containers, ensure these model files exist:

```
backend/modelrun/saved_models/
├── dbnet_best.pth              # Text detection model
├── korean_recognition_best.pth # Text recognition model
└── configs/
    └── recognition/
        └── korean_char_map.txt # Character mapping file
```

## 🛠️ Container Services

### Backend Service (finsight-ocr-backend)

- **Base Image**: python:3.9-slim
- **Framework**: FastAPI + uvicorn
- **Dependencies**: PyTorch 2.1.2, OpenCV, OCR libraries
- **Port**: 8000
- **Health Check**: `/api/v1/health`
- **Resource Limits**: 4GB RAM, 2GB reserved

### Frontend Service (finsight-ocr-frontend)

- **Base Image**: nginx:1.25-alpine
- **Build**: Node.js 18 + Vite production build
- **Port**: 80
- **Features**: Gzip compression, security headers, API proxy
- **Health Check**: HTTP 200 response

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Backend Configuration
BACKEND_PORT=8000
WORKERS=1

# Frontend Configuration
FRONTEND_PORT=80

# API Configuration
API_URL=http://backend:8000

# Resource Limits
BACKEND_MEMORY_LIMIT=4g
BACKEND_MEMORY_RESERVATION=2g
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./backend/modelrun/saved_models` | `/app/modelrun/saved_models` | Model weights (read-only) |
| `./backend/modelrun/configs` | `/app/modelrun/configs` | Model configurations |
| `./backend/document` | `/app/document` | Document type configs |
| `./backend/output` | `/app/output` | OCR results output |

## 🔍 Monitoring and Debugging

### Health Checks

```bash
# Check backend health
curl http://localhost:8000/api/v1/health

# Check frontend
curl http://localhost/

# Check container health
docker-compose ps
```

### Logs

```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Check last 100 lines
docker-compose logs --tail=100 backend
```

### Resource Monitoring

```bash
# Monitor resource usage
docker stats

# Check specific container
docker stats finsight-ocr-backend
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Model Files Not Found
```bash
# Check if model files exist
ls -la backend/modelrun/saved_models/

# Expected files:
# dbnet_best.pth
# korean_recognition_best.pth
```

#### 2. Out of Memory Errors
```bash
# Check available system resources
docker system df
docker system prune

# Adjust memory limits in docker-compose.yml
```

#### 3. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :80
netstat -tulpn | grep :8000

# Modify ports in docker-compose.yml if needed
```

#### 4. CORS Issues
- Verify nginx proxy configuration in `frontend/nginx.conf`
- Check backend CORS settings in `backend/main.py`

### Debug Mode

```bash
# Run containers in foreground for debugging
docker-compose up

# Access container shell
docker exec -it finsight-ocr-backend bash
docker exec -it finsight-ocr-frontend sh

# Check container logs in real-time
docker logs -f finsight-ocr-backend
```

## 🔄 Maintenance

### Updates

```bash
# Pull latest images
docker-compose pull

# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose restart
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Clean up unused images
docker image prune -a

# Full system cleanup
docker system prune -a
```

### Backup

```bash
# Backup model files
tar -czf models-backup.tar.gz backend/modelrun/saved_models/

# Backup output data
tar -czf output-backup.tar.gz backend/output/
```

## 🚀 Production Deployment

### With Reverse Proxy (Recommended)

```nginx
# nginx.conf for production reverse proxy
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/HTTPS Setup

```yaml
# docker-compose.yml with SSL
services:
  frontend:
    ports:
      - "443:443"
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
```

### Scaling

```bash
# Scale backend instances
docker-compose up -d --scale backend=3

# Load balancing with nginx upstream
```

## 📊 Performance Optimization

### Resource Allocation

- **Backend**: Minimum 2GB RAM, recommended 4GB+
- **Frontend**: Minimum 128MB RAM
- **Storage**: 10GB+ for models and temporary files

### Optimization Tips

1. **Use GPU if available**: Add GPU support to backend container
2. **Enable model caching**: Pre-load models on container start
3. **Optimize image sizes**: Use multi-stage builds
4. **Monitor resource usage**: Set up logging and monitoring

## 🔐 Security Considerations

- Run containers as non-root users
- Use security headers in nginx
- Regularly update base images
- Scan images for vulnerabilities
- Use secrets management for sensitive data
- Enable firewall rules for exposed ports

## 📝 Additional Commands

```bash
# Build specific service
docker-compose build backend

# Run one-time command
docker-compose run backend python -c "print('Hello')"

# Export container as image
docker commit finsight-ocr-backend my-registry/finsight-ocr:latest

# Push to registry
docker push my-registry/finsight-ocr:latest
```