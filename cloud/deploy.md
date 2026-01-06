# PrintIQ Deployment Guide

## Overview

PrintIQ is containerized and ready for deployment to cloud environments. This guide covers deployment options from development to production.

---

## 1. Local Development

### Run Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data/generate_data.py

# Train models
python src/train.py

# Evaluate models
python src/evaluate.py

# Start API server (with auto-reload)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access API at: http://localhost:8000
Swagger UI: http://localhost:8000/docs

### Run with Docker Locally

```bash
# Build image
docker build -t printiq:latest .

# Run container
docker run -p 8000:8000 printiq:latest

# Test endpoint
curl -X POST http://localhost:8000/api/v1/predict/failure \
  -H "Content-Type: application/json" \
  -d '{
    "printer_age": 24,
    "head_type": "piezo",
    "ink_viscosity": 35.5,
    "paper_gsm": 80.0,
    "humidity": 45.0,
    "temperature": 22.0,
    "coverage_pct": 65.0,
    "nozzles_clean": true
  }'
```

---

## 2. Azure Container Instances (ACI)

### Prerequisites

```bash
az login
az account set --subscription <SUBSCRIPTION_ID>
az group create --name printiq-rg --location eastus
```

### Push to Azure Container Registry (ACR)

```bash
# Create registry
az acr create --resource-group printiq-rg --name printiqacr --sku Basic

# Login to ACR
az acr login --name printiqacr

# Build and push
az acr build --registry printiqacr --image printiq:latest .

# Or manually build and push
docker build -t printiqacr.azurecr.io/printiq:latest .
docker push printiqacr.azurecr.io/printiq:latest
```

### Deploy to ACI

```bash
az container create \
  --resource-group printiq-rg \
  --name printiq-api \
  --image printiqacr.azurecr.io/printiq:latest \
  --cpu 1 \
  --memory 1.5 \
  --ports 8000 \
  --registry-login-server printiqacr.azurecr.io \
  --registry-username <USERNAME> \
  --registry-password <PASSWORD> \
  --environment-variables LOG_LEVEL=INFO \
  --restart-policy OnFailure

# Get container URL
az container show --resource-group printiq-rg --name printiq-api \
  --query ipAddress.fqdn
```

### Monitor Logs

```bash
az container logs --resource-group printiq-rg --name printiq-api
```

---

## 3. Azure App Service

### Create App Service Plan

```bash
az appservice plan create \
  --name printiq-plan \
  --resource-group printiq-rg \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group printiq-rg \
  --plan printiq-plan \
  --name printiq-api \
  --deployment-container-image-name printiqacr.azurecr.io/printiq:latest

# Configure container
az webapp config container set \
  --name printiq-api \
  --resource-group printiq-rg \
  --docker-custom-image-name printiqacr.azurecr.io/printiq:latest \
  --docker-registry-server-url https://printiqacr.azurecr.io \
  --docker-registry-server-user <USERNAME> \
  --docker-registry-server-password <PASSWORD>

# Enable continuous deployment
az webapp deployment container config \
  --name printiq-api \
  --resource-group printiq-rg \
  --enable-continuous-deployment
```

---

## 4. Kubernetes (AKS)

### Create AKS Cluster

```bash
az aks create \
  --resource-group printiq-rg \
  --name printiq-aks \
  --node-count 2 \
  --vm-set-type VirtualMachineScaleSets \
  --load-balancer-sku standard \
  --enable-managed-identity

# Get credentials
az aks get-credentials --resource-group printiq-rg --name printiq-aks
```

### Deploy with Helm

Create `printiq-helm/Chart.yaml`:

```yaml
apiVersion: v2
name: printiq
description: Print Failure & Quality Intelligence Platform
type: application
version: 1.0.0
appVersion: "1.0"
```

Create `printiq-helm/values.yaml`:

```yaml
replicaCount: 2

image:
  repository: printiqacr.azurecr.io/printiq
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
  targetCPUUtilizationPercentage: 80
```

Deploy:

```bash
helm install printiq ./printiq-helm --namespace printiq --create-namespace
```

---

## 5. Environment Variables

### Required

```bash
# API Configuration (optional - has defaults)
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# Logging
LOG_LEVEL=INFO
```

### Optional (Production)

```bash
# API Security
API_KEY=<your-secret-key>
CORS_ORIGINS=["https://yourapp.com"]

# Monitoring
NEW_RELIC_LICENSE_KEY=<key>
SENTRY_DSN=<dsn>
```

---

## 6. Production Checklist

- [ ] Models are trained and serialized in `models/` directory
- [ ] Dependencies pinned in `requirements.txt`
- [ ] Dockerfile tested locally
- [ ] API health checks pass (`GET /health`)
- [ ] CORS configured for your domain
- [ ] API keys/secrets in Azure Key Vault
- [ ] Logging configured (stdout to container logs)
- [ ] Resource limits set (CPU, memory)
- [ ] Auto-scaling enabled
- [ ] Health probes configured
- [ ] Error handling and validation in place
- [ ] API documentation accessible at `/docs`
- [ ] Monitoring and alerts configured
- [ ] Backup strategy for model artifacts

---

## 7. Scaling Considerations

### Vertical Scaling
- Increase CPU/memory per instance
- Recommended: 1-2 CPU, 1-2GB RAM per instance

### Horizontal Scaling
- Run multiple replicas behind load balancer
- Recommended: 2-5 instances based on traffic
- Use Kubernetes HPA or App Service auto-scale

### Performance

**Inference time (per prediction):**
- Failure: ~5ms
- Quality: ~5ms
- Explanation: ~50-100ms (SHAP computation)

**Throughput:**
- Single instance: ~200 predictions/second
- 4 instances: ~800 predictions/second

---

## 8. Troubleshooting

### Models not found
```
Error: Model artifacts not found
Solution: Run python src/train.py before deploying
```

### Out of memory
```
Error: Memory limit exceeded
Solution: Increase memory in deployment config or reduce batch size
```

### Slow predictions
```
Solution: 
- Increase CPU allocation
- Enable explanation caching
- Use inference batching
```

### API timeout
```
Solution:
- Increase timeout in client code
- Reduce explanation computation for API calls
```

---

## 9. Monitoring & Logging

### Container Logs
```bash
# Azure Container Instances
az container logs --resource-group printiq-rg --name printiq-api

# App Service
az webapp log tail --name printiq-api --resource-group printiq-rg

# Kubernetes
kubectl logs -f deployment/printiq -n printiq
```

### Metrics to Monitor
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx responses)
- Prediction count
- Model accuracy (track with production data)
- GPU/CPU utilization
- Memory usage

### Health Checks
```bash
# Readiness
GET /health

# Liveness
GET /api/v1/predict/failure (with valid input)
```

---

## 10. Updating Models

### Rolling Update
```bash
# Train new models locally
python src/train.py

# Rebuild Docker image
docker build -t printiqacr.azurecr.io/printiq:v1.1 .

# Push to registry
docker push printiqacr.azurecr.io/printiq:v1.1

# Deploy new version
az webapp config container set \
  --name printiq-api \
  --resource-group printiq-rg \
  --docker-custom-image-name printiqacr.azurecr.io/printiq:v1.1
```

### Canary Deployment (Kubernetes)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: printiq-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: printiq
      version: v1.1
  template:
    metadata:
      labels:
        app: printiq
        version: v1.1
    spec:
      containers:
      - name: printiq
        image: printiqacr.azurecr.io/printiq:v1.1
        ports:
        - containerPort: 8000
```

---

## Support & Questions

For issues or questions, refer to:
- README.md - Project overview
- Notebooks - Data & modeling details
- src/config.py - Configuration options
- api/main.py - API implementation

---

**Last Updated**: 2024
**Status**: Production Ready
