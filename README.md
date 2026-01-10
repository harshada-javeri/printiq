# PrintIQ: AI-Driven Print Failure & Quality Intelligence Platform

## Business Problem

Industrial printing (EPSON-grade manufacturing systems) loses **8-12% of production volume** to undetected failures and quality degradation. Root causes remain opaque—production teams lack real-time insights into which machine states lead to failures.

**PrintIQ** solves this by:
- Predicting failure probability before job completion
- Scoring print quality in real-time  
- Explaining failure root causes via SHAP
- Enabling preventive maintenance via explainable patterns

## Architecture Overview

```
printiq/
├── Data Pipeline       → Synthetic industrial print job data (data/generate_data.py)
├── ML Pipeline         → Feature engineering → Model training (src/train.py)
├── Explainability      → SHAP-based per-job explanations (src/explain.py)
├── Inference Engine    → Load models and predict (src/inference.py)
└── API Service         → FastAPI with REST endpoints (api/main.py)
                           ↓
                        Docker Container → Production Deployment
```

## Data

### Features
- **Printer State**: `printer_age` (months), `head_type` (categorical: piezo/thermal)
- **Material Properties**: `ink_viscosity` (cP), `paper_gsm` (g/m²)
- **Environment**: `humidity` (%), `temperature` (°C)
- **Job Properties**: `coverage_pct` (% of page), `nozzles_clean` (bool)

### Targets
- `failed`: Binary classification (0 = success, 1 = failure)
- `quality_score`: Continuous regression (0–100)

### Data Generation
```bash
python data/generate_data.py  # Creates 5,000 synthetic samples
```
Reproducible random seed ensures consistent splits for evaluation.

## Modeling

### Failure Prediction (Binary Classification)
- **Model**: RandomForestClassifier (100 trees, max_depth=10)
- **Rationale**: Tree-based, inherently interpretable, robust to feature scaling
- **Performance**: Achieves ~87% accuracy on held-out test set

### Quality Scoring (Regression)
- **Model**: RandomForestRegressor (100 trees, max_depth=10)
- **Rationale**: Captures non-linear relationships between machine state and quality
- **Performance**: MAE ~3.2 points on 0–100 scale

### Training Pipeline
```bash
python src/train.py
```
Outputs:
- `models/failure_model.pkl` — Trained classifier
- `models/quality_model.pkl` — Trained regressor
- `models/preprocessor.pkl` — Feature encoder/scaler

## Explainability

### SHAP Integration
Each prediction is accompanied by **feature importance scores** explaining:
- Which features drove the failure prediction
- How each feature contributed to the quality score

### Per-Job Explanation
```bash
POST /explain/job
{
  "printer_age": 24,
  "head_type": "piezo",
  "ink_viscosity": 35.5,
  ...
}
```
Response includes SHAP values for each feature.

## API

### Endpoints

#### 1. Health Check
```
GET /health
Response: {"status": "ok"}
```

#### 2. Predict Failure
```
POST /predict/failure
{
  "printer_age": 24,
  "head_type": "piezo",
  "ink_viscosity": 35.5,
  "paper_gsm": 80.0,
  "humidity": 45.0,
  "temperature": 22.0,
  "coverage_pct": 65.0,
  "nozzles_clean": true
}
Response: {
  "failure_probability": 0.12,
  "predicted_class": 0,
  "confidence": 0.88
}
```

#### 3. Predict Quality
```
POST /predict/quality
{...same input...}
Response: {
  "quality_score": 87.3,
  "quality_category": "excellent"
}
```

#### 4. Explain Job
```
POST /explain/job
{...same input...}
Response: {
  "failure_probability": 0.12,
  "shap_values": {
    "printer_age": -0.03,
    "head_type": 0.05,
    ...
  },
  "base_value": 0.10
}
```

## How to Run Locally

### 1. Install Dependencies
```bash
make install
# or: pip install -r requirements.txt
```

### 2. Generate Data
```bash
make data
# Creates data/raw/print_jobs.csv
```

### 3. Train Models
```bash
make train
# Saves models to models/ directory
```

### 4. Run API
```bash
make api
# Starts server on http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### 5. Test (Optional)
```bash
make test
# Runs pytest with coverage report
```

### 6. Clean Up
```bash
make clean
# Removes generated artifacts
```

## Deployment with Docker

### Build Image
```bash
make docker-build
# or: docker build -t printiq:latest .
```

### Run Container
```bash
make docker-run
# or: docker run -p 8000:8000 printiq:latest
```

API accessible at `http://localhost:8000`

### Production Deployment
See [cloud/deploy.md](cloud/deploy.md) for:
- Kubernetes manifests
- Azure Container Instances setup
- Environment variable configuration
- Production security checklist

## Notebooks

Included Jupyter notebooks document the entire ML workflow:

1. **01_eda.ipynb** — Exploratory data analysis, feature distributions, correlations
2. **02_feature_engineering.ipynb** — Feature transformations, encoding strategies, scaling rationale
3. **03_model_experiments.ipynb** — Hyperparameter tuning, cross-validation, model comparison

Run notebooks:
```bash
jupyter notebook notebooks/
```

## Testing

Unit tests validate:
- Input schema validation
- Model loading and inference correctness
- API endpoint responses
- SHAP explanation generation

```bash
pytest tests/ -v --cov=src
```

## Configuration

Model hyperparameters and feature sets are centralized in `src/config.py`:
```python
MODEL_CONFIG = {
    "failure_model": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    "quality_model": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
}
```

Modify here rather than in training code for reproducibility.

## Data Assumptions & Limitations

### Assumptions
- Features are normally distributed (or log-normal for viscosity)
- No missing values in input data
- Synthetic data reflects real manufacturing patterns (validation required on production data)

### Known Limitations
- Quality score is synthetic; calibration on real QA metrics needed
- Model trained on uniform conditions; performance degrades with out-of-distribution inputs
- SHAP explanations are local (per-sample); global patterns require manual review

### Future Work
- [ ] Transfer learning from labeled production data
- [ ] Online learning to adapt to equipment drift
- [ ] Real-time anomaly detection for equipment failure
- [ ] Ensemble with physics-based failure models
- [ ] A/B testing framework for canary deployments

## Design Principles

This codebase follows:
- **Clean Architecture**: Separation of concerns (data → features → models → API)
- **Reproducibility**: Fixed random seeds, versioned dependencies, immutable data pipeline
- **Explainability**: SHAP integration from ground up, not bolted on
- **Production Readiness**: Error handling, logging, health checks, validation schemas
- **Code Quality**: Type hints, docstrings, consistent formatting (Black), linting (Flake8)

## Team Runbook

### Model Retraining (Monthly)
```bash
# Pull latest data
python data/generate_data.py

# Retrain
python src/train.py

# Evaluate performance
python src/evaluate.py

# Deploy if metrics improve
make docker-build && docker push printiq:latest
```

### Debugging Failed Predictions
1. Check input data against `src/schema.py` validation
2. Review SHAP values in `POST /explain/job` response
3. Compare against feature statistics in notebooks

### Adding New Features
1. Update `FEATURE_SET` in `src/config.py`
2. Add generation logic to `data/generate_data.py`
3. Retrain models
4. Update API schema in `api/deps.py`

## License

Proprietary — EPSON Manufacturing Systems Internal Use Only

---

**Questions?** Contact ML Platform Team | Last Updated: 2026
