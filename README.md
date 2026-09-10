# Multimodal AI Crop-Stress Detection System

> **Early Crop-Stress Detection using Multimodal AI (Leaf Images + Environmental Sensors)**

A production-quality, locally runnable, end-to-end software prototype for detecting crop stress from leaf images and environmental sensor data.

---

## 🎯 Project Overview

This system implements **three comparative models** as required by the research experiment:

| Model | Modality | Description |
|-------|----------|-------------|
| **Model A** | Image Only | Leaf image → CNN → Visual features → Stress prediction |
| **Model B** | Environmental Only | Sensor data → ML model → Environmental features → Stress prediction |
| **Model C** | Multimodal | Image + Sensors → Feature fusion → Combined prediction |

**Stress Categories:** Healthy, Mild Stress, Moderate Stress, Severe Stress

---

## 🏗️ Architecture

```
project-root/
├── config/                 # Configuration files
├── data/
│   ├── raw/               # Original dataset
│   ├── processed/         # Cleaned & validated data
│   ├── train/             # Training split
│   ├── validation/        # Validation split
│   └── test/              # Test split (held-out)
├── models/
│   ├── image_only/        # Model A checkpoints
│   ├── sensor_only/       # Model B checkpoints
│   └── multimodal/        # Model C checkpoints
├── ml/
│   ├── preprocessing/     # Image & sensor preprocessing
│   ├── models/            # Model definitions
│   ├── training/          # Training pipelines
│   ├── evaluation/        # Evaluation & comparison
│   └── explainability/    # Grad-CAM, SHAP
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── api/           # API routes
│   │   ├── core/          # Config, security
│   │   ├── db/            # Database models
│   │   ├── schemas/       # Pydantic schemas
│   │   └── services/      # Business logic
├── frontend/              # React + Tailwind frontend
│   ├── src/
│   └── public/
├── scripts/               # Utility scripts
├── tests/                 # Unit, integration, e2e tests
├── notebooks/             # Jupyter notebooks
├── logs/                  # Application logs
└── docs/                  # Documentation
```

---

## ⚙️ Prerequisites

- **Python 3.10+** (tested on 3.14.0)
- **Node.js 18+** (tested on 24.20.0)
- **RAM:** 8GB+ recommended (16GB+ for training)
- **OS:** Windows 10/11, Linux, or macOS
- **GPU:** Optional (CUDA 11.8+ for faster training)

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
# Navigate to project directory
cd "C:\Users\Kartik Khadria\OneDrive\Attachments\Documents\Default Project"

# Create Python virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Linux/macOS)
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Python Dependencies

```bash
# Install core dependencies (CPU-only PyTorch for Windows)
pip install -r requirements.txt

# If you have CUDA GPU, install GPU version instead:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional for local dev)
# Notepad .env
```

### 5. Prepare Dataset

**⚠️ CRITICAL: This system requires a paired multimodal dataset.**

You need a dataset with **matched records**:
```
image_id | image_path | crop | soil_moisture | temperature | humidity | rainfall | light_intensity | stress_label
```

See [Dataset Preparation](#dataset-preparation) below for details.

### 6. Run the Complete Pipeline

```bash
# 1. Validate & preprocess dataset
python scripts/prepare_dataset.py

# 2. Train all three models
python scripts/train_all.py

# 3. Evaluate & compare models
python scripts/evaluate_all.py

# 4. Start backend API
python -m backend.app.main

# 5. Start frontend (in new terminal)
cd frontend && npm run dev
```

### 7. Access the Application

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/api/health

---

## 📊 Dataset Preparation

### Required Format

The system expects a **CSV manifest file** with paired image-sensor-label data:

```csv
image_id,image_path,crop,soil_moisture,temperature,humidity,rainfall,light_intensity,stress_label
img_001,images/wheat_001.jpg,wheat,45.2,24.5,65.0,12.3,850,Healthy
img_002,images/rice_002.jpg,rice,78.5,28.1,82.0,5.2,1200,Mild Stress
img_003,images/maize_003.jpg,maize,22.1,35.2,45.0,0.0,1800,Severe Stress
```

### Supported Datasets

| Dataset | Images | Sensors | Labels | Status |
|---------|--------|---------|--------|--------|
| PlantVillage | ✅ | ❌ | ✅ | Image-only |
| CropStress Multimodal | ✅ | ✅ | ✅ | **Ideal** |
| Custom collected | ✅ | ✅ | ✅ | **Recommended** |

### If No Paired Dataset Exists

The system **explicitly supports synthetic/prototype data** with clear labeling:

1. Place images in `data/raw/images/`
2. Create `data/raw/manifest.csv` with environmental values
3. Run validation: `python scripts/validate_dataset.py`
4. All synthetic data is marked with `source: "synthetic"` in metadata

**Never mix synthetic and real data silently.**

### Dataset Validation

```bash
# Validate dataset integrity
python scripts/validate_dataset.py --manifest data/raw/manifest.csv --images data/raw/images

# Output: validation_report.json with:
# - Missing/corrupted images
# - Invalid sensor values
# - Class distribution
# - Train/val/test split verification
```

---

## 🧪 Training Pipeline

### Train Individual Models

```bash
# Model A: Image Only
python -m ml.training.train_image_only --config config.yaml

# Model B: Environmental Only
python -m ml.training.train_sensor_only --config config.yaml

# Model C: Multimodal Fusion
python -m ml.training.train_multimodal --config config.yaml
```

### Train All Models (Recommended)

```bash
python scripts/train_all.py --config config.yaml
```

**Training Outputs:**
- `models/{model_type}/best_model.pth` (or `.pkl` for sklearn)
- `models/{model_type}/preprocessor.pkl` (fitted scalers/encoders)
- `models/{model_type}/metadata.json` (metrics, config, class mapping)
- `models/checkpoints/` (epoch checkpoints)

### Training Monitoring

```bash
# View training logs
tail -f logs/training.log

# TensorBoard (if enabled)
tensorboard --logdir models/checkpoints/
```

---

## 📈 Evaluation & Comparison

```bash
# Evaluate all models on test set
python scripts/evaluate_all.py --config config.yaml

# Generate comparison report
python scripts/compare_models.py --config config.yaml
```

**Outputs:**
- `docs/evaluation_plots/confusion_matrices.png`
- `docs/evaluation_plots/metrics_comparison.png`
- `docs/evaluation_plots/per_class_metrics.png`
- `docs/model_comparison_report.json`
- `docs/model_comparison_report.md`

---

## 🔧 Backend API

### Start Server

```bash
python -m backend.app.main
# or
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Multimodal prediction |
| POST | `/api/predict/image` | Image-only prediction |
| POST | `/api/predict/sensor` | Sensor-only prediction |
| GET | `/api/predictions` | Prediction history |
| GET | `/api/predictions/{id}` | Single prediction |
| GET | `/api/models/comparison` | Model comparison metrics |
| GET | `/api/analytics` | Analytics dashboard data |
| POST | `/api/validate-input` | Validate input before prediction |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "image=@leaf.jpg" \
  -F "soil_moisture=45.2" \
  -F "temperature=24.5" \
  -F "humidity=65.0" \
  -F "rainfall=12.3" \
  -F "light_intensity=850" \
  -F "crop=wheat"
```

### Example Response

```json
{
  "prediction_id": "pred_abc123",
  "timestamp": "2026-09-09T10:30:00Z",
  "prediction": "Moderate Stress",
  "confidence": 0.87,
  "risk_level": "moderate",
  "probabilities": {
    "Healthy": 0.05,
    "Mild Stress": 0.12,
    "Moderate Stress": 0.87,
    "Severe Stress": 0.06
  },
  "contributing_factors": [
    {"factor": "low_soil_moisture", "impact": 0.34, "description": "Soil moisture (45%) below optimal"},
    {"factor": "high_temperature", "impact": 0.28, "description": "Temperature (24.5°C) above optimal range"},
    {"factor": "visual_wilting", "impact": 0.25, "description": "Leaf edges show curling/discoloration"}
  ],
  "recommendations": [
    "Check irrigation system - soil moisture is below 50%",
    "Monitor temperature - consider shade cloth during peak hours",
    "Re-scan leaf in 24 hours to track progression",
    "Consider soil moisture sensor calibration"
  ],
  "explanations": {
    "image_gradcam": "data:image/png;base64,...",
    "sensor_shap_values": {"soil_moisture": 0.34, "temperature": 0.28, ...}
  },
  "model_version": "multimodal_v1.0.0",
  "processing_time_ms": 245
}
```

---

## 🌐 Frontend

### Development

```bash
cd frontend
npm run dev
```

### Production Build

```bash
cd frontend
npm run build
# Output in frontend/build/
```

### Screens

1. **Home** - System overview, start analysis
2. **Analysis Input** - Image upload + sensor form
3. **Processing** - Real-time pipeline visualization
4. **Prediction Results** - Prediction, confidence, factors, recommendations
5. **Model Comparison** - Actual metrics from evaluation
6. **Analytics** - Feature contributions, history, confusion matrices

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/e2e/           # End-to-end tests

# With coverage
pytest --cov=ml --cov=backend --cov-report=html
```

### Test Categories

- **Data:** Validation, preprocessing, splitting
- **ML:** Model loading, prediction shapes, class mapping
- **Backend:** Health, prediction, validation, error responses
- **Frontend:** Rendering, form validation, API failure states
- **Integration:** Frontend → Backend → Model → Database → Response

---

## 🐳 Docker (Optional)

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Services:**
- `backend` - FastAPI on port 8000
- `frontend` - Nginx + React on port 3000
- `db` - SQLite volume (persistent)

---

## ⚙️ Configuration

All settings in `config.yaml` and `.env`:

| Category | File | Description |
|----------|------|-------------|
| Model architecture | `config.yaml` | Backbones, fusion type, dimensions |
| Training hyperparams | `config.yaml` | LR, batch size, epochs, schedulers |
| Data paths | `config.yaml` / `.env` | Dataset locations, output dirs |
| API settings | `config.yaml` / `.env` | Host, port, CORS, DB URL |
| Explainability | `config.yaml` | Grad-CAM layer, SHAP settings |

---

## 📝 Logging

Logs written to `logs/cropstress.log` with rotation (10MB, 5 files).

```bash
# View recent logs
tail -f logs/cropstress.log

# Filter errors
grep ERROR logs/cropstress.log
```

---

## 🔬 Research Integrity

This system **strictly distinguishes**:

| Label | Meaning |
|-------|---------|
| **REAL EXPERIMENTAL RESULT** | From actual trained models on real data |
| **DEMO** | UI demonstration with placeholder data |
| **SIMULATED DATA** | Synthetic data for prototype testing |
| **FUTURE WORK** | Planned but not implemented |

**No fake metrics, no hardcoded accuracies, no fabricated predictions.**

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Run tests: `pytest`
4. Run linting: `ruff check . && black --check .`
5. Submit PR

---

## 📞 Support

For issues, check:
1. `logs/cropstress.log` for backend errors
2. Browser console for frontend errors
3. `docs/TROUBLESHOOTING.md` for common problems

---

## 🎓 Academic Use

This system is designed for:
- College project exhibitions
- Technical demonstrations
- Research prototypes
- Viva voce presentations

**Citation:**
```bibtex
@software{cropstress2026,
  title={Multimodal AI-Based Early Crop-Stress Detection and Decision Support System},
  author={Kartik Khadria},
  year={2026},
  version={1.0.0}
}
```