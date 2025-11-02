# Disease Diagnosis ML Models

AI-powered disease prediction from symptoms using Random Forest and XGBoost

## 🚀 Quick Start

### 1. Download Dataset
Download from: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

Extract to: `ml_models/data/`

### 2. Install Dependencies
```bash
cd insightcare-backend/ml_models
pip install -r requirements.txt
```

### 3. Train Models
```bash
python train_model.py
```

This will:
- Train Random Forest (target: 75%+ accuracy)
- Train XGBoost (target: 80%+ accuracy)
- Save best model to `models/best_model.pkl`
- Training time: 5-10 minutes

### 4. Test Model
```bash
python test_model.py
```

Interactive symptom checker - enter symptoms and get predictions!

## 📁 Files

```
ml_models/
├── data/
│   └── dataset.csv              # Disease-symptom dataset (download first)
├── models/
│   ├── best_model.pkl           # Best performing model
│   ├── random_forest_model.pkl  # Random Forest model
│   ├── xgboost_model.pkl        # XGBoost model
│   ├── symptom_encoder.pkl      # Symptom encoder (MultiLabelBinarizer)
│   ├── label_encoder.pkl        # Label encoder (for XGBoost)
│   └── metadata.pkl             # Model metadata & stats
├── train_model.py               # Complete training pipeline
├── disease_classifier.py        # Production classifier class
├── test_model.py               # Interactive testing
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎯 Usage in Code

```python
from disease_classifier import DiseaseClassifier

# Initialize
classifier = DiseaseClassifier()
classifier.load_model()

# Predict
result = classifier.predict(['fever', 'cough', 'headache'])

print(f"Disease: {result['disease']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 3: {result['top_3']}")
```

## 🔌 FastAPI Integration

```python
# app/api/ml_diagnosis.py
from fastapi import APIRouter
from ml_models.disease_classifier import DiseaseClassifier

router = APIRouter(prefix="/api/ml", tags=["ML Diagnosis"])
classifier = DiseaseClassifier()
classifier.load_model()

@router.post("/diagnose")
async def diagnose(symptoms: list[str]):
    result = classifier.predict(symptoms)
    return result
```

## 📊 Model Performance

| Model | Accuracy | Training Time | Model Size |
|-------|----------|---------------|------------|
| Random Forest | 75-85% | 2-5 min | 20-50 MB |
| XGBoost | 80-88% | 3-7 min | 10-30 MB |

## 📋 Dataset Info

- **Source**: Kaggle Disease-Symptom Dataset
- **Samples**: 4,920 patients
- **Diseases**: 41 diseases
- **Symptoms**: 17 symptoms per patient
- **Format**: CSV with Disease + Symptom_1 to Symptom_17 columns

## 🧪 Testing

### Run All Tests
```bash
python test_model.py
```

### Example Test Cases
```python
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
# Expected: Fungal infection

symptoms = ['high_fever', 'headache', 'nausea']
# Expected: Malaria, Dengue, or similar

symptoms = ['stomach_pain', 'acidity', 'vomiting']
# Expected: Gastroenteritis, GERD, etc.
```

## 🚀 Deployment (Railway)

### Option 1: Include model in git
```bash
git add models/best_model.pkl models/symptom_encoder.pkl
git commit -m "Add trained ML models"
git push
```

### Option 2: Load on startup
Add to `app/main.py`:
```python
from ml_models.disease_classifier import DiseaseClassifier

classifier = DiseaseClassifier()

@app.on_event("startup")
async def load_models():
    classifier.load_model()
    print("✅ ML models loaded")
```

## 🔧 Troubleshooting

### Model accuracy < 75%?
1. Check dataset is complete (4,920 rows)
2. Try hyperparameter tuning
3. Add more training data (Synthea)

### Prediction errors?
1. Ensure symptoms are lowercase
2. Use underscores: `high_fever` not `high fever`
3. Check available symptoms: `classifier.get_available_symptoms()`

### Out of memory?
1. Reduce `n_estimators` to 50
2. Reduce `max_depth` to 10
3. Use smaller model (XGBoost instead of RF)

## 📚 Next Steps

- [x] Train baseline models (75%+ accuracy)
- [x] Create production classifier class
- [ ] Integrate with FastAPI backend
- [ ] Add symptom autocomplete
- [ ] Deploy to Railway
- [ ] Add confidence thresholds
- [ ] Implement A/B testing

## 📖 Documentation

- **Dataset**: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- **scikit-learn**: https://scikit-learn.org/stable/
- **XGBoost**: https://xgboost.readthedocs.io/

---

**Training Date**: Run `train_model.py` to see  
**Best Accuracy**: Check `models/metadata.pkl`  
**Developer**: Yash (InsightCare Team)
