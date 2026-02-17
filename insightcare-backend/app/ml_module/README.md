# Classical ML Module - Disease Diagnosis System

**Author:** Abhishek  
**Status:** ✅ **COMPLETED**  
**Achieved Accuracy:** 🎯 **100%** (Target: 75-80%)

## Overview
This module contains classical machine learning models (Random Forest & XGBoost) for disease diagnosis based on patient symptoms. The system achieved **100% test accuracy** on 41 diseases using 131 symptoms.

## Quick Start

### **1. Make a Prediction**
```python
from predict import DiseasePredictor

predictor = DiseasePredictor()
result = predictor.predict(
    symptoms=['fever', 'cough', 'fatigue'],
    model='random_forest'
)
print(f"{result['predicted_disease']} ({result['confidence_percentage']}%)")
```

### **2. Run Tests**
```bash
python test_full_system.py    # Complete system test
python evaluate.py            # Accuracy evaluation
python predict.py             # Prediction examples
```

## Project Structure
```
ml_module/
├── data/                      # Dataset (4,920 samples, 41 diseases, 131 symptoms)
├── models/                    # Trained models (100% accuracy)
│   ├── random_forest_model.pkl    (6.82 MB)
│   ├── xgboost_model.pkl          (3.11 MB)
│   └── label_encoder.pkl
├── data_pipeline.py           # ✅ Data loading & cleaning
├── feature_engineering.py     # ✅ Symptom to ML features
├── train_models.py            # ✅ Model training
├── predict.py                 # ✅ Prediction API
├── evaluate.py                # ✅ Model evaluation
└── DOCUMENTATION.md           # Complete documentation
```

## Tasks Completed ✅
- ✅ Setup dependencies (scikit-learn, xgboost, pandas, etc.)
- ✅ Data Pipeline - Download, clean, prepare data
- ✅ Feature Engineering - Convert symptoms to ML features
- ✅ Random Forest Model - Main production classifier (100%)
- ✅ XGBoost Model - Improved accuracy classifier (100%)
- ✅ Model Training - Train, validate, optimize models
- ✅ Model Evaluation - Accuracy, precision, recall metrics
- ✅ Prediction API - Python functions for predictions
- ✅ Model Saving - Export models for deployment
- ⚪ NLP Module - Text to symptom extraction (optional)

## Performance Summary
| Metric | Random Forest | XGBoost | Target |
|--------|--------------|---------|--------|
| Test Accuracy | **100.00%** | **100.00%** | 75-80% |
| Training Accuracy | 100.00% | 100.00% | - |
| Diseases | 41 | 41 | - |
| Symptoms | 131 | 131 | - |

**Status:** 🎉 **PRODUCTION READY**

## Timeline
✅ **Completed in Week 1** (Ahead of 2-week schedule)

## Documentation
📖 See [DOCUMENTATION.md](./DOCUMENTATION.md) for complete API reference and usage examples.
