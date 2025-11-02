# Quantum vs Classical Performance Comparison Report

## Executive Summary

This report compares the performance of quantum and classical machine learning models 
for disease diagnosis based on symptoms.

## Performance Metrics

```
              Model               Type Accuracy Training Time Prediction Speed  Dataset Size  Feature Encoding       Status
      Random Forest Classical Ensemble     100%         ~2.5s     Fast (<0.1s) 4,920 samples Severity-weighted ✅ Production
            XGBoost Classical Ensemble     100%         ~3.0s     Fast (<0.1s) 4,920 samples Severity-weighted ✅ Production
Classical SVM (RBF)   Classical Kernel     ~95%         ~0.5s     Fast (<0.1s)    50 samples  Standard scaling   ✅ Baseline
   Quantum SVM (ZZ)     Quantum Kernel    ~53%*       ~3.4s**     Slow (~3s)**  35 samples**    Quantum ZZ map  ⚪ Demo only
```

**Notes:**
- `*` QSVM tested on small demo dataset only
- `**` Quantum kernel computation is computationally expensive

## Key Findings


1. **Classical Models Dominate Overall:**
   • Random Forest: 100% accuracy on full dataset (4,920 samples)
   • XGBoost: 100% accuracy on full dataset
   • Both models are production-ready and extremely fast

2. **QSVM Performance:**
   • 53.33% accuracy on small demo (35 training samples, 3 diseases)
   • Requires significant computational resources
   • Quantum kernel computation takes 3+ seconds for small datasets
   • Not practical for production use with current implementation

3. **Why Classical Outperforms:**
   • Dataset characteristics: symptoms have clear patterns
   • Tree-based models excel at categorical/structured data
   • Large dataset (4,920 samples) favors classical approaches
   • No quantum advantage for this specific problem structure

4. **When Quantum Might Help:**
   • Very high-dimensional feature spaces
   • Complex non-linear decision boundaries
   • Small training sets with complex patterns
   • Problems with known quantum advantage

5. **Quantum Integration Status:**
   • ✅ Quantum circuits implemented (angle, pauli, ZZ encoding)
   • ✅ Quantum kernel computation working
   • ✅ QSVM classifier functional
   • ✅ Integration with classical pipeline complete
   • ⚪ Production QSVM not recommended for this use case

6. **Recommendations:**
   • **Production:** Use Random Forest + XGBoost ensemble
   • **Research:** Quantum encoding ready for experimentation
   • **Future:** Monitor quantum hardware advances
   • **Hybrid:** Current hybrid predictor offers best of both worlds
    

## Technical Specifications


Classical Models:
├─ Random Forest
│  ├─ Trees: 100
│  ├─ Max depth: None (full trees)
│  ├─ Features: 131 symptoms
│  ├─ Classes: 41 diseases
│  └─ Training: 3,936 samples (80% split)
│
├─ XGBoost
│  ├─ Estimators: 100
│  ├─ Learning rate: 0.1
│  ├─ Max depth: 6
│  └─ Objective: multi:softmax

Quantum Components:
├─ Feature Encoding
│  ├─ Original features: 131 symptoms
│  ├─ Quantum features: 10 qubits
│  ├─ Encoding types: Angle, Pauli, ZZ
│  └─ Circuit depth: 1-15 (depending on type)
│
├─ Quantum Kernel
│  ├─ Type: Fidelity-based
│  ├─ Backend: Qiskit Aer Simulator
│  ├─ Feature map: ZZFeatureMap
│  └─ Reps: 1-2

Hybrid System:
├─ Primary: Random Forest + XGBoost
├─ Quantum: Feature encoding available
├─ Integration: HybridPredictor class
└─ Status: Production-ready (classical), Demo (quantum)
    

## Conclusions

1. **For Production Use:** Classical models (RF + XGBoost) are optimal
2. **Quantum Advantage:** Not observed for this specific problem
3. **Future Research:** Quantum encoding infrastructure is ready for experimentation
4. **Recommendation:** Continue with classical models, revisit quantum as hardware improves

## Generated: 2025-11-02 16:22:35
