"""
Advanced Model Training - Hyperparameter Optimization & Enhanced Training
Author: AI Developer
Description: Further train and optimize hybrid models with advanced techniques
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer
)
from xgboost import XGBClassifier
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineering


class AdvancedModelTrainer:
    """
    Advanced training with hyperparameter optimization and ensemble methods
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the advanced trainer
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # Models
        self.rf_model = None
        self.xgb_model = None
        self.ensemble_model = None
        
        # Best parameters
        self.best_rf_params = None
        self.best_xgb_params = None
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Feature engineering
        self.feature_engineering = None
        
        print(f"\n{'='*80}")
        print("ADVANCED MODEL TRAINER - HYBRID MODEL OPTIMIZATION")
        print(f"{'='*80}")
        print(f"✓ Test size: {test_size * 100}%")
        print(f"✓ Random state: {random_state}")
        print(f"✓ Training mode: Hyperparameter Optimization + Ensemble")
    
    def prepare_data(self, use_severity: bool = True, augment: bool = True):
        """
        Load and prepare data for training with optional augmentation
        
        Args:
            use_severity: Whether to use severity-weighted features
            augment: Whether to apply data augmentation
        """
        print(f"\n{'='*80}")
        print("DATA PREPARATION WITH AUGMENTATION")
        print(f"{'='*80}")
        
        # Initialize pipeline
        pipeline = DataPipeline()
        pipeline.load_data()
        df = pipeline.prepare_data()
        pipeline.get_unique_symptoms(df)
        pipeline.get_unique_diseases(df)
        pipeline.create_severity_dict()
        
        # Initialize feature engineering
        self.feature_engineering = FeatureEngineering(pipeline)
        self.feature_engineering.df = df
        
        # Prepare features and labels
        X, y = self.feature_engineering.prepare_training_data(use_severity)
        
        print(f"\n📊 Original dataset:")
        print(f"  • Total samples: {X.shape[0]}")
        print(f"  • Features: {X.shape[1]}")
        print(f"  • Classes: {len(np.unique(y))}")
        
        # Apply data augmentation if requested
        if augment:
            X, y = self._augment_data(X, y)
            print(f"\n✨ After augmentation:")
            print(f"  • Total samples: {X.shape[0]}")
            print(f"  • Features: {X.shape[1]}")
        
        # Split data with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  • Training set: {self.X_train.shape[0]} samples")
        print(f"  • Test set: {self.X_test.shape[0]} samples")
        print(f"  • Train/Test ratio: {self.X_train.shape[0]/self.X_test.shape[0]:.2f}:1")
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment training data with noise and variations
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Augmented X and y
        """
        print(f"\n🔄 Applying data augmentation...")
        
        X_augmented = [X]
        y_augmented = [y]
        
        # Add small random noise (5% augmentation)
        noise_samples = int(0.05 * len(X))
        for i in range(noise_samples):
            idx = np.random.randint(0, len(X))
            sample = X[idx].copy()
            
            # Add small gaussian noise to non-zero features
            noise = np.random.normal(0, 0.1, sample.shape)
            noise = noise * (sample != 0)  # Only add noise to active symptoms
            augmented_sample = np.clip(sample + noise, 0, 10)  # Keep in valid range
            
            X_augmented.append(augmented_sample.reshape(1, -1))
            y_augmented.append(y[idx:idx+1])
        
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        
        print(f"  ✓ Added {len(X_final) - len(X)} augmented samples")
        
        return X_final, y_final
    
    def optimize_random_forest(self, quick_mode: bool = False) -> Dict:
        """
        Optimize Random Forest with GridSearchCV
        
        Args:
            quick_mode: Use smaller parameter grid for faster training
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*80}")
        print("RANDOM FOREST HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        
        if quick_mode:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2'],
                'bootstrap': [True],
                'class_weight': ['balanced', None]
            }
            cv = 3
            print("⚡ Quick mode: Reduced parameter grid")
        else:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            cv = 5
            print("🔍 Full mode: Comprehensive parameter search")
        
        print(f"\n📊 Grid search configuration:")
        print(f"  • Parameters to test: {len(param_grid)} dimensions")
        print(f"  • Cross-validation folds: {cv}")
        print(f"  • Total combinations: ~{np.prod([len(v) for v in param_grid.values()])}")
        
        # Initialize base model
        rf_base = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        # Grid search
        print(f"\n🔄 Starting grid search (this may take several minutes)...")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Grid search complete! Time: {elapsed_time/60:.2f} minutes")
        
        # Best model
        self.rf_model = grid_search.best_estimator_
        self.best_rf_params = grid_search.best_params_
        
        print(f"\n🏆 Best parameters found:")
        for param, value in self.best_rf_params.items():
            print(f"  • {param}: {value}")
        
        # Evaluate
        results = self._evaluate_model(self.rf_model, "Random Forest (Optimized)")
        results['best_params'] = self.best_rf_params
        results['cv_best_score'] = grid_search.best_score_
        results['optimization_time'] = elapsed_time
        
        print(f"\n📈 Best CV Score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
        
        return results
    
    def optimize_xgboost(self, quick_mode: bool = False) -> Dict:
        """
        Optimize XGBoost with RandomizedSearchCV
        
        Args:
            quick_mode: Use fewer iterations for faster training
            
        Returns:
            Dictionary with optimization results
        """
        print(f"\n{'='*80}")
        print("XGBOOST HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 4, 5, 6, 7, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0]
        }
        
        if quick_mode:
            n_iter = 20
            cv = 3
            print("⚡ Quick mode: 20 iterations")
        else:
            n_iter = 50
            cv = 5
            print("🔍 Full mode: 50 iterations")
        
        print(f"\n📊 Random search configuration:")
        print(f"  • Parameters to sample: {len(param_distributions)} dimensions")
        print(f"  • Random iterations: {n_iter}")
        print(f"  • Cross-validation folds: {cv}")
        
        # Initialize base model
        xgb_base = XGBClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        # Random search
        print(f"\n🔄 Starting random search (this may take several minutes)...")
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='accuracy',
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Random search complete! Time: {elapsed_time/60:.2f} minutes")
        
        # Best model
        self.xgb_model = random_search.best_estimator_
        self.best_xgb_params = random_search.best_params_
        
        print(f"\n🏆 Best parameters found:")
        for param, value in self.best_xgb_params.items():
            print(f"  • {param}: {value}")
        
        # Evaluate
        results = self._evaluate_model(self.xgb_model, "XGBoost (Optimized)")
        results['best_params'] = self.best_xgb_params
        results['cv_best_score'] = random_search.best_score_
        results['optimization_time'] = elapsed_time
        
        print(f"\n📈 Best CV Score: {random_search.best_score_:.4f} ({random_search.best_score_*100:.2f}%)")
        
        return results
    
    def create_ensemble(self) -> Dict:
        """
        Create ensemble model combining RF and XGBoost
        
        Returns:
            Dictionary with ensemble results
        """
        print(f"\n{'='*80}")
        print("CREATING ENSEMBLE MODEL")
        print(f"{'='*80}")
        
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Train individual models first!")
        
        # Create voting classifier (soft voting for probability averaging)
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.rf_model),
                ('xgb', self.xgb_model)
            ],
            voting='soft',  # Use probability averaging
            weights=[1, 1],  # Equal weights
            n_jobs=-1
        )
        
        print(f"\n🔄 Training ensemble model...")
        start_time = time.time()
        self.ensemble_model.fit(self.X_train, self.y_train)
        elapsed_time = time.time() - start_time
        
        print(f"✓ Ensemble training complete! Time: {elapsed_time:.2f} seconds")
        
        # Evaluate
        results = self._evaluate_model(self.ensemble_model, "Ensemble (RF + XGBoost)")
        results['training_time'] = elapsed_time
        
        return results
    
    def _evaluate_model(self, model, model_name: str) -> Dict:
        """
        Evaluate model performance with detailed metrics
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*80}")
        
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        train_precision = precision_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Overfitting check
        accuracy_drop = train_accuracy - test_accuracy
        
        # Print results
        print(f"\n📈 Training Set Performance:")
        print(f"  • Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"  • Precision: {train_precision:.4f}")
        print(f"  • Recall:    {train_recall:.4f}")
        print(f"  • F1-Score:  {train_f1:.4f}")
        
        print(f"\n📊 Test Set Performance:")
        print(f"  • Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  • Precision: {test_precision:.4f}")
        print(f"  • Recall:    {test_recall:.4f}")
        print(f"  • F1-Score:  {test_f1:.4f}")
        
        print(f"\n🔍 Generalization:")
        print(f"  • Accuracy drop: {accuracy_drop*100:.2f}%")
        if accuracy_drop > 0.05:
            print(f"  ⚠️  Warning: Possible overfitting (>5% drop)")
        elif accuracy_drop < -0.02:
            print(f"  ⚠️  Warning: Unusual - test better than train")
        else:
            print(f"  ✅ Good generalization")
        
        # Cross-validation
        print(f"\n🔄 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            n_jobs=-1
        )
        print(f"  • CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"  • Mean: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        print(f"  • Std:  {cv_scores.std():.4f} (±{cv_scores.std()*100:.2f}%)")
        
        results = {
            'model_name': model_name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'accuracy_drop': accuracy_drop,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        return results
    
    def analyze_feature_importance(self):
        """Analyze and display feature importance from models"""
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        
        if self.rf_model is None:
            print("⚠️  No Random Forest model available")
            return
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        symptoms_list = self.feature_engineering.pipeline.symptoms_list
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        print(f"\n🔝 Top 20 Most Important Symptoms:")
        print(f"{'Rank':<6} {'Symptom':<30} {'Importance':<12} {'Bar'}")
        print(f"{'-'*80}")
        
        for i in range(min(20, len(indices))):
            idx = indices[i]
            symptom = symptoms_list[idx]
            importance = importances[idx]
            bar = '█' * int(importance * 100)
            print(f"{i+1:<6} {symptom:<30} {importance:.6f}    {bar}")
        
        # Save importance to file
        importance_data = pd.DataFrame({
            'symptom': [symptoms_list[i] for i in indices],
            'importance': importances[indices]
        })
        
        importance_path = Path(__file__).parent / "models" / "feature_importance.csv"
        importance_data.to_csv(importance_path, index=False)
        print(f"\n✓ Feature importance saved to: {importance_path}")
    
    def save_models(self, models_dir: str = None, prefix: str = "optimized"):
        """
        Save optimized models to disk
        
        Args:
            models_dir: Directory to save models
            prefix: Prefix for model filenames
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        else:
            models_dir = Path(models_dir)
        
        models_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*80}")
        print("SAVING OPTIMIZED MODELS")
        print(f"{'='*80}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Random Forest
        if self.rf_model:
            rf_path = models_dir / f"{prefix}_random_forest_{timestamp}.pkl"
            with open(rf_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            print(f"✓ Random Forest saved: {rf_path}")
            
            # Also save as current best
            rf_best_path = models_dir / "random_forest_model.pkl"
            with open(rf_best_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            print(f"✓ Updated best RF model: {rf_best_path}")
        
        # Save XGBoost
        if self.xgb_model:
            xgb_path = models_dir / f"{prefix}_xgboost_{timestamp}.pkl"
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgb_model, f)
            print(f"✓ XGBoost saved: {xgb_path}")
            
            # Also save as current best
            xgb_best_path = models_dir / "xgboost_model.pkl"
            with open(xgb_best_path, 'wb') as f:
                pickle.dump(self.xgb_model, f)
            print(f"✓ Updated best XGB model: {xgb_best_path}")
        
        # Save Ensemble
        if self.ensemble_model:
            ensemble_path = models_dir / f"{prefix}_ensemble_{timestamp}.pkl"
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            print(f"✓ Ensemble saved: {ensemble_path}")
        
        # Save parameters
        if self.best_rf_params or self.best_xgb_params:
            params_path = models_dir / f"{prefix}_best_params_{timestamp}.pkl"
            with open(params_path, 'wb') as f:
                pickle.dump({
                    'rf_params': self.best_rf_params,
                    'xgb_params': self.best_xgb_params,
                    'timestamp': timestamp
                }, f)
            print(f"✓ Best parameters saved: {params_path}")
        
        # Save feature engineering
        if self.feature_engineering:
            self.feature_engineering.save_encoders()
            print(f"✓ Feature encoders saved")


def main():
    """Advanced training pipeline"""
    print("\n" + "="*80)
    print("🚀 ADVANCED HYBRID MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Ask user for training mode
    print("\n📋 Select training mode:")
    print("  1. Quick mode (faster, less thorough)")
    print("  2. Full mode (slower, comprehensive optimization)")
    
    mode = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    quick_mode = (mode == "1")
    
    if quick_mode:
        print("\n⚡ QUICK MODE SELECTED")
    else:
        print("\n🔍 FULL MODE SELECTED - This will take longer!")
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(test_size=0.2, random_state=42)
    
    # Prepare data
    trainer.prepare_data(use_severity=True, augment=True)
    
    # Optimize Random Forest
    rf_results = trainer.optimize_random_forest(quick_mode=quick_mode)
    
    # Optimize XGBoost
    xgb_results = trainer.optimize_xgboost(quick_mode=quick_mode)
    
    # Create Ensemble
    ensemble_results = trainer.create_ensemble()
    
    # Analyze feature importance
    trainer.analyze_feature_importance()
    
    # Save models
    trainer.save_models(prefix="optimized")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ ADVANCED TRAINING COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print(f"\n  Random Forest (Optimized):")
    print(f"    • Test Accuracy: {rf_results['test_accuracy']*100:.2f}%")
    print(f"    • CV Accuracy:   {rf_results['cv_mean']*100:.2f}% (±{rf_results['cv_std']*100:.2f}%)")
    print(f"    • Optimization time: {rf_results['optimization_time']/60:.2f} min")
    
    print(f"\n  XGBoost (Optimized):")
    print(f"    • Test Accuracy: {xgb_results['test_accuracy']*100:.2f}%")
    print(f"    • CV Accuracy:   {xgb_results['cv_mean']*100:.2f}% (±{xgb_results['cv_std']*100:.2f}%)")
    print(f"    • Optimization time: {xgb_results['optimization_time']/60:.2f} min")
    
    print(f"\n  Ensemble Model:")
    print(f"    • Test Accuracy: {ensemble_results['test_accuracy']*100:.2f}%")
    print(f"    • CV Accuracy:   {ensemble_results['cv_mean']*100:.2f}% (±{ensemble_results['cv_std']*100:.2f}%)")
    
    # Determine best model
    best_acc = max(
        rf_results['test_accuracy'],
        xgb_results['test_accuracy'],
        ensemble_results['test_accuracy']
    )
    
    if ensemble_results['test_accuracy'] == best_acc:
        best_model = "Ensemble"
    elif rf_results['test_accuracy'] == best_acc:
        best_model = "Random Forest"
    else:
        best_model = "XGBoost"
    
    print(f"\n🏆 Best Model: {best_model} ({best_acc*100:.2f}%)")
    print(f"\n✨ Models are now optimized and ready for production!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
