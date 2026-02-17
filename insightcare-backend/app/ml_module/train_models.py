"""
Model Training - Train Random Forest and XGBoost Models
Author: Abhishek
Description: Train, validate, and optimize ML models for disease diagnosis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import pickle
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from data_pipeline import DataPipeline
from feature_engineering import FeatureEngineering


class ModelTrainer:
    """
    Train and evaluate Random Forest and XGBoost models
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the model trainer
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # Models
        self.rf_model = None
        self.xgb_model = None
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Feature engineering
        self.feature_engineering = None
        
        print(f"\n✓ ModelTrainer initialized")
        print(f"  • Test size: {test_size * 100}%")
        print(f"  • Random state: {random_state}")
    
    def prepare_data(self, use_severity: bool = True):
        """
        Load and prepare data for training
        
        Args:
            use_severity: Whether to use severity-weighted features
        """
        print(f"\n{'='*70}")
        print("PREPARING DATA FOR TRAINING")
        print(f"{'='*70}")
        
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
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\n✓ Data split complete:")
        print(f"  • Training set: {self.X_train.shape[0]} samples")
        print(f"  • Test set: {self.X_test.shape[0]} samples")
        print(f"  • Features: {self.X_train.shape[1]}")
    
    def train_random_forest(self, 
                           n_estimators: int = 100,
                           max_depth: int = None,
                           min_samples_split: int = 2,
                           min_samples_leaf: int = 1,
                           verbose: bool = True) -> Dict:
        """
        Train Random Forest classifier
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            verbose: Print training progress
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*70}")
        print("TRAINING RANDOM FOREST MODEL")
        print(f"{'='*70}")
        
        # Initialize model
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        print(f"\n📊 Model parameters:")
        print(f"  • n_estimators: {n_estimators}")
        print(f"  • max_depth: {max_depth}")
        print(f"  • min_samples_split: {min_samples_split}")
        print(f"  • min_samples_leaf: {min_samples_leaf}")
        
        # Train model
        print(f"\n🔄 Training...")
        self.rf_model.fit(self.X_train, self.y_train)
        print(f"✓ Training complete!")
        
        # Evaluate
        results = self._evaluate_model(self.rf_model, "Random Forest")
        
        return results
    
    def train_xgboost(self,
                     n_estimators: int = 100,
                     max_depth: int = 6,
                     learning_rate: float = 0.1,
                     subsample: float = 0.8,
                     colsample_bytree: float = 0.8,
                     verbose: bool = True) -> Dict:
        """
        Train XGBoost classifier
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            verbose: Print training progress
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*70}")
        print("TRAINING XGBOOST MODEL")
        print(f"{'='*70}")
        
        # Initialize model
        self.xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        print(f"\n📊 Model parameters:")
        print(f"  • n_estimators: {n_estimators}")
        print(f"  • max_depth: {max_depth}")
        print(f"  • learning_rate: {learning_rate}")
        print(f"  • subsample: {subsample}")
        print(f"  • colsample_bytree: {colsample_bytree}")
        
        # Train model
        print(f"\n🔄 Training...")
        self.xgb_model.fit(self.X_train, self.y_train)
        print(f"✓ Training complete!")
        
        # Evaluate
        results = self._evaluate_model(self.xgb_model, "XGBoost")
        
        return results
    
    def _evaluate_model(self, model, model_name: str) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*70}")
        
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
        
        # Cross-validation
        print(f"\n🔄 Cross-validation (5-fold)...")
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, n_jobs=-1)
        print(f"  • CV Scores: {cv_scores}")
        print(f"  • Mean CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        print(f"  • Std CV Accuracy:  {cv_scores.std():.4f}")
        
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
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return results
    
    def compare_models(self, rf_results: Dict, xgb_results: Dict):
        """
        Compare Random Forest and XGBoost performance
        
        Args:
            rf_results: Random Forest results
            xgb_results: XGBoost results
        """
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        
        print(f"\n{'Metric':<20} {'Random Forest':<20} {'XGBoost':<20} {'Winner'}")
        print(f"{'-'*70}")
        
        metrics = [
            ('Test Accuracy', 'test_accuracy'),
            ('Test Precision', 'test_precision'),
            ('Test Recall', 'test_recall'),
            ('Test F1-Score', 'test_f1'),
            ('CV Mean Accuracy', 'cv_mean')
        ]
        
        for metric_name, metric_key in metrics:
            rf_val = rf_results[metric_key]
            xgb_val = xgb_results[metric_key]
            winner = "🏆 RF" if rf_val > xgb_val else "🏆 XGB" if xgb_val > rf_val else "🤝 Tie"
            
            print(f"{metric_name:<20} {rf_val:>6.4f} ({rf_val*100:>5.2f}%)  {xgb_val:>6.4f} ({xgb_val*100:>5.2f}%)  {winner}")
    
    def save_models(self, models_dir: str = None):
        """
        Save trained models to disk
        
        Args:
            models_dir: Directory to save models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        else:
            models_dir = Path(models_dir)
        
        models_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print("SAVING MODELS")
        print(f"{'='*70}")
        
        # Save Random Forest
        if self.rf_model:
            rf_path = models_dir / "random_forest_model.pkl"
            with open(rf_path, 'wb') as f:
                pickle.dump(self.rf_model, f)
            print(f"✓ Random Forest saved: {rf_path}")
        
        # Save XGBoost
        if self.xgb_model:
            xgb_path = models_dir / "xgboost_model.pkl"
            with open(xgb_path, 'wb') as f:
                pickle.dump(self.xgb_model, f)
            print(f"✓ XGBoost saved: {xgb_path}")
        
        # Save feature engineering
        if self.feature_engineering:
            self.feature_engineering.save_encoders()
    
    def load_models(self, models_dir: str = None):
        """
        Load trained models from disk
        
        Args:
            models_dir: Directory containing saved models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        else:
            models_dir = Path(models_dir)
        
        print(f"\n{'='*70}")
        print("LOADING MODELS")
        print(f"{'='*70}")
        
        # Load Random Forest
        rf_path = models_dir / "random_forest_model.pkl"
        if rf_path.exists():
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            print(f"✓ Random Forest loaded: {rf_path}")
        
        # Load XGBoost
        xgb_path = models_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            with open(xgb_path, 'rb') as f:
                self.xgb_model = pickle.load(f)
            print(f"✓ XGBoost loaded: {xgb_path}")


def main():
    """Train and evaluate models"""
    print("="*70)
    print("DISEASE DIAGNOSIS MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    
    # Prepare data
    trainer.prepare_data(use_severity=True)
    
    # Train Random Forest
    rf_results = trainer.train_random_forest(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    # Train XGBoost
    xgb_results = trainer.train_xgboost(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Compare models
    trainer.compare_models(rf_results, xgb_results)
    
    # Save models
    trainer.save_models()
    
    print(f"\n{'='*70}")
    print("✅ MODEL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n🎯 Target: 75-80% accuracy")
    print(f"📊 Random Forest Test Accuracy: {rf_results['test_accuracy']*100:.2f}%")
    print(f"📊 XGBoost Test Accuracy: {xgb_results['test_accuracy']*100:.2f}%")
    
    if rf_results['test_accuracy'] >= 0.75 or xgb_results['test_accuracy'] >= 0.75:
        print(f"\n🎉 SUCCESS! Target accuracy achieved!")
    else:
        print(f"\n⚠️  Models need improvement. Consider hyperparameter tuning.")


if __name__ == "__main__":
    main()
