#!/usr/bin/env python3
"""
Test Random Forest Model with RF-Selected Features

This script trains and evaluates a Random Forest model using only the top 40 features
identified by the Random Forest feature importance analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class RFSelectedFeaturesClassifier:
    """
    Train a classifier using Random Forest selected features.
    """
    
    def __init__(self, 
                 data_path: str = 'datasets/ddos.parquet',
                 rf_features_path: str = 'feature_analysis/top_40_features.json',
                 output_dir: str = 'rf_selected_model/'):
        """
        Initialize the classifier.
        
        Args:
            data_path: Path to the dataset
            rf_features_path: Path to Random Forest selected features
            output_dir: Directory to save model and results
        """
        self.data_path = data_path
        self.rf_features_path = rf_features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.selected_features = None
        self.model = None
        self.label_encoder = None
    
    def load_rf_features(self) -> list:
        """Load the Random Forest selected features."""
        print(f"Loading RF-selected features from {self.rf_features_path}...")
        
        with open(self.rf_features_path, 'r') as f:
            data = json.load(f)
            features = data['features']
        
        print(f"✅ Loaded {len(features)} RF-selected features")
        return features
    
    def load_and_prepare_data(self) -> tuple:
        """Load dataset and prepare features using RF-selected features."""
        print("Loading and preparing dataset...")
        
        # Load data
        self.df = pd.read_parquet(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        
        # Load RF features
        self.selected_features = self.load_rf_features()
        
        # Check feature availability
        missing_features = [f for f in self.selected_features if f not in self.df.columns]
        if missing_features:
            print(f"⚠️  Warning: {len(missing_features)} features not found in dataset")
            print(f"Missing features: {missing_features[:5]}...")
            self.selected_features = [f for f in self.selected_features if f in self.df.columns]
        
        print(f"Using {len(self.selected_features)} available features")
        
        # Prepare features and target
        X = self.df[self.selected_features].copy()
        y = self.df['label'].copy()
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            X = X.fillna(X.median())
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features prepared: {X.shape}")
        print(f"Target classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def train_and_evaluate(self, X, y, test_size=0.2, random_state=42):
        """Train Random Forest model and evaluate performance."""
        print(f"\nTraining Random Forest with {len(self.selected_features)} RF-selected features...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Training completed! 🎉")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_predictions': y_pred,
            'test_actual': y_test,
            'feature_names': self.selected_features
        }
    
    def plot_results(self, results, save_plot=True):
        """Create visualization of results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Random Forest Model - RF Selected Features', fontsize=16)
        
        # Confusion Matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Feature Importance (Top 20)
        feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_20 = feature_importance.head(20)
        axes[0, 1].barh(range(len(top_20)), top_20['importance'])
        axes[0, 1].set_yticks(range(len(top_20)))
        axes[0, 1].set_yticklabels(top_20['feature'])
        axes[0, 1].set_title('Top 20 Feature Importance')
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].invert_yaxis()
        
        # Class Distribution
        class_counts = pd.Series(self.df['label'].value_counts())
        axes[1, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Dataset Class Distribution')
        
        # Performance by Class
        report = results['classification_report']
        classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[c]['f1-score'] for c in classes]
        
        axes[1, 1].bar(classes, f1_scores, color=['red', 'green', 'orange'])
        axes[1, 1].set_title('F1-Score by Class')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.output_dir / 'rf_selected_features_results.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to: {plot_path}")
        
        plt.show()
    
    def save_model_and_results(self, results):
        """Save the trained model and results."""
        print(f"\nSaving model and results to {self.output_dir}...")
        
        # Save model
        model_path = self.output_dir / 'rf_classifier_rf_selected.joblib'
        joblib.dump(self.model, model_path)
        
        # Save label encoder
        encoder_path = self.output_dir / 'label_encoder_rf_selected.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save features used
        features_path = self.output_dir / 'features_rf_selected.json'
        features_data = {
            'features': self.selected_features,
            'n_features': len(self.selected_features),
            'method': 'random_forest_selected',
            'source': 'Random Forest feature importance analysis'
        }
        with open(features_path, 'w') as f:
            json.dump(features_data, f, indent=2)
        
        # Save results
        results_path = self.output_dir / 'results_rf_selected.json'
        results_summary = {
            'method': 'random_forest_selected',
            'accuracy': results['accuracy'],
            'n_features': len(self.selected_features),
            'classification_report': results['classification_report'],
            'feature_list': self.selected_features
        }
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"✅ Model and results saved:")
        print(f"   - Model: {model_path}")
        print(f"   - Encoder: {encoder_path}")
        print(f"   - Features: {features_path}")
        print(f"   - Results: {results_path}")
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        print("🚀 TRAINING MODEL WITH RF-SELECTED FEATURES")
        print("=" * 60)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Train and evaluate
        results = self.train_and_evaluate(X, y)
        
        # Create visualizations
        self.plot_results(results)
        
        # Save everything
        self.save_model_and_results(results)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Final accuracy: {results['accuracy']:.4f}")
        print(f"🔝 Used {len(self.selected_features)} RF-selected features")
        
        return results


def compare_with_existing_methods():
    """Compare RF-selected features performance with existing methods."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON WITH EXISTING METHODS")
    print("="*60)
    
    # Load results from existing methods
    existing_results = {}
    saved_models_dir = Path('saved_models')
    
    for results_file in saved_models_dir.glob('results_*.json'):
        method = results_file.stem.replace('results_', '')
        if not method.endswith('_quick_test'):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    existing_results[method] = data['accuracy']
            except:
                continue
    
    # Load RF-selected results
    rf_results_path = Path('rf_selected_model/results_rf_selected.json')
    if rf_results_path.exists():
        with open(rf_results_path, 'r') as f:
            rf_data = json.load(f)
            existing_results['rf_selected'] = rf_data['accuracy']
    
    # Create comparison table
    if existing_results:
        print("\nMethod Comparison:")
        print("-" * 40)
        sorted_results = sorted(existing_results.items(), key=lambda x: x[1], reverse=True)
        
        for i, (method, accuracy) in enumerate(sorted_results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            print(f"{rank_emoji} {method:<20} {accuracy:.4f}")


def main():
    """Main execution function."""
    # Train model with RF-selected features
    classifier = RFSelectedFeaturesClassifier()
    results = classifier.run_complete_pipeline()
    
    # Compare with existing methods
    compare_with_existing_methods()


if __name__ == "__main__":
    main()
