#!/usr/bin/env python3
"""
Model inference script for making predictions with trained DDoS classifiers.
"""

import pandas as pd
import joblib
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse

class DDosPredictor:
    """
    Load and use trained DDoS classification models for making predictions.
    """
    
    def __init__(self, models_dir: str = 'saved_models/'):
        """
        Initialize the predictor with path to saved models.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.available_methods = self._find_available_methods()
        self.loaded_models = {}
        
    def _find_available_methods(self) -> List[str]:
        """Find all available trained methods."""
        methods = []
        for model_file in self.models_dir.glob('rf_classifier_*.joblib'):
            method = model_file.stem.replace('rf_classifier_', '')
            if method != 'quick_test':  # Skip test models
                methods.append(method)
        return methods
    
    def load_model(self, method: str) -> Dict[str, Any]:
        """
        Load a specific trained model and its artifacts.
        
        Args:
            method: The feature selection method name
            
        Returns:
            Dictionary containing model, encoder, and features
        """
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not available. Available: {self.available_methods}")
        
        # Load model components
        model_path = self.models_dir / f'rf_classifier_{method}.joblib'
        encoder_path = self.models_dir / f'label_encoder_{method}.joblib'
        features_path = self.models_dir / f'features_{method}.json'
        
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        
        with open(features_path, 'r') as f:
            features_info = json.load(f)
        
        self.loaded_models[method] = {
            'model': model,
            'encoder': encoder,
            'features': features_info['features'],
            'n_features': features_info['n_features']
        }
        
        print(f"✅ Loaded model for '{method}' method")
        print(f"   Features: {len(features_info['features'])}")
        
        return self.loaded_models[method]
    
    def predict(self, data: pd.DataFrame, method: str, 
                return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions using a specific method.
        
        Args:
            data: DataFrame with network traffic features
            method: Feature selection method to use
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        if method not in self.loaded_models:
            self.load_model(method)
        
        model_info = self.loaded_models[method]
        model = model_info['model']
        encoder = model_info['encoder']
        required_features = model_info['features']
        
        # Prepare features
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        X = data[required_features].copy()
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        # Make predictions
        y_pred_encoded = model.predict(X)
        y_pred_labels = encoder.inverse_transform(y_pred_encoded)
        
        results = {
            'predictions': y_pred_labels,
            'encoded_predictions': y_pred_encoded,
            'method': method,
            'n_samples': len(data)
        }
        
        if return_probabilities:
            y_pred_proba = model.predict_proba(X)
            results['probabilities'] = y_pred_proba
            results['class_labels'] = encoder.classes_
        
        return results
    
    def predict_ensemble(self, data: pd.DataFrame, 
                        methods: List[str] = None,
                        voting: str = 'hard') -> Dict[str, Any]:
        """
        Make ensemble predictions using multiple methods.
        
        Args:
            data: DataFrame with network traffic features
            methods: List of methods to use (default: all available)
            voting: 'hard' for majority vote, 'soft' for probability averaging
            
        Returns:
            Dictionary with ensemble predictions
        """
        if methods is None:
            methods = self.available_methods
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each method
        for method in methods:
            try:
                result = self.predict(data, method, return_probabilities=True)
                predictions[method] = result['encoded_predictions']
                probabilities[method] = result['probabilities']
            except Exception as e:
                print(f"Warning: Could not get predictions from {method}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No successful predictions from any method")
        
        # Load encoder (all methods should have the same classes)
        first_method = list(predictions.keys())[0]
        encoder = self.loaded_models[first_method]['encoder']
        
        if voting == 'hard':
            # Majority voting
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), 
                axis=0, arr=pred_array
            )
        else:
            # Soft voting (average probabilities)
            prob_array = np.array(list(probabilities.values()))
            avg_probabilities = np.mean(prob_array, axis=0)
            ensemble_pred = np.argmax(avg_probabilities, axis=1)
        
        ensemble_labels = encoder.inverse_transform(ensemble_pred)
        
        return {
            'ensemble_predictions': ensemble_labels,
            'ensemble_encoded': ensemble_pred,
            'individual_predictions': {
                method: encoder.inverse_transform(pred) 
                for method, pred in predictions.items()
            },
            'methods_used': list(predictions.keys()),
            'voting_method': voting
        }
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get a summary of all available models."""
        summary_data = []
        
        for method in self.available_methods:
            results_path = self.models_dir / f'results_{method}.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                summary_data.append({
                    'Method': method,
                    'Accuracy': f"{results['accuracy']:.4f}",
                    'Features': results['n_features'],
                    'F1-Score (Macro)': f"{results['classification_report']['macro avg']['f1-score']:.4f}"
                })
        
        return pd.DataFrame(summary_data).sort_values('Accuracy', ascending=False)


def main():
    """Demo script showing how to use the predictor."""
    parser = argparse.ArgumentParser(description='DDoS Classification Predictor')
    parser.add_argument('--method', type=str, default='extra_tree',
                       help='Feature selection method to use')
    parser.add_argument('--data-path', type=str, default='datasets/ddos.parquet',
                       help='Path to data for prediction')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of samples to predict (for demo)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble prediction with all methods')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DDosPredictor()
    
    # Show available models
    print("Available trained models:")
    print(predictor.get_model_summary())
    print()
    
    # Load test data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    
    # Sample data for demonstration
    sample_data = df.sample(n=min(args.sample_size, len(df)), random_state=42)
    actual_labels = sample_data['label'].values
    feature_data = sample_data.drop(['label', 'activity'], axis=1, errors='ignore')
    
    print(f"Making predictions on {len(sample_data)} samples...")
    
    if args.ensemble:
        # Ensemble prediction
        results = predictor.predict_ensemble(feature_data)
        predicted_labels = results['ensemble_predictions']
        
        print(f"\n🔮 Ensemble Predictions (using {len(results['methods_used'])} methods):")
        print(f"Methods used: {', '.join(results['methods_used'])}")
        
        # Show individual method predictions for first 5 samples
        print(f"\nFirst 5 predictions breakdown:")
        for i in range(min(5, len(predicted_labels))):
            print(f"Sample {i+1}: Actual={actual_labels[i]}, Ensemble={predicted_labels[i]}")
            for method, preds in results['individual_predictions'].items():
                print(f"  {method}: {preds[i]}")
    
    else:
        # Single method prediction
        results = predictor.predict(feature_data, args.method)
        predicted_labels = results['predictions']
        
        print(f"\n🔮 Predictions using '{args.method}' method:")
    
    # Calculate accuracy
    accuracy = (predicted_labels == actual_labels).mean()
    print(f"\n📊 Accuracy on sample: {accuracy:.4f}")
    
    # Show prediction distribution
    unique, counts = np.unique(predicted_labels, return_counts=True)
    print(f"\nPrediction distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(predicted_labels)*100:.1f}%)")


if __name__ == "__main__":
    main()
