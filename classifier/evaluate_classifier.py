#!/usr/bin/env python3
"""
DDoS Classifier Evaluation Script

This script loads a saved random forest classifier (extra_tree method) and evaluates it
on a CSV file, showing predictions vs actual labels for each row.
"""

import pandas as pd
import joblib
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import List, Dict, Tuple


class ClassifierEvaluator:
    """
    A class to load and evaluate saved DDoS classifiers on new data.
    """
    
    def __init__(self, model_method: str = 'extra_tree',
                 models_dir: str = 'saved_models/',
                 features_dir: str = 'features/'):
        """
        Initialize the evaluator.
        
        Args:
            model_method: The feature selection method for the model to load
            models_dir: Directory containing saved models
            features_dir: Directory containing feature selection JSON files
        """
        self.model_method = model_method
        self.models_dir = Path(models_dir)
        self.features_dir = Path(features_dir)
        
        self.classifier = None
        self.label_encoder = None
        self.features = None
        
    def load_model(self) -> Tuple[object, object]:
        """Load the saved classifier and label encoder."""
        print(f"Loading {self.model_method} classifier...")
        
        # Load the classifier
        classifier_path = self.models_dir / f'rf_classifier_{self.model_method}.joblib'
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {classifier_path}")
        
        self.classifier = joblib.load(classifier_path)
        print(f"✓ Loaded classifier from {classifier_path}")
        
        # Load the label encoder
        encoder_path = self.models_dir / f'label_encoder_{self.model_method}.joblib'
        if not encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        
        self.label_encoder = joblib.load(encoder_path)
        print(f"✓ Loaded label encoder from {encoder_path}")
        print(f"✓ Label classes: {list(self.label_encoder.classes_)}")
        
        return self.classifier, self.label_encoder
    
    def load_features(self) -> List[str]:
        """Load the feature list for the specified method."""
        features_path = self.features_dir / f'{self.model_method}.json'
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        with open(features_path, 'r') as f:
            data = json.load(f)
            # Extract features, excluding 'label' and 'activity' if present
            self.features = [f for f in data['features'] if f not in ['label', 'activity']]
        
        print(f"✓ Loaded {len(self.features)} features for {self.model_method} method")
        return self.features
    
    def load_test_data(self, csv_path: str, max_features: int = 40) -> pd.DataFrame:
        """
        Load and prepare test data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            max_features: Maximum number of features to use (default: 40)
            
        Returns:
            Prepared DataFrame with features and labels
        """
        print(f"\nLoading test data from {csv_path}...")
        
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Test data file not found: {csv_path}")
        
        # Load the CSV file
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} rows from CSV")
        
        # Check if required columns exist
        if 'label' not in df.columns:
            raise ValueError("CSV file must contain a 'label' column")
        
        # Use only the top N features
        features_to_use = self.features[:max_features]
        
        # Check which features are available in the data
        available_features = [f for f in features_to_use if f in df.columns]
        missing_features = [f for f in features_to_use if f not in df.columns]
        
        if missing_features:
            print(f"⚠️  Warning: {len(missing_features)} features not found in CSV:")
            for feature in missing_features[:5]:  # Show first 5 missing features
                print(f"   - {feature}")
            if len(missing_features) > 5:
                print(f"   ... and {len(missing_features) - 5} more")
        
        print(f"✓ Using {len(available_features)} out of {len(features_to_use)} features")
        
        # Prepare the feature matrix
        X = df[available_features].copy()
        y = df['label'].copy()
        
        # Include activity column if it exists for display purposes
        activity = df['activity'].copy() if 'activity' in df.columns else None
        
        return X, y, activity, available_features
    
    def evaluate_and_display(self, csv_path: str, max_features: int = 40, 
                           show_all_predictions: bool = False, max_display: int = 20):
        """
        Evaluate the classifier on test data and display results.
        
        Args:
            csv_path: Path to the CSV file
            max_features: Maximum number of features to use
            show_all_predictions: Whether to show all predictions or just errors
            max_display: Maximum number of predictions to display
        """
        # Load model and features
        self.load_model()
        self.load_features()
        
        # Load and prepare test data
        X, y_true, activity, features_used = self.load_test_data(csv_path, max_features)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred_encoded = self.classifier.predict(X)
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Decode predictions and true labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Encode true labels for comparison
        y_true_encoded = self.label_encoder.transform(y_true)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total samples: {len(y_true)}")
        print(f"Features used: {len(features_used)}")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Show detailed classification report
        print(f"\nDetailed Classification Report:")
        try:
            print(classification_report(y_true, y_pred, target_names=self.label_encoder.classes_, zero_division=0))
        except ValueError as e:
            # Handle case where not all classes are present in test data
            print("Note: Not all trained classes are present in the test data.")
            print(classification_report(y_true, y_pred, zero_division=0))
        
        # Display individual predictions
        print(f"\n{'='*60}")
        print(f"INDIVIDUAL PREDICTIONS")
        print(f"{'='*60}")
        
        # Create a DataFrame for easier display
        results_df = pd.DataFrame({
            'Row': range(len(y_true)),
            'Predicted': y_pred,
            'Actual': y_true,
            'Correct': y_pred == y_true,
            'Confidence': np.max(y_pred_proba, axis=1)
        })
        
        if activity is not None:
            results_df['Activity'] = activity
        
        # Filter results based on user preference
        if show_all_predictions:
            display_df = results_df.head(max_display)
            print(f"Showing first {len(display_df)} predictions:")
        else:
            # Show only incorrect predictions
            incorrect_df = results_df[~results_df['Correct']]
            display_df = incorrect_df.head(max_display)
            print(f"Showing {len(display_df)} incorrect predictions (out of {len(incorrect_df)} total errors):")
        
        # Display results in a nice format
        if len(display_df) > 0:
            print(f"\n{'Row':<5} {'Predicted':<15} {'Actual':<15} {'Confidence':<12} {'Status':<10}", end="")
            if activity is not None:
                print(f" {'Activity':<25}")
            else:
                print()
            print("-" * (60 + (25 if activity is not None else 0)))
            
            for _, row in display_df.iterrows():
                status = "✓ Correct" if row['Correct'] else "✗ Wrong"
                print(f"{row['Row']:<5} {row['Predicted']:<15} {row['Actual']:<15} {row['Confidence']:<12.4f} {status:<10}", end="")
                if activity is not None:
                    print(f" {row['Activity']:<25}")
                else:
                    print()
        else:
            if not show_all_predictions:
                print("🎉 All predictions are correct!")
        
        # Show summary statistics
        correct_count = (y_pred == y_true).sum()
        incorrect_count = len(y_true) - correct_count
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Correct predictions: {correct_count}/{len(y_true)} ({correct_count/len(y_true)*100:.2f}%)")
        print(f"✗ Incorrect predictions: {incorrect_count}/{len(y_true)} ({incorrect_count/len(y_true)*100:.2f}%)")
        
        # Show confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred, labels=self.label_encoder.classes_)
        
        # Create a simple text-based confusion matrix
        classes = self.label_encoder.classes_
        
        # Check which classes are actually present in the data
        unique_true = set(y_true)
        unique_pred = set(y_pred)
        present_classes = unique_true.union(unique_pred)
        
        if len(present_classes) < len(classes):
            print(f"Note: Only {len(present_classes)} out of {len(classes)} classes present in test data: {sorted(present_classes)}")
        
        header = "Actual \\ Predicted"
        print(f"\n{header:<20}", end="")
        for cls in classes:
            print(f"{cls:<15}", end="")
        print()
        print("-" * (20 + len(classes) * 15))
        
        for i, actual_cls in enumerate(classes):
            print(f"{actual_cls:<20}", end="")
            for j, _ in enumerate(classes):
                print(f"{cm[i,j]:<15}", end="")
            print()
        
        return results_df


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate DDoS Classifier on CSV data')
    parser.add_argument('csv_file', help='Path to the CSV file to evaluate')
    parser.add_argument('--method', '-m', default='extra_tree', 
                       help='Feature selection method (default: extra_tree)')
    parser.add_argument('--max-features', '-f', type=int, default=40,
                       help='Maximum number of features to use (default: 40)')
    parser.add_argument('--show-all', '-a', action='store_true',
                       help='Show all predictions, not just incorrect ones')
    parser.add_argument('--max-display', '-d', type=int, default=20,
                       help='Maximum number of predictions to display (default: 20)')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator instance
        evaluator = ClassifierEvaluator(model_method=args.method)
        
        # Run evaluation
        results = evaluator.evaluate_and_display(
            csv_path=args.csv_file,
            max_features=args.max_features,
            show_all_predictions=args.show_all,
            max_display=args.max_display
        )
        
        print(f"\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
