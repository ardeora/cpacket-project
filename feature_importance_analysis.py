#!/usr/bin/env python3
"""
Feature Importance Analysis Script for DDoS Classification

This script trains a Random Forest classifier using ALL available features 
from the dataset and then analyzes feature importance to identify the top 
40 most important features for classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using Random Forest on the complete dataset.
    """
    
    def __init__(self, data_path: str = 'datasets/ddos.parquet', 
                 output_dir: str = 'feature_analysis/'):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the dataset
            output_dir: Directory to save analysis results
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.model = None
        self.label_encoder = None
        self.feature_importance_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset."""
        print("Loading dataset...")
        try:
            self.df = pd.read_parquet(self.data_path)
            print(f"Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"Target distribution:\n{self.df['label'].value_counts()}")
            return self.df
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def prepare_features(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Prepare all features for training.
        
        Returns:
            Tuple of (features_df, target_array, feature_names)
        """
        print("\nPreparing features...")
        
        # Identify feature columns (exclude target and identifier columns)
        exclude_columns = ['label', 'activity']
        available_exclude = [col for col in exclude_columns if col in self.df.columns]
        
        # Get all feature columns
        feature_columns = [col for col in self.df.columns if col not in available_exclude]
        
        print(f"Total features available: {len(feature_columns)}")
        print(f"Excluded columns: {available_exclude}")
        
        # Extract features and target
        X = self.df[feature_columns].copy()
        y = self.df['label'].copy()
        
        # Handle missing values
        print("Checking for missing values...")
        missing_counts = X.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            print(f"Found {total_missing} missing values across {(missing_counts > 0).sum()} features")
            print("Filling missing values with median...")
            X = X.fillna(X.median())
        else:
            print("No missing values found!")
        
        # Check for infinite values
        print("Checking for infinite values...")
        inf_mask = np.isinf(X.values)
        if inf_mask.any():
            print(f"Found {inf_mask.sum()} infinite values, replacing with finite values...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Feature data types: {X.dtypes.value_counts().to_dict()}")
        
        return X, y_encoded, feature_columns
    
    def train_full_model(self, X: pd.DataFrame, y: np.ndarray, 
                        n_estimators: int = 200, 
                        max_depth: int = 20,
                        test_size: float = 0.2,
                        random_state: int = 42) -> Dict:
        """
        Train Random Forest on all features and evaluate performance.
        
        Args:
            X: Feature matrix
            y: Target array (encoded)
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary with training results
        """
        print(f"\nTraining Random Forest with ALL {X.shape[1]} features...")
        print(f"Parameters: {n_estimators} estimators, max_depth={max_depth}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Training completed! 🎉")
        
        # Evaluate model
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return {
            'model': self.model,
            'accuracy': accuracy,
            'classification_report': report,
            'feature_names': list(X.columns),
            'n_features': X.shape[1]
        }
    
    def analyze_feature_importance(self, feature_names: List[str], 
                                 top_n: int = 40) -> pd.DataFrame:
        """
        Analyze and rank feature importance.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance rankings
        """
        print(f"\nAnalyzing feature importance (top {top_n})...")
        
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create feature importance DataFrame
        self.feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_percentage': importances * 100
        }).sort_values('importance', ascending=False)
        
        # Add ranking
        self.feature_importance_df['rank'] = range(1, len(self.feature_importance_df) + 1)
        
        # Get top N features
        top_features = self.feature_importance_df.head(top_n)
        
        print(f"Top {top_n} Most Important Features:")
        print("=" * 60)
        for idx, row in top_features.iterrows():
            print(f"{row['rank']:2d}. {row['feature']:<35} {row['importance']:.6f} ({row['importance_percentage']:.3f}%)")
        
        return top_features
    
    def plot_feature_importance(self, top_n: int = 40, save_plots: bool = True):
        """
        Create visualizations for feature importance.
        
        Args:
            top_n: Number of top features to visualize
            save_plots: Whether to save plots to disk
        """
        if self.feature_importance_df is None:
            raise ValueError("Feature importance analysis must be run first!")
        
        top_features = self.feature_importance_df.head(top_n)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Feature Importance Analysis - Top {top_n} Features', fontsize=16)
        
        # 1. Horizontal bar plot of top features
        axes[0, 0].barh(range(len(top_features)), top_features['importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('Feature Importance')
        axes[0, 0].set_title(f'Top {top_n} Feature Importances')
        axes[0, 0].invert_yaxis()
        
        # 2. Cumulative importance
        cumulative_importance = top_features['importance_percentage'].cumsum()
        axes[0, 1].plot(range(1, len(top_features) + 1), cumulative_importance, 'o-')
        axes[0, 1].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% threshold')
        axes[0, 1].axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
        axes[0, 1].set_xlabel('Number of Features')
        axes[0, 1].set_ylabel('Cumulative Importance (%)')
        axes[0, 1].set_title('Cumulative Feature Importance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature importance distribution
        axes[1, 0].hist(self.feature_importance_df['importance'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(top_features.iloc[-1]['importance'], color='r', linestyle='--', 
                          label=f'Top {top_n} threshold')
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].set_title('Distribution of All Feature Importances')
        axes[1, 0].legend()
        
        # 4. Top 20 features (more readable)
        top_20 = top_features.head(20)
        axes[1, 1].barh(range(len(top_20)), top_20['importance'])
        axes[1, 1].set_yticks(range(len(top_20)))
        axes[1, 1].set_yticklabels(top_20['feature'])
        axes[1, 1].set_xlabel('Feature Importance')
        axes[1, 1].set_title('Top 20 Feature Importances (Detailed)')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f'feature_importance_top_{top_n}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nFeature importance plot saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self, top_features: pd.DataFrame, model_results: Dict, 
                    top_n: int = 40):
        """
        Save analysis results to files.
        
        Args:
            top_features: DataFrame with top features
            model_results: Dictionary with model training results
            top_n: Number of top features
        """
        print(f"\nSaving results to {self.output_dir}...")
        
        # Save top features as JSON (for use in other scripts)
        top_features_json = {
            'features': top_features['feature'].tolist(),
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_features_analyzed': len(self.feature_importance_df),
            'model_accuracy': model_results['accuracy'],
            'n_estimators': self.model.n_estimators,
            'description': f'Top {top_n} features by Random Forest importance'
        }
        
        json_path = self.output_dir / f'top_{top_n}_features.json'
        with open(json_path, 'w') as f:
            json.dump(top_features_json, f, indent=2)
        
        # Save detailed feature importance CSV
        csv_path = self.output_dir / 'all_feature_importance.csv'
        self.feature_importance_df.to_csv(csv_path, index=False)
        
        # Save top features CSV
        top_csv_path = self.output_dir / f'top_{top_n}_features.csv'
        top_features.to_csv(top_csv_path, index=False)
        
        # Save model summary
        summary = {
            'analysis_summary': {
                'total_features': model_results['n_features'],
                'top_features_selected': top_n,
                'model_accuracy': model_results['accuracy'],
                'dataset_shape': list(self.df.shape),
                'class_distribution': self.df['label'].value_counts().to_dict()
            },
            'model_performance': model_results['classification_report'],
            'top_features': top_features_json['features']
        }
        
        summary_path = self.output_dir / 'analysis_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved:")
        print(f"   - Top {top_n} features (JSON): {json_path}")
        print(f"   - Top {top_n} features (CSV): {top_csv_path}")
        print(f"   - All feature importance: {csv_path}")
        print(f"   - Analysis summary: {summary_path}")
    
    def run_complete_analysis(self, top_n: int = 40, **model_params):
        """
        Run the complete feature importance analysis pipeline.
        
        Args:
            top_n: Number of top features to analyze
            **model_params: Additional parameters for RandomForestClassifier
        """
        print("🔍 STARTING COMPLETE FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Prepare features
        X, y, feature_names = self.prepare_features()
        
        # Train model
        model_results = self.train_full_model(X, y, **model_params)
        
        # Analyze feature importance
        top_features = self.analyze_feature_importance(feature_names, top_n)
        
        # Create visualizations
        self.plot_feature_importance(top_n)
        
        # Save results
        self.save_results(top_features, model_results, top_n)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📊 Model accuracy with ALL features: {model_results['accuracy']:.4f}")
        print(f"🔝 Top {top_n} features identified and saved")
        
        return {
            'top_features': top_features,
            'model_results': model_results,
            'feature_importance_df': self.feature_importance_df
        }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze feature importance using Random Forest on all dataset features'
    )
    parser.add_argument('--data-path', type=str, default='datasets/ddos.parquet',
                       help='Path to the dataset')
    parser.add_argument('--output-dir', type=str, default='feature_analysis/',
                       help='Directory to save analysis results')
    parser.add_argument('--top-n', type=int, default=40,
                       help='Number of top features to analyze')
    parser.add_argument('--n-estimators', type=int, default=200,
                       help='Number of trees in Random Forest')
    parser.add_argument('--max-depth', type=int, default=20,
                       help='Maximum depth of trees')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FeatureImportanceAnalyzer(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Run analysis
    results = analyzer.run_complete_analysis(
        top_n=args.top_n,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size
    )
    
    # Print summary
    print(f"\n📋 SUMMARY:")
    print(f"   Total features in dataset: {len(results['feature_importance_df'])}")
    print(f"   Model accuracy: {results['model_results']['accuracy']:.4f}")
    print(f"   Top {args.top_n} features saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
