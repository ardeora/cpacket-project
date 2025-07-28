import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import List, Dict, Tuple, Optional
import numpy as np


class DDosClassifier:
    """
    A scalable DDoS classifier that can train Random Forest models using
    different feature selection methods and handle large datasets efficiently.
    """
    
    def __init__(self, data_path: str = 'datasets/ddos.parquet', 
                 features_dir: str = 'features/',
                 models_dir: str = 'saved_models/',
                 plots_dir: str = 'plots/',
                 results_dir: str = 'results/'):
        """
        Initialize the classifier with paths to data and feature configurations.
        
        Args:
            data_path: Path to the parquet dataset file
            features_dir: Directory containing feature selection JSON files
            models_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.label_encoder = None
        self.feature_methods = {}
        self.trained_models = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from parquet file."""
        print("Loading data from parquet file...")
        try:
            self.df = pd.read_parquet(self.data_path)
            print(f"Successfully loaded data. Shape: {self.df.shape}")
            print(f"Target distribution:\n{self.df['label'].value_counts()}")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file '{self.data_path}' not found.")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def load_feature_methods(self) -> Dict[str, List[str]]:
        """Load all feature selection methods from JSON files."""
        print("\nLoading feature selection methods...")
        
        for json_file in self.features_dir.glob("*.json"):
            method_name = json_file.stem
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Extract features, excluding 'label' and 'activity' if present
                    features = [f for f in data['features'] if f not in ['label', 'activity']]
                    self.feature_methods[method_name] = features
                    print(f"  - {method_name}: {len(features)} features")
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
        
        if not self.feature_methods:
            raise ValueError("No valid feature selection methods found in features directory.")
        
        return self.feature_methods
    
    def get_features_for_method(self, method: str, max_features: int = 40) -> List[str]:
        """
        Get the top N features for a specific method.
        
        Args:
            method: Feature selection method name
            max_features: Maximum number of features to return (default: 40)
            
        Returns:
            List of feature names
        """
        if method not in self.feature_methods:
            raise ValueError(f"Method '{method}' not found. Available methods: {list(self.feature_methods.keys())}")
        
        features = self.feature_methods[method][:max_features]
        
        # Verify all features exist in the dataset
        available_features = [f for f in features if f in self.df.columns]
        missing_features = [f for f in features if f not in self.df.columns]
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in dataset: {missing_features[:5]}...")
        
        print(f"Using {len(available_features)} features for method '{method}'")
        return available_features
    
    def prepare_data(self, features: List[str], test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and target variables for training.
        
        Args:
            features: List of feature column names
            test_size: Proportion of dataset for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\nPreparing data with {len(features)} features...")
        
        # Extract features and target
        X = self.df[features].copy()
        y = self.df['label'].copy()
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("Handling missing values...")
            X = X.fillna(X.median())
        
        # Encode labels if not already done
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_encoded
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200, **rf_params) -> RandomForestClassifier:
        """
        Train a Random Forest classifier with optimized parameters.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees in the forest
            
        Returns:
            Trained RandomForestClassifier
        """
        print(f"\nTraining Random Forest with {n_estimators} estimators...")
        
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,  # Use all available CPU cores
            class_weight='balanced',  # Handle imbalanced classes
            **rf_params  # Additional parameters for flexibility
        )
        
        rf_classifier.fit(X_train, y_train)
        print("Model training complete! 🎉")
        
        return rf_classifier
    
    def evaluate_model(self, model: RandomForestClassifier, X_test: np.ndarray, 
                      y_test: np.ndarray, method_name: str) -> Dict:
        """
        Evaluate the trained model and return metrics.
        
        Args:
            model: Trained classifier
            X_test: Test features
            y_test: Test labels
            method_name: Name of the feature selection method
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\nEvaluating model for method: {method_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns if hasattr(X_test, 'columns') else range(X_test.shape[1]),
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': feature_importance,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def plot_results(self, results: Dict, method_name: str, save_plots: bool = True):
        """
        Create visualizations for model results.
        
        Args:
            results: Dictionary containing evaluation results
            method_name: Name of the feature selection method
            save_plots: Whether to save plots to disk
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Results - {method_name.title()} Features', fontsize=16)
        
        # Confusion Matrix
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Feature Importance (Top 20)
        top_features = results['feature_importance'].head(20)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_title('Top 20 Feature Importance')
        axes[0, 1].set_xlabel('Importance')
        
        # Class Distribution
        class_counts = pd.Series(self.df['label'].value_counts())
        axes[1, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Class Distribution in Dataset')
        
        # Accuracy by Class
        report = results['classification_report']
        classes = [c for c in report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[c]['f1-score'] for c in classes]
        
        axes[1, 1].bar(classes, f1_scores)
        axes[1, 1].set_title('F1-Score by Class')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.plots_dir / f'{method_name}_results.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to: {plot_path}")
        
        # plt.show()
    
    def save_model(self, model: RandomForestClassifier, method_name: str, 
                  features: List[str], results: Dict):
        """
        Save the trained model and related artifacts.
        
        Args:
            model: Trained classifier
            method_name: Feature selection method name
            features: List of features used
            results: Evaluation results
        """
        # Save model
        model_path = self.models_dir / f'rf_classifier_{method_name}.joblib'
        joblib.dump(model, model_path)
        
        # Save label encoder
        encoder_path = self.models_dir / f'label_encoder_{method_name}.joblib'
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save results summary
        results_path = self.results_dir / f'results_{method_name}.json'
        results_summary = {
            'method': method_name,
            'accuracy': results['accuracy'],
            'n_features': len(features),
            'classification_report': results['classification_report']
        }
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nModel artifacts saved:")
        print(f"  - Model: {model_path}")
        print(f"  - Encoder: {encoder_path}")
        print(f"  - Results: {results_path}")
    
    def train_all_methods(self, max_features: int = 40, **rf_params):
        """
        Train Random Forest models for all available feature selection methods.
        
        Args:
            max_features: Maximum number of features to use per method
            **rf_params: Additional parameters for RandomForestClassifier
        """
        print(f"\n{'='*60}")
        print(f"TRAINING MODELS FOR ALL FEATURE SELECTION METHODS")
        print(f"Max features per method: {max_features}")
        print(f"{'='*60}")
        
        for method_name in self.feature_methods.keys():
            print(f"\n{'*'*40}")
            print(f"TRAINING MODEL: {method_name.upper()}")
            print(f"{'*'*40}")
            
            try:
                # Get features for this method
                features = self.get_features_for_method(method_name, max_features)
                
                if not features:
                    print(f"No valid features found for method '{method_name}'. Skipping...")
                    continue
                
                # Prepare data
                X_train, X_test, y_train, y_test = self.prepare_data(features)
                
                # Train model
                model = self.train_random_forest(X_train, y_train, **rf_params)
                
                # Evaluate model
                results = self.evaluate_model(model, X_test, y_test, method_name)
                
                # Create visualizations
                self.plot_results(results, method_name)
                
                # Save model and results
                self.save_model(model, method_name, features, results)
                
                # Store in memory for comparison
                self.trained_models[method_name] = {
                    'model': model,
                    'features': features,
                    'results': results
                }
                
                print(f"\n✅ Successfully completed training for '{method_name}'")
                
            except Exception as e:
                print(f"❌ Error training model for '{method_name}': {str(e)}")
                continue
        
        # Print summary
        self.print_models_summary()
    
    def print_models_summary(self):
        """Print a summary comparison of all trained models."""
        if not self.trained_models:
            print("No models have been trained yet.")
            return
        
        print(f"\n{'='*60}")
        print("MODELS SUMMARY")
        print(f"{'='*60}")
        
        summary_data = []
        for method_name, model_info in self.trained_models.items():
            results = model_info['results']
            summary_data.append({
                'Method': method_name,
                'Features': len(model_info['features']),
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1-Score (Macro)': f"{results['classification_report']['macro avg']['f1-score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        print(summary_df.to_string(index=False))
        
        # Find best model
        best_method = summary_df.iloc[0]['Method']
        print(f"\n🏆 Best performing method: {best_method}")


def main():
    """Main execution function."""
    # Initialize classifier
    classifier = DDosClassifier()
    
    # Load data
    classifier.load_data()
    
    # Load feature selection methods
    classifier.load_feature_methods()
    
    # Train models for all methods with 40 features each
    classifier.train_all_methods(
        n_estimators=200,
        random_state=42
    )
    
    print("\n🎉 All models have been trained and saved!")
    print(f"Check the '{classifier.models_dir}' directory for saved models and results.")


if __name__ == "__main__":
    main()