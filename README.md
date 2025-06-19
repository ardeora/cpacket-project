# DDoS Traffic Classification Project

A scalable machine learning pipeline for DDoS traffic classification using Random Forest algorithms with multiple feature selection methods.

## 📊 Dataset Overview

- **Dataset**: `datasets/ddos.parquet`
- **Size**: 540,494 rows × 319 columns
- **Classes**:
  - Benign: 349,178 samples (64.6%)
  - Attack: 170,436 samples (31.5%)
  - Suspicious: 20,880 samples (3.9%)

## 🎯 Features

### Multiple Feature Selection Methods

The project supports three feature selection methods, each with curated lists of the most important features:

1. **Information Gain** (`features/information_gain.json`) - 40 features
2. **Extra Trees** (`features/extra_tree.json`) - 40 features
3. **ANOVA F-test** (`features/anova.json`) - 40 features

### Key Capabilities

- ✅ **Scalable Architecture**: Object-oriented design for easy extension
- ✅ **Multiple Models**: Train Random Forest models for each feature selection method
- ✅ **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- ✅ **Model Persistence**: Automatic saving of models, encoders, and results
- ✅ **Feature Analysis**: Feature importance ranking and visualization
- ✅ **Large Dataset Support**: Efficient handling of 540k+ samples
- ✅ **Class Imbalance Handling**: Balanced class weights and stratified splitting

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (if not already active)
source cpacket/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Quick Test (Single Method)

```bash
python quick_test.py
```

This trains a Random Forest model using Information Gain features (20 features, 50 trees) for quick validation.

### 3. Full Training Pipeline

```bash
python classifier.py
```

This trains Random Forest models for all three feature selection methods with optimized parameters:

- 40 features per method
- 200 trees per model
- Advanced hyperparameters for better performance

## 📁 Project Structure

```
cpacket-project/
├── classifier.py           # Main classifier implementation
├── quick_test.py           # Quick validation script
├── requirements.txt        # Python dependencies
├── datasets/
│   └── ddos.parquet       # Main dataset
├── features/              # Feature selection configurations
│   ├── anova.json         # ANOVA F-test selected features
│   ├── extra_tree.json    # Extra Trees selected features
│   └── information_gain.json # Information Gain selected features
└── saved_models/          # Generated models and results
    ├── rf_classifier_*.joblib
    ├── label_encoder_*.joblib
    ├── features_*.json
    ├── results_*.json
    └── *_results.png
```

## 🔧 Classifier Class API

### DDosClassifier

```python
from classifier import DDosClassifier

# Initialize
classifier = DDosClassifier(
    data_path='datasets/ddos.parquet',
    features_dir='features/',
    models_dir='saved_models/'
)

# Load data and feature methods
classifier.load_data()
classifier.load_feature_methods()

# Train all models
classifier.train_all_methods(max_features=40)

# Or train individual model
features = classifier.get_features_for_method('information_gain', max_features=40)
X_train, X_test, y_train, y_test = classifier.prepare_data(features)
model = classifier.train_random_forest(X_train, y_train)
```

### Key Methods

- `load_data()`: Load parquet dataset
- `load_feature_methods()`: Load feature selection configurations
- `get_features_for_method(method, max_features)`: Get features for specific method
- `prepare_data(features)`: Prepare train/test splits
- `train_random_forest()`: Train Random Forest model
- `evaluate_model()`: Comprehensive model evaluation
- `save_model()`: Save model artifacts
- `train_all_methods()`: Train models for all feature methods

## 📈 Model Performance

The classifier automatically generates:

1. **Classification Reports**: Precision, recall, F1-score per class
2. **Confusion Matrices**: Visual representation of prediction accuracy
3. **Feature Importance**: Ranking of most influential features
4. **Performance Comparisons**: Summary table across all methods
5. **Visualizations**: Comprehensive plots saved as PNG files

## 🎛️ Configuration Options

### Random Forest Parameters

```python
classifier.train_all_methods(
    max_features=40,           # Number of features to use
    n_estimators=200,          # Number of trees
    max_depth=20,              # Maximum tree depth
    min_samples_split=5,       # Min samples to split node
    min_samples_leaf=2,        # Min samples at leaf
    random_state=42            # Reproducibility seed
)
```

### Feature Selection

- Modify JSON files in `features/` directory to customize feature sets
- Add new feature selection methods by creating new JSON files
- Each method automatically discovered and processed

## 📊 Expected Results

With the default configuration, you can expect:

- **Training Time**: ~2-5 minutes per method (depending on hardware)
- **Accuracy**: 90-95% across different feature methods
- **Memory Usage**: ~2-4GB during training
- **Model Size**: ~50-200MB per saved model

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `max_features` or use data sampling
2. **Missing Features**: Check that feature names in JSON files match dataset columns
3. **Long Training Time**: Reduce `n_estimators` or `max_features`

### Performance Optimization

- Use `n_jobs=-1` for parallel processing (default)
- Reduce dataset size for experimentation
- Adjust hyperparameters based on available resources

## 🔄 Extending the Classifier

### Adding New Feature Selection Methods

1. Create new JSON file in `features/` directory
2. Format: `{"features": ["feature1", "feature2", ...]}`
3. Run classifier - new method automatically detected

### Custom Evaluation Metrics

Extend the `evaluate_model()` method to include additional metrics like ROC-AUC, precision-recall curves, etc.

### Different Algorithms

The modular design allows easy integration of other classifiers (SVM, XGBoost, etc.) by extending the base class.

## 📝 Notes

- All models use balanced class weights to handle class imbalance
- Stratified sampling ensures representative train/test splits
- Feature importance helps identify the most discriminative network features
- Models and results are automatically saved for later use
- The pipeline is designed to scale with larger datasets and additional feature methods
