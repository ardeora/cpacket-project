# DDoS Traffic Classification Project

A high-performance machine learning pipeline for DDoS traffic classification using Random Forest algorithms with multiple feature selection methods.

## 📊 Dataset Overview

- **Dataset**: `datasets/ddos.parquet`
- **Size**: 540,494 rows × 319 columns
- **Classes**:
  - Benign: 349,178 samples (64.6%)
  - Attack: 170,436 samples (31.5%)
  - Suspicious: 20,880 samples (3.9%)

## 🎯 Features & Performance

### Feature Selection Methods

The project includes three different feature selection approaches:

1. **🏆 Extra Trees** (`features/extra_tree.json`) - 40 features → **94.98% accuracy** (BEST)
2. **Information Gain** (`features/information_gain.json`) - 40 features → **94.40% accuracy**
3. **ANOVA F-test** (`features/anova.json`) - 40 features → **85.29% accuracy**

### Key Capabilities

- ✅ **High Performance**: Best model achieves 94.98% accuracy on 540k+ samples
- ✅ **Scalable Architecture**: Object-oriented design for easy extension
- ✅ **Multiple Models**: Trained Random Forest models for each feature selection method
- ✅ **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- ✅ **Organized Results**: Clean separation of models, results, and plots
- ✅ **Feature Analysis**: Advanced feature importance ranking and visualization
- ✅ **Class Imbalance Handling**: Balanced class weights and stratified splitting

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source cpacket/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Train All Models

```bash
python classifier.py
```

This trains Random Forest models for all feature selection methods:

- Uses optimized hyperparameters (200 trees, balanced classes)
- Automatically saves models, results, and visualizations
- Generates performance comparison reports

## 📁 Project Structure

```
cpacket-project/
├── classifier.py           # Main classifier implementation
├── requirements.txt        # Python dependencies
├── datasets/
│   └── ddos.parquet       # DDoS traffic dataset (540k samples)
├── features/              # Feature selection configurations
│   ├── anova.json         # ANOVA F-test selected features
│   ├── extra_tree.json    # Extra Trees selected features
│   └── information_gain.json # Information Gain selected features
├── saved_models/          # Trained models and encoders
│   ├── rf_classifier_*.joblib    # Trained Random Forest models
│   └── label_encoder_*.joblib    # Label encoders for each method
├── results/               # Model performance results
│   ├── results_anova.json
│   ├── results_extra_tree.json
│   └── results_information_gain.json
└── plots/                 # Generated visualizations
    ├── anova_results.png
    ├── extra_tree_results.png
    └── information_gain_results.png
```

## 🔧 Using the Classifier

### Training Models

```python
from classifier import DDosClassifier

# Initialize classifier
classifier = DDosClassifier(
    data_path='datasets/ddos.parquet',
    features_dir='features/',
    models_dir='saved_models/',
    results_dir='results/',
    plots_dir='plots/'
)

# Load data and feature methods
classifier.load_data()
classifier.load_feature_methods()

# Train all models
classifier.train_all_methods(max_features=40)
```

### Loading Pre-trained Models

```python
import joblib

# Load the best performing model (Extra Trees features)
model = joblib.load('saved_models/rf_classifier_extra_tree.joblib')
encoder = joblib.load('saved_models/label_encoder_extra_tree.joblib')

# Make predictions
predictions = model.predict(new_data)
labels = encoder.inverse_transform(predictions)
```

## 📈 Model Performance

### Current Results (Accuracy on Test Set)

| Method           | Accuracy   | F1-Score (Macro) | Features | Status      |
| ---------------- | ---------- | ---------------- | -------- | ----------- |
| **Extra Trees**  | **94.98%** | **84.5%**        | 40       | 🏆 **Best** |
| Information Gain | 94.40%     | 83.2%            | 40       | �           |
| ANOVA F-test     | 85.29%     | 69.7%            | 40       | 🥉          |

### Performance Details

The classifier automatically generates:

1. **Classification Reports**: Precision, recall, F1-score per class
2. **Confusion Matrices**: Visual representation of prediction accuracy
3. **Feature Importance**: Ranking of most influential features
4. **Performance Plots**: Comprehensive visualizations saved in `plots/`
5. **Results Files**: Detailed metrics saved in `results/`

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

## 📊 Best Features Identified

The Extra Trees method identified the most important features for DDoS detection. Each feature selection method discovers different patterns in the data that are useful for classification.

## 📈 Expected Results

With the current configuration:

- **Training Time**: 2-5 minutes per method (depending on hardware)
- **Best Accuracy**: 94.98% (Extra Trees features)
- **Memory Usage**: 2-4GB during training
- **Model Size**: 50-200MB per saved model

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error**: Reduce `max_features` or use data sampling
2. **Missing Features**: Check that feature names in JSON files match dataset columns
3. **Long Training Time**: Reduce `n_estimators` or `max_features`

### Performance Optimization

- Use `n_jobs=-1` for parallel processing (default)
- Models are saved for reuse - no need to retrain
- Check `results/` directory for detailed performance metrics

## 🔄 Extending the Classifier

### Adding New Feature Selection Methods

1. Create new JSON file in `features/` directory
2. Format: `{"features": ["feature1", "feature2", ...]}`
3. Run classifier - new method automatically detected

### Custom Models

The modular design allows easy integration of other classifiers (SVM, XGBoost, etc.) by extending the base class.

## 📁 Files Reference

### Key Files

- **`classifier.py`**: Main training script
- **`datasets/ddos.parquet`**: DDoS traffic dataset
- **`features/*.json`**: Feature selection configurations
- **`saved_models/rf_classifier_extra_tree.joblib`**: Best performing model
- **`results/results_extra_tree.json`**: Best model performance metrics
- **`plots/extra_tree_results.png`**: Best model visualizations

### Generated Results

All training runs automatically generate:

- Trained models in `saved_models/`
- Performance metrics in `results/`
- Visualization plots in `plots/`

## 🎯 Key Achievements

- ✅ **94.98% accuracy** on 540k+ sample DDoS dataset
- ✅ **Automated feature selection** identifies optimal feature combinations
- ✅ **Clean, organized structure** with separated models, results, and plots
- ✅ **Production-ready models** saved and ready for deployment
- ✅ **Comprehensive evaluation** with detailed metrics and visualizations

_This project demonstrates state-of-the-art performance in DDoS traffic classification using advanced feature selection and Random Forest algorithms._
