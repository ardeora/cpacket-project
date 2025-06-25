# TCP Flag Synthetic Data Generation

This project provides a comprehensive setup for generating synthetic network traffic data using CTGAN and TVAE models, specifically designed for TCP flag analysis and attack detection datasets.

## 🚀 Quick Start

### 1. Setup and Data Analysis

Run the initial data analysis and metadata setup:

```bash
python synthetic_data_setup.py
```

This will:

- Analyze your TCP flag dataset
- Detect column types and patterns
- Create metadata for CTGAN/TVAE training
- Generate comprehensive pattern analysis
- Create visualizations

### 2. Advanced Pattern Analysis

For detailed TCP flag and network traffic pattern analysis:

```bash
python tcp_pattern_analyzer.py
```

This provides:

- TCP flag co-occurrence analysis
- Network traffic pattern detection
- Attack signature identification
- Feature importance analysis
- Interactive visualizations

### 3. Train Synthetic Data Models

Train CTGAN and TVAE models:

```bash
python train_synthetic_models_improved.py
```

This will:

- Train both CTGAN and TVAE models
- Generate synthetic data
- Evaluate data quality
- Create comparison visualizations

## 📊 Generated Outputs

### Metadata and Configuration

- `synthetic_datasets/metadata/` - Metadata files for model training
- `synthetic_datasets/metadata/training_configs.json` - Model hyperparameters

### Pattern Analysis Results

- `synthetic_datasets/patterns/detected_patterns.json` - Comprehensive pattern analysis
- `synthetic_datasets/patterns/comprehensive_tcp_analysis.json` - TCP-specific patterns

### Trained Models

- `synthetic_datasets/models/ctgan_model.pkl` - Trained CTGAN model
- `synthetic_datasets/models/tvae_model.pkl` - Trained TVAE model

### Synthetic Data

- `synthetic_datasets/ctgan_synthetic_data.csv` - CTGAN generated data
- `synthetic_datasets/tvae_synthetic_data.csv` - TVAE generated data

### Visualizations

- `synthetic_datasets/visualizations/` - All analysis visualizations
- Distribution comparisons, correlation heatmaps, feature importance plots

### Evaluation Results

- `synthetic_datasets/ctgan_evaluation.json` - CTGAN quality metrics
- `synthetic_datasets/tvae_evaluation.json` - TVAE quality metrics

## 🔍 Key Features Detected

### TCP Flag Analysis

The system automatically identifies and analyzes:

1. **TCP Flag Features**:

   - SYN flag percentages and counts
   - RST flag percentages and counts
   - PSH flag percentages (forward/backward)
   - Flag co-occurrence patterns

2. **Network Traffic Features**:

   - Header byte statistics (min, max, mean, median, mode)
   - Inter-arrival time patterns
   - Packet rates (forward/backward)
   - Window size analysis
   - Directional flow patterns

3. **Attack Patterns**:
   - Feature importance for attack detection
   - Attack-specific statistical signatures
   - Pattern differences between attack and normal traffic

## 📈 Pattern Analysis Results

### High Correlations Detected

The analysis identifies features with correlations > 0.7:

- Timing features (IAT patterns)
- Header byte relationships
- Rate correlations
- Flag co-occurrences

### TCP Flag Relationships

- Flag co-occurrence patterns
- Mutual information between flags
- Attack-specific flag signatures
- Sequential flag patterns

### Network Traffic Patterns

- Timing distribution analysis
- Directional flow asymmetries
- Rate pattern anomalies
- Header size distributions

## 🛠 Customization

### Adjust Training Parameters

Edit the training configurations in:

```json
{
  "ctgan": {
    "epochs": 300,
    "batch_size": 500,
    "generator_dim": [256, 256],
    "discriminator_dim": [256, 256]
  },
  "tvae": {
    "epochs": 300,
    "batch_size": 500,
    "embedding_dim": 128,
    "compress_dims": [128, 128]
  }
}
```

### Modify Feature Categories

The system automatically categorizes features, but you can customize:

```python
feature_categories = {
    'tcp_flags': ['syn_flag_*', 'rst_flag_*', 'psh_flag_*'],
    'header_bytes': ['*header_bytes*'],
    'timing_features': ['*IAT*', '*time*'],
    'rate_features': ['*rate*'],
    'window_features': ['*win*'],
    'packet_counts': ['*count*', '*total*']
}
```

## 📊 Quality Evaluation

The system evaluates synthetic data quality through:

1. **Statistical Comparison**:

   - Mean, standard deviation, min/max differences
   - Distribution shape preservation
   - Correlation structure preservation

2. **Categorical Distribution**:

   - Attack type distribution fidelity
   - Label preservation accuracy

3. **Feature Relationships**:
   - TCP flag co-occurrence preservation
   - Network pattern consistency
   - Attack signature maintenance

## 🎯 Use Cases

### 1. Privacy-Preserving Data Sharing

Generate synthetic network traffic data that preserves statistical properties while protecting privacy.

### 2. Data Augmentation

Increase dataset size for training machine learning models on network security.

### 3. Testing and Validation

Create controlled test datasets with known attack patterns.

### 4. Research and Development

Generate data for algorithm development without accessing sensitive network logs.

## 🔧 Advanced Usage

### Custom Pattern Detection

Add your own pattern detection methods:

```python
def detect_custom_patterns(self):
    # Your custom analysis
    patterns = {}
    # Add to self.patterns
    return patterns
```

### Custom Metadata Creation

Override metadata detection:

```python
def create_custom_metadata(self):
    metadata = SingleTableMetadata()
    # Custom column definitions
    return metadata
```

### Model Fine-tuning

Adjust model parameters based on your specific needs:

```python
# For better categorical handling
ctgan = CTGANSynthesizer(
    metadata=metadata,
    epochs=500,  # More epochs for better quality
    batch_size=256,  # Smaller batches for stability
    discriminator_steps=5,  # More discriminator training
    verbose=True
)
```

## 📝 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or number of epochs
2. **Poor Quality**: Increase epochs or adjust network architecture
3. **Training Instability**: Reduce learning rates or adjust discriminator steps

### Performance Tips

1. **CUDA Support**: Enable GPU training for faster performance
2. **Data Preprocessing**: Remove highly correlated features if needed
3. **Hyperparameter Tuning**: Use the detected patterns to guide parameter selection

## 📚 Dependencies

```bash
pip install ctgan sdv pandas numpy matplotlib seaborn plotly scipy scikit-learn networkx
```

## 🤝 Contributing

Feel free to extend the analysis with:

- Additional pattern detection methods
- New visualization techniques
- Custom evaluation metrics
- Domain-specific analysis

## 📄 License

This project is designed for research and educational purposes. Please ensure compliance with your organization's data usage policies.
