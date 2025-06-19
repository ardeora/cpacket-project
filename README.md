# DDoS Network Traffic Feature Analysis Project

## Overview

This project performs comprehensive feature analysis on network traffic data to identify the most important features for DDoS attack detection and classification. The analysis includes feature importance ranking, correlation analysis, and provides curated feature sets optimized for synthetic data generation and machine learning models.

## Dataset

- **Source**: DDoS network traffic dataset (`datasets/ddos.parquet`)
- **Format**: Parquet file containing network flow features
- **Labels**:
  - 3-class labels: benign/suspicious/attack
  - Detailed activity labels: 26 different attack types and benign traffic
- **Features**: 300+ numeric features including timing, header, port, and flag statistics

## Project Structure

```
├── basic-exploration.py          # Initial dataset exploration and basic statistics
├── feature-analysis.py           # Feature quality assessment and class separation analysis
├── feature-importance.py         # Random Forest and Mutual Information scoring
├── feature-recommendation.py     # Correlation analysis and final feature set creation
├── datasets/
│   └── ddos.parquet              # Network traffic dataset
├── output/                       # Generated analysis results
│   ├── feature_set_essential_15.txt    # Top 15 features (recommended starting point)
│   ├── feature_set_core_25.txt         # Core 25 features for better quality
│   ├── feature_set_extended_40.txt     # Extended 40 features for production
│   ├── feature_set_full_reduced.txt    # All features after redundancy removal
│   ├── feature_rankings_combined.csv   # Complete feature rankings with scores
│   ├── feature_importance_*.csv        # Individual importance scores
│   ├── mutual_info_*.csv               # Mutual information scores
│   ├── high_correlations.csv           # Highly correlated feature pairs
│   └── feature_composition_analysis.csv # Feature set composition by category
└── cpacket/                      # Python virtual environment
```

## Analysis Pipeline

### 1. Basic Exploration (`basic-exploration.py`)

- Dataset loading and basic statistics
- Missing value analysis
- Label distribution examination
- Data type assessment

### 2. Feature Analysis (`feature-analysis.py`)

- Identification of constant and near-constant features
- Class separation analysis
- Zero variance feature detection
- Feature value range analysis

### 3. Feature Importance (`feature-importance.py`)

- **Random Forest Feature Importance**: Calculated for both 3-class labels and detailed activity classification
- **Mutual Information Scores**: Measures statistical dependency between features and labels
- **Combined Ranking**: Weighted combination of all importance metrics
- **Sampling**: Uses 50,000 samples for computational efficiency

### 4. Feature Recommendation (`feature-recommendation.py`)

- **Correlation Analysis**: Identifies highly correlated feature pairs (|r| > 0.8)
- **Feature Grouping**: Categorizes features by type (IAT, Header, Port, Flag, Rate, etc.)
- **Redundancy Removal**: Eliminates highly correlated features while preserving importance
- **Feature Set Creation**: Generates optimized feature sets of different sizes

## Key Findings

### Top Feature Categories

1. **IAT (Inter-Arrival Time) Features**: Critical for timing pattern detection
2. **Header Byte Statistics**: Capture packet structure characteristics
3. **Port Features**: Distinguish service-based attacks
4. **Flag Percentages**: Indicate protocol behavior patterns
5. **Rate Features**: Measure traffic intensity

### Recommended Feature Sets

#### Essential 15 Features (Recommended Starting Point)

```
fwd_mode_header_bytes, fwd_packets_IAT_total, packet_IAT_max,
src_port, dst_port, fwd_init_win_bytes, fwd_total_header_bytes,
total_header_bytes, median_packets_delta_time, syn_flag_percentage_in_total,
packets_rate, duration, bwd_packets_IAT_total, median_packets_delta_len,
ack_flag_percentage_in_total
```

#### Core 25 Features (Better Quality)

- Includes additional timing and structural features
- Better captures subtle attack type differences
- Still computationally manageable

#### Extended 40 Features (Production Use)

- Maximum fidelity for complex attack detection
- Recommended for production deployment
- Includes comprehensive feature coverage

## Usage

### Prerequisites

```bash
# Create virtual environment (already included in project)
python -m venv cpacket
source cpacket/bin/activate  # On macOS/Linux

# Install required packages
pip install -r requirements.txt

# Or install manually:
# pip install pandas numpy matplotlib seaborn scikit-learn pyarrow
```

### Running the Analysis

```bash
# Run complete analysis pipeline
python basic-exploration.py
python feature-analysis.py
python feature-importance.py
python feature-recommendation.py
```

### Loading Feature Sets

```python
# Load recommended feature set
with open('output/feature_set_essential_15.txt', 'r') as f:
    essential_features = [line.strip() for line in f if not line.startswith('#') and line.strip()]

# Load feature rankings
rankings = pd.read_csv('output/feature_rankings_combined.csv')
```

## Synthetic Data Generation Recommendations

### Key Considerations

1. **Preserve Timing Relationships**: IAT features are critical - maintain realistic timing patterns
2. **Maintain Feature Correlations**: Use correlation matrix to preserve feature dependencies
3. **Validate Port Distributions**: Ensure generated port numbers follow realistic patterns
4. **Protocol Compliance**: Verify flag combinations are protocol-compliant

### Recommended Approach

1. **Start with Essential_15**: Provides best discrimination with minimal complexity
2. **Include Activity Column**: Use detailed activity labels (26 types) as conditioning variable
3. **Validate Generated Data**: Ensure statistical properties match original dataset
4. **Progressive Enhancement**: Move to Core_25 or Extended_40 for improved quality

## Output Files Description

| File                               | Description                                      |
| ---------------------------------- | ------------------------------------------------ | --- | ------ |
| `feature_set_*.txt`                | Curated feature sets of different sizes          |
| `feature_rankings_combined.csv`    | Complete feature rankings with normalized scores |
| `feature_importance_*.csv`         | Random Forest importance scores                  |
| `mutual_info_*.csv`                | Mutual information scores for different targets  |
| `high_correlations.csv`            | Pairs of highly correlated features (            | r   | > 0.8) |
| `feature_composition_analysis.csv` | Breakdown of feature sets by category            |

## Technical Details

- **Algorithm**: Random Forest (100 estimators) + Mutual Information
- **Sampling**: 50,000 samples for computational efficiency
- **Correlation Threshold**: 0.8 for redundancy removal
- **Scoring Weights**: 30% RF Label + 30% RF Activity + 20% MI Label + 20% MI Activity
- **Missing Values**: None detected in the dataset

## Next Steps

1. **Model Training**: Use recommended feature sets to train DDoS detection models
2. **Synthetic Data Generation**: Apply findings to generate realistic network traffic data
3. **Real-time Detection**: Implement feature extraction pipeline for live traffic analysis
4. **Performance Optimization**: Fine-tune feature sets based on specific use case requirements

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pyarrow >= 5.0.0 (for parquet files)

---

_This analysis provides a data-driven approach to feature selection for network security applications, with particular focus on DDoS attack detection and synthetic data generation._
