# Dataset Analysis Tools

This project contains comprehensive tools for analyzing the `attack-tcp-flag-osyn.csv` dataset. The analysis includes statistical summaries, visualizations, and detailed unique value exploration presented in a beautiful web interface.

## 📁 Files Generated

- `analyze_dataset.py` - Main analysis script that generates a comprehensive webpage
- `quick_column_analyzer.py` - Quick command-line tool to analyze specific columns
- `dataset_analysis_report.html` - Beautiful webpage with complete analysis results
- `dataset_analysis_report_data.json` - Raw analysis data in JSON format

## 🚀 Usage

### 1. Generate Complete Analysis Report

Run the main analysis script to generate a comprehensive webpage:

```bash
python analyze_dataset.py
```

This will create:
- `dataset_analysis_report.html` - Interactive webpage with all analysis results
- `dataset_analysis_report_data.json` - Raw data for further processing

### 2. Quick Column Analysis

Use the quick analyzer to examine specific columns:

```bash
# List all available columns
python quick_column_analyzer.py

# Analyze a specific column
python quick_column_analyzer.py column_name
```

Examples:
```bash
python quick_column_analyzer.py label
python quick_column_analyzer.py max_header_bytes
python quick_column_analyzer.py packets_rate
```

## 📊 Analysis Features

### Comprehensive Web Report

The main analysis generates a beautiful webpage containing:

1. **Dataset Overview**
   - Total rows and columns
   - Memory usage
   - Missing values count

2. **Data Visualizations**
   - Label distribution charts
   - Activity type distributions
   - Correlation heatmaps
   - Feature distribution plots

3. **Unique Values Analysis**
   - Interactive accordion view for each column
   - Complete list of unique values (or sample for large datasets)
   - Value counts and percentages
   - Data type information

4. **Statistical Summary**
   - Descriptive statistics for numerical columns
   - Mean, median, standard deviation, quartiles
   - Min and max values

5. **Categorical Analysis**
   - Value counts for categorical features
   - Most and least common values

### Quick Column Analyzer Features

- **Column Listing**: Shows all 42 columns with their unique value counts
- **Detailed Analysis**: For any specific column shows:
  - Data type and null value information
  - Complete list of unique values with counts and percentages
  - Statistical summary (for numerical columns)

## 📈 Dataset Summary

The `attack-tcp-flag-osyn.csv` dataset contains:

- **757 rows** of network traffic data
- **42 columns** of features
- **0.33 MB** memory usage
- **All records labeled as "Attack"** with activity type "Attack-TCP-Flag-OSYN"

### Key Columns

- `label`: Classification label (all "Attack")
- `activity`: Specific attack type (all "Attack-TCP-Flag-OSYN")
- `max_header_bytes`: Header size information (5 unique values: 20, 24, 28, 32, 40)
- `packets_rate`: Network packet rate (216 unique values)
- Various IAT (Inter-Arrival Time) features
- TCP flag percentages and counts
- Forward and backward packet statistics

## 🔍 Example Analysis Results

### Label Distribution
- **Attack**: 757 records (100%)

### Activity Distribution  
- **Attack-TCP-Flag-OSYN**: 757 records (100%)

### Max Header Bytes Distribution
- **20 bytes**: 439 records (58.0%)
- **24 bytes**: 169 records (22.3%)
- **40 bytes**: 91 records (12.0%)
- **32 bytes**: 52 records (6.9%)
- **28 bytes**: 6 records (0.8%)

## 🛠️ Technical Requirements

The analysis uses the following Python packages:
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (plotting)
- seaborn (statistical visualizations)
- pathlib (file operations)
- json (data serialization)
- base64 (image encoding for web)

## 📱 Web Interface Features

The generated webpage includes:
- **Responsive design** that works on desktop and mobile
- **Interactive accordion sections** for exploring unique values
- **Beautiful visualizations** with high-quality plots
- **Professional styling** with gradients and shadows
- **Searchable and sortable tables**
- **Timestamp** showing when the analysis was performed

## 🔄 Updating the Analysis

To regenerate the analysis with updated data:

1. Replace the CSV file with new data
2. Run `python analyze_dataset.py`
3. Open the new `dataset_analysis_report.html` in your browser

The analysis automatically adapts to different dataset structures and sizes.

---

**Generated on**: Analysis timestamp is included in the webpage
**Dataset**: attack-tcp-flag-osyn.csv (757 records, 42 features)
**Tools**: Python-based analysis with web visualization
