# DDoS Classifier Evaluation

This document explains how to use the evaluation script to test the saved DDoS classifier models on new CSV data.

## Overview

The evaluation script (`evaluate_classifier.py`) provides a comprehensive way to test trained DDoS classifier models on new datasets. It supports command-line usage and programmatic access, with detailed prediction analysis and error reporting.

## Quick Start

```bash
# Activate the virtual environment
source cpacket/bin/activate

# Basic evaluation on the test dataset
python evaluate_classifier.py datasets/attack-tcp-flag-osyn.csv

# Show all predictions with detailed output
python evaluate_classifier.py datasets/attack-tcp-flag-osyn.csv --show-all --max-display 50
```

## Usage

### Command Line Interface

The main evaluation script can be run from the command line with various options:

```bash
# Basic usage - evaluate on attack dataset using extra_tree model (default)
python evaluate_classifier.py datasets/attack-tcp-flag-osyn.csv

# Show all predictions (not just errors)
python evaluate_classifier.py datasets/attack-tcp-flag-osyn.csv --show-all

# Use a different feature selection method
python evaluate_classifier.py datasets/attack-tcp-flag-osyn.csv --method anova

# Limit number of features and display results
python evaluate_classifier.py datasets/attack-tcp-flag-osyn.csv --max-features 20 --max-display 15

# Show help
python evaluate_classifier.py --help
```

### Command Line Options

- `csv_file` - Path to the CSV file to evaluate (required)
- `--method` / `-m` - Feature selection method (default: extra_tree)
- `--max-features` / `-f` - Maximum number of features to use (default: 40)
- `--show-all` / `-a` - Show all predictions, not just incorrect ones
- `--max-display` / `-d` - Maximum number of predictions to display (default: 20)

### Programmatic Usage

You can also use the `ClassifierEvaluator` class directly in your Python code:

```python
from evaluate_classifier import ClassifierEvaluator

# Create evaluator
evaluator = ClassifierEvaluator(model_method='extra_tree')

# Evaluate on CSV file
results = evaluator.evaluate_and_display(
    csv_path='datasets/attack-tcp-flag-osyn.csv',
    max_features=40,
    show_all_predictions=False,
    max_display=10
)

# Access results DataFrame
print(f"Accuracy: {results['Correct'].mean():.4f}")
```

## CSV File Requirements

The CSV file must:

1. **Include a `label` column** with the true class labels
2. **Include feature columns** that match the features used during training
3. **Optionally include an `activity` column** for more detailed display

## Output

The evaluation script provides:

1. **Overall accuracy** and detailed classification metrics
2. **Individual predictions** showing:

   - Row number
   - Predicted label
   - Actual label
   - Prediction confidence
   - Whether the prediction was correct
   - Activity type (if available)

3. **Summary statistics** including:

   - Total correct/incorrect predictions
   - Confusion matrix

4. **Filtering options**:
   - Show all predictions or only errors
   - Limit number of rows displayed

## Available Models

The script can load any of the saved classifier models from the `saved_models/` directory:

- `extra_tree` - Extra Trees feature selection (default, best performance)
- `anova` - ANOVA F-test feature selection
- `information_gain` - Information Gain feature selection

## Project Integration

The evaluation script integrates seamlessly with the main project structure:

```
cpacket-project/
├── evaluate_classifier.py    # This evaluation script
├── saved_models/            # Pre-trained models (automatically loaded)
├── features/                # Feature selection configs (automatically used)
└── datasets/                # Test datasets
    ├── ddos.parquet         # Original training data
    └── attack-tcp-flag-osyn.csv # Test/evaluation data
```

The script automatically:

- Loads models from `saved_models/`
- Uses feature configurations from `features/`
- Handles missing features gracefully
- Provides detailed error analysis

## Example Output

```
============================================================
EVALUATION RESULTS
============================================================
Total samples: 757
Features used: 40
Overall Accuracy: 0.8388 (83.88%)

============================================================
INDIVIDUAL PREDICTIONS
============================================================
Showing 10 incorrect predictions (out of 122 total errors):

Row   Predicted       Actual          Confidence   Status     Activity
-------------------------------------------------------------------------------------
12    Benign          Attack          0.8578       ✗ Wrong    Attack-TCP-Flag-OSYN
20    Benign          Attack          0.5742       ✗ Wrong    Attack-TCP-Flag-OSYN
...

============================================================
SUMMARY
============================================================
✓ Correct predictions: 635/757 (83.88%)
✗ Incorrect predictions: 122/757 (16.12%)
```

## Notes

- **Virtual Environment**: Make sure to activate the virtual environment with `source cpacket/bin/activate` before running the script
- **Feature Handling**: The script automatically handles missing features by using only available ones
- **Confidence Scores**: Represent the maximum probability assigned by the classifier
- **Confusion Matrix**: Shows detailed breakdown of predictions vs actual labels
- **Warning Messages**: About undefined metrics are normal when some classes are not present in the test data
- **Performance**: Best results typically achieved with the `extra_tree` method (94.98% accuracy)
