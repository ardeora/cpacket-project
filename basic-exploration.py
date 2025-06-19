import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the parquet file and basic exploration
print("=== STEP 1: Loading and Basic Exploration ===")

# Load the dataset
df = pd.read_parquet('datasets/ddos.parquet')  # Replace with your actual file path

print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Display basic info
print("\n=== Basic Dataset Info ===")
print(df.info())

print("\n=== First few rows ===")
print(df.head())

print("\n=== Column names (first 20) ===")
print(df.columns.tolist()[:20])

print("\n=== Data types summary ===")
print(df.dtypes.value_counts())

# Check for label column - common names for labels in network data
possible_label_columns = ['label', 'Label', 'class', 'Class', 'target', 'Target', 
                         'attack_type', 'category', 'Category']

label_column = None
for col in possible_label_columns:
    if col in df.columns:
        label_column = col
        break

if label_column:
    print(f"\n=== Label Column Found: '{label_column}' ===")
    print("Label distribution:")
    print(df[label_column].value_counts())
    print("\nLabel percentages:")
    print(df[label_column].value_counts(normalize=True) * 100)
else:
    print("\n=== Label column not automatically detected ===")
    print("Please specify which column contains the labels (benign/suspicious/attack)")
    print("Available columns:")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

print("\n=== Missing Values Summary ===")
missing_summary = df.isnull().sum()
missing_cols = missing_summary[missing_summary > 0]
if len(missing_cols) > 0:
    print(f"Columns with missing values: {len(missing_cols)}")
    print(missing_cols.head(10))
else:
    print("No missing values found!")

print("\n=== Sample of Numeric vs Non-Numeric Columns ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Non-numeric columns: {len(non_numeric_cols)}")
print(f"Non-numeric columns: {non_numeric_cols.tolist()}")