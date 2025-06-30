import pandas as pd
import numpy as np
from pathlib import Path

script_path = Path(__file__).resolve().parent
dataset_path = script_path.parent / 'datasets'

# Load the data
df = pd.read_csv(f'{dataset_path}/attack_tcp_flag_osyn.csv')

print("=== Script 1 ===")
print("=== BASIC DATA INFO ===")
print(f"Dataset shape: {df.shape}")
print(f"Attack type: {df['activity'].unique()}")
print(f"Label: {df['label'].unique()}")

print("\n=== FEATURE STATISTICS ===")
# Get numerical columns (excluding label and activity)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Basic statistics
stats = df[numeric_cols].describe()
print(stats.round(4))

print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sum())

print("\n=== SAMPLE ROWS ===")
print(df.head(3))


print("\n=== Script 2 ===")

import matplotlib.pyplot as plt
import seaborn as sns

# Analyze feature distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("=== FEATURE RANGES AND DISTRIBUTIONS ===")
for col in numeric_cols[:10]:  # First 10 features
    values = df[col].dropna()
    print(f"\n{col}:")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
    print(f"  Mean: {values.mean():.4f}, Std: {values.std():.4f}")
    print(f"  Unique values: {values.nunique()}")
    
    # Check if mostly zeros
    zero_pct = (values == 0).mean() * 100
    print(f"  Zero percentage: {zero_pct:.2f}%")

print("\n=== CORRELATION WITH KEY FEATURES ===")
# Look at correlations between key features
key_features = ['total_header_bytes', 'packets_rate', 'syn_flag_percentage_in_total', 
                'rst_flag_percentage_in_total', 'packet_IAT_mean']

if all(col in df.columns for col in key_features):
    corr_matrix = df[key_features].corr()
    print(corr_matrix.round(3))


print("\n=== Script 3 ===")

# Analyze relationships between features for prompt engineering
print("=== FEATURE RELATIONSHIPS FOR OSYN ATTACK ===")

# Group related features
header_features = [col for col in df.columns if 'header' in col.lower()]
flag_features = [col for col in df.columns if 'flag' in col.lower()]
timing_features = [col for col in df.columns if 'IAT' in col or 'rate' in col]

print(f"Header-related features ({len(header_features)}): {header_features[:5]}...")
print(f"Flag-related features ({len(flag_features)}): {flag_features}")
print(f"Timing-related features ({len(timing_features)}): {timing_features[:5]}...")

# Analyze value patterns
print("\n=== VALUE PATTERNS ===")
for feature_group, features in [("Header", header_features), ("Flag", flag_features), ("Timing", timing_features[:5])]:
    print(f"\n{feature_group} Features:")
    for col in features[:3]:  # First 3 of each group
        if col in df.columns:
            values = df[col].dropna()
            print(f"  {col}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")