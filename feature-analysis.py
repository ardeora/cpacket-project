import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Feature Analysis and Quality Assessment
print("=== STEP 2: Feature Analysis and Quality Assessment ===")

# Load the dataset
df = pd.read_parquet('datasets/ddos.parquet')

# Set up the analysis
label_column = 'label'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Analyzing {len(numeric_cols)} numeric features...")

# Check the 'activity' column
print("\n=== Activity Column Analysis ===")
print("Activity values:")
print(df['activity'].value_counts())
print("\nActivity vs Label relationship:")
print(pd.crosstab(df['activity'], df['label']))

# 1. Identify constant and near-constant features
print("\n=== Constant and Near-Constant Features ===")
constant_features = []
near_constant_features = []

for col in numeric_cols:
    unique_values = df[col].nunique()
    if unique_values == 1:
        constant_features.append(col)
    elif unique_values <= 2:
        near_constant_features.append(col)

print(f"Constant features (1 unique value): {len(constant_features)}")
if constant_features:
    print(f"First 10 constant features: {constant_features[:10]}")

print(f"Near-constant features (≤2 unique values): {len(near_constant_features)}")
if near_constant_features:
    print(f"First 10 near-constant features: {near_constant_features[:10]}")

# 2. Basic statistics by class
print("\n=== Feature Statistics by Class ===")
feature_stats = []

for col in numeric_cols[:5]:  # First 5 features as example
    stats_by_class = df.groupby(label_column)[col].agg(['mean', 'std', 'min', 'max'])
    print(f"\n{col}:")
    print(stats_by_class.round(4))
    
    # Calculate variance between classes
    class_means = df.groupby(label_column)[col].mean()
    between_class_var = class_means.var()
    within_class_var = df.groupby(label_column)[col].std().mean()
    
    feature_stats.append({
        'feature': col,
        'between_class_variance': between_class_var,
        'within_class_variance': within_class_var,
        'separation_ratio': between_class_var / (within_class_var + 1e-8)
    })

# 3. Identify features with high variance across classes
print("\n=== Features with High Class Separation (first 5 analyzed) ===")
stats_df = pd.DataFrame(feature_stats)
stats_df = stats_df.sort_values('separation_ratio', ascending=False)
print(stats_df)

# 4. Check for highly correlated features (sample)
print("\n=== High Correlation Analysis (sample of first 20 features) ===")
sample_cols = numeric_cols[:20]
corr_matrix = df[sample_cols].corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(sample_cols)):
    for j in range(i+1, len(sample_cols)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val > 0.9:
            high_corr_pairs.append((sample_cols[i], sample_cols[j], corr_val))

print(f"Highly correlated pairs (|r| > 0.9) in sample: {len(high_corr_pairs)}")
for pair in high_corr_pairs[:5]:
    print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

# 5. Zero variance features
print("\n=== Zero Variance Features ===")
zero_var_features = []
for col in numeric_cols:
    if df[col].var() == 0:
        zero_var_features.append(col)

print(f"Zero variance features: {len(zero_var_features)}")
if zero_var_features:
    print(f"First 10: {zero_var_features[:10]}")

# 6. Feature value ranges
print("\n=== Feature Value Ranges (sample) ===")
range_stats = []
for col in numeric_cols[:10]:
    col_min = df[col].min()
    col_max = df[col].max()
    col_range = col_max - col_min
    range_stats.append({
        'feature': col,
        'min': col_min,
        'max': col_max,
        'range': col_range,
        'has_negative': col_min < 0
    })

range_df = pd.DataFrame(range_stats)
print(range_df)

# 7. Summary for next steps
features_to_remove = list(set(constant_features + zero_var_features))
print(f"\n=== Summary ===")
print(f"Total numeric features: {len(numeric_cols)}")
print(f"Features to potentially remove (constant/zero variance): {len(features_to_remove)}")
print(f"Features remaining for analysis: {len(numeric_cols) - len(features_to_remove)}")

print(f"\nClass distribution reminder:")
print(df[label_column].value_counts())

print(f"\nNext steps will include:")
print("1. Remove constant/zero variance features")
print("2. Calculate feature importance using Random Forest")
print("3. Calculate mutual information scores")
print("4. Analyze correlations across all features")
print("5. Create final feature ranking")