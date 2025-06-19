import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Step 4: Correlation Analysis and Final Recommendations
print("=== STEP 4: Correlation Analysis and Final Recommendations ===")

# Load the dataset and feature rankings
df = pd.read_parquet('datasets/ddos.parquet')
feature_rankings = pd.read_csv('feature_rankings_combined.csv')

# Get top features from our analysis
top_features = feature_rankings.head(50)['feature'].tolist()

print(f"Analyzing correlations among top 50 features...")

# Sample data for correlation analysis
sample_size = 10000
sample_idx = np.random.choice(len(df), sample_size, replace=False)
df_sample = df.iloc[sample_idx]

# Get correlation matrix for top features
X_top = df_sample[top_features]
corr_matrix = X_top.corr()

print(f"Correlation matrix shape: {corr_matrix.shape}")

# 1. Find highly correlated feature groups
print("\n=== Highly Correlated Feature Groups (|r| > 0.8) ===")

# Find pairs with high correlation
high_corr_pairs = []
for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val > 0.8:
            high_corr_pairs.append((top_features[i], top_features[j], corr_val))

print(f"Found {len(high_corr_pairs)} highly correlated pairs:")
high_corr_pairs_sorted = sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)

for i, (feat1, feat2, corr) in enumerate(high_corr_pairs_sorted[:15]):
    print(f"{i+1:2d}. {feat1[:30]:<30} <-> {feat2[:30]:<30} : {corr:.3f}")

# 2. Identify feature groups by correlation clustering
print("\n=== Feature Grouping by Correlation ===")

# Group features by common patterns
feature_groups = {
    'IAT_Features': [f for f in top_features if 'IAT' in f],
    'Header_Features': [f for f in top_features if 'header' in f],
    'Port_Features': [f for f in top_features if 'port' in f],
    'Flag_Features': [f for f in top_features if 'flag' in f],
    'Rate_Features': [f for f in top_features if 'rate' in f],
    'Delta_Features': [f for f in top_features if 'delta' in f],
    'Other_Features': []
}

# Classify remaining features
for feat in top_features:
    classified = False
    for group_name, group_features in feature_groups.items():
        if feat in group_features:
            classified = True
            break
    if not classified:
        feature_groups['Other_Features'].append(feat)

for group_name, group_features in feature_groups.items():
    if group_features:
        print(f"{group_name}: {len(group_features)} features")
        print(f"  {group_features[:5]}")  # Show first 5

# 3. Remove highly redundant features
print("\n=== Feature Redundancy Reduction ===")

# Strategy: For each highly correlated pair, keep the one with higher combined score
features_to_remove = set()
for feat1, feat2, corr in high_corr_pairs_sorted:
    if feat1 not in features_to_remove and feat2 not in features_to_remove:
        # Get combined scores
        score1 = feature_rankings[feature_rankings['feature'] == feat1]['combined_score'].iloc[0]
        score2 = feature_rankings[feature_rankings['feature'] == feat2]['combined_score'].iloc[0]
        
        # Remove the one with lower score
        if score1 > score2:
            features_to_remove.add(feat2)
            print(f"Removing {feat2[:40]:<40} (corr={corr:.3f}, lower score)")
        else:
            features_to_remove.add(feat1)
            print(f"Removing {feat1[:40]:<40} (corr={corr:.3f}, lower score)")

# 4. Create final feature sets
print(f"\n=== Final Feature Recommendations ===")

# Remove redundant features
reduced_features = [f for f in top_features if f not in features_to_remove]
print(f"Features after redundancy removal: {len(reduced_features)} (removed {len(features_to_remove)})")

# Create different sized feature sets
feature_sets = {
    'Essential_15': reduced_features[:15],
    'Core_25': reduced_features[:25],
    'Extended_40': reduced_features[:40],
    'Full_Reduced': reduced_features
}

for set_name, feature_set in feature_sets.items():
    print(f"\n{set_name} ({len(feature_set)} features):")
    for i, feat in enumerate(feature_set):
        score = feature_rankings[feature_rankings['feature'] == feat]['combined_score'].iloc[0]
        print(f"  {i+1:2d}. {feat:<35} (score: {score:.3f})")

# 5. Feature set analysis by category
print(f"\n=== Feature Set Composition Analysis ===")

for set_name, feature_set in feature_sets.items():
    composition = {}
    for group_name, group_features in feature_groups.items():
        count = len([f for f in feature_set if f in group_features])
        if count > 0:
            composition[group_name] = count
    
    print(f"\n{set_name} composition:")
    for category, count in composition.items():
        percentage = (count / len(feature_set)) * 100
        print(f"  {category}: {count} features ({percentage:.1f}%)")

# 6. Generate recommendations for synthetic data generation
print(f"\n=== RECOMMENDATIONS FOR SYNTHETIC DATA GENERATION ===")

print("""
RECOMMENDED APPROACH:

1. START WITH ESSENTIAL_15 SET:
   - These 15 features provide the best discrimination power
   - Good balance across feature types (IAT, headers, ports)
   - Minimal redundancy after correlation analysis

2. FOR BETTER QUALITY, USE CORE_25 SET:
   - Includes more nuanced timing and structural features
   - Better captures subtle differences between attack types
   - Still computationally manageable

3. FOR PRODUCTION USE:
   - Consider EXTENDED_40 set for maximum fidelity
   - Include 'activity' column as additional conditioning variable
   - The detailed activity labels (26 types) are extremely valuable

4. FEATURE ENGINEERING NOTES:
   - IAT features are critical - preserve timing relationships
   - Header byte statistics capture packet structure
   - Port features distinguish service-based attacks
   - Flag percentages indicate protocol behavior

5. SYNTHETIC DATA VALIDATION:
   - Ensure generated data maintains feature correlations
   - Validate timing relationships (IAT features)
   - Check port number distributions match realistic patterns
   - Verify flag combinations are protocol-compliant
""")

# 7. Save final recommendations
print(f"\n=== Saving Final Results ===")

# Save feature sets
for set_name, feature_set in feature_sets.items():
    filename = f"feature_set_{set_name.lower()}.txt"
    with open(filename, 'w') as f:
        f.write(f"# {set_name} Feature Set\n")
        f.write(f"# Generated from correlation analysis\n")
        f.write(f"# Total features: {len(feature_set)}\n\n")
        for feat in feature_set:
            f.write(f"{feat}\n")
    print(f"Saved {filename}")

# Save correlation analysis
correlation_summary = pd.DataFrame(high_corr_pairs_sorted, 
                                  columns=['Feature1', 'Feature2', 'Correlation'])
correlation_summary.to_csv('high_correlations.csv', index=False)
print("Saved high_correlations.csv")

# Save feature composition analysis
composition_data = []
for set_name, feature_set in feature_sets.items():
    for group_name, group_features in feature_groups.items():
        count = len([f for f in feature_set if f in group_features])
        if count > 0:
            composition_data.append({
                'feature_set': set_name,
                'category': group_name,
                'count': count,
                'percentage': (count / len(feature_set)) * 100
            })

composition_df = pd.DataFrame(composition_data)
composition_df.to_csv('feature_composition_analysis.csv', index=False)
print("Saved feature_composition_analysis.csv")

print(f"\n=== ANALYSIS COMPLETE ===")
print(f"Recommended starting point: Use ESSENTIAL_15 set with 'activity' column")
print(f"Files generated: feature_set_*.txt, high_correlations.csv, feature_composition_analysis.csv")