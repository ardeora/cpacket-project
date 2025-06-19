import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Step 3: Feature Importance Analysis
print("=== STEP 3: Feature Importance Analysis ===")

# Load the dataset
df = pd.read_parquet('datasets/ddos.parquet')

# Prepare the data
label_column = 'label'
activity_column = 'activity'

# Remove constant and zero variance features identified in Step 2
features_to_remove = ['bwd_urg_flag_counts', 'bwd_urg_flag_percentage_in_total', 
                      'bwd_urg_flag_percentage_in_bwd_packets']

print(f"Removing {len(features_to_remove)} constant/zero variance features...")

# Get numeric columns excluding the ones to remove
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
clean_numeric_cols = [col for col in numeric_cols if col not in features_to_remove]

print(f"Features for analysis: {len(clean_numeric_cols)}")

# Prepare features and targets
X = df[clean_numeric_cols]
y_label = df[label_column]
y_activity = df[activity_column]

# Encode labels for sklearn
le_label = LabelEncoder()
le_activity = LabelEncoder()
y_label_encoded = le_label.fit_transform(y_label)
y_activity_encoded = le_activity.fit_transform(y_activity)

print(f"Label classes: {le_label.classes_}")
print(f"Activity classes: {len(le_activity.classes_)} different activities")

# Sample data for faster computation (if dataset is large)
if len(df) > 50000:
    sample_size = 50000
    print(f"\nSampling {sample_size} rows for faster computation...")
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    X_sample = X.iloc[sample_idx]
    y_label_sample = y_label_encoded[sample_idx]
    y_activity_sample = y_activity_encoded[sample_idx]
else:
    X_sample = X
    y_label_sample = y_label_encoded
    y_activity_sample = y_activity_encoded

print(f"Using {len(X_sample)} samples for analysis")

# 1. Random Forest Feature Importance for 3-class labels
print("\n=== Random Forest Feature Importance (3-class labels) ===")
rf_label = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_label.fit(X_sample, y_label_sample)

# Get feature importance
importance_label = pd.DataFrame({
    'feature': clean_numeric_cols,
    'importance': rf_label.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 most important features for 3-class classification:")
print(importance_label.head(15))

# 2. Random Forest Feature Importance for detailed activity labels
print("\n=== Random Forest Feature Importance (Activity labels) ===")
rf_activity = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_activity.fit(X_sample, y_activity_sample)

# Get feature importance
importance_activity = pd.DataFrame({
    'feature': clean_numeric_cols,
    'importance': rf_activity.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 most important features for activity classification:")
print(importance_activity.head(15))

# 3. Mutual Information for 3-class labels
print("\n=== Mutual Information Scores (3-class labels) ===")
mi_scores_label = mutual_info_classif(X_sample, y_label_sample, random_state=42)

mi_label = pd.DataFrame({
    'feature': clean_numeric_cols,
    'mutual_info': mi_scores_label
}).sort_values('mutual_info', ascending=False)

print("Top 15 features by mutual information (3-class):")
print(mi_label.head(15))

# 4. Mutual Information for activity labels
print("\n=== Mutual Information Scores (Activity labels) ===")
mi_scores_activity = mutual_info_classif(X_sample, y_activity_sample, random_state=42)

mi_activity = pd.DataFrame({
    'feature': clean_numeric_cols,
    'mutual_info': mi_scores_activity
}).sort_values('mutual_info', ascending=False)

print("Top 15 features by mutual information (activity):")
print(mi_activity.head(15))

# 5. Combined ranking
print("\n=== Combined Feature Ranking ===")

# Normalize scores to 0-1 range for comparison
importance_label['rf_label_norm'] = importance_label['importance'] / importance_label['importance'].max()
importance_activity['rf_activity_norm'] = importance_activity['importance'] / importance_activity['importance'].max()
mi_label['mi_label_norm'] = mi_label['mutual_info'] / mi_label['mutual_info'].max()
mi_activity['mi_activity_norm'] = mi_activity['mutual_info'] / mi_activity['mutual_info'].max()

# Merge all rankings
combined = pd.DataFrame({'feature': clean_numeric_cols})
combined = combined.merge(importance_label[['feature', 'rf_label_norm']], on='feature')
combined = combined.merge(importance_activity[['feature', 'rf_activity_norm']], on='feature')
combined = combined.merge(mi_label[['feature', 'mi_label_norm']], on='feature')
combined = combined.merge(mi_activity[['feature', 'mi_activity_norm']], on='feature')

# Calculate combined score (you can adjust weights)
combined['combined_score'] = (
    0.3 * combined['rf_label_norm'] + 
    0.3 * combined['rf_activity_norm'] +
    0.2 * combined['mi_label_norm'] + 
    0.2 * combined['mi_activity_norm']
)

combined = combined.sort_values('combined_score', ascending=False)

print("Top 20 features by combined ranking:")
print(combined[['feature', 'combined_score', 'rf_label_norm', 'rf_activity_norm']].head(20))

# 6. Save results for next step
print("\n=== Saving Results ===")
print("Saving feature rankings to CSV files...")
combined.to_csv('feature_rankings_combined.csv', index=False)
importance_label.to_csv('feature_importance_label.csv', index=False)
importance_activity.to_csv('feature_importance_activity.csv', index=False)
mi_label.to_csv('mutual_info_label.csv', index=False)
mi_activity.to_csv('mutual_info_activity.csv', index=False)

print("Files saved successfully!")

# 7. Summary statistics
print(f"\n=== Summary ===")
print(f"Total features analyzed: {len(clean_numeric_cols)}")
print(f"Features with RF importance > 0.001 (label): {len(importance_label[importance_label['importance'] > 0.001])}")
print(f"Features with RF importance > 0.001 (activity): {len(importance_activity[importance_activity['importance'] > 0.001])}")
print(f"Features with MI score > 0.1 (label): {len(mi_label[mi_label['mutual_info'] > 0.1])}")
print(f"Features with MI score > 0.1 (activity): {len(mi_activity[mi_activity['mutual_info'] > 0.1])}")

# Identify top features for synthetic data generation
top_features = combined.head(50)['feature'].tolist()
print(f"\nRecommended top 50 features for synthetic data generation:")
print(top_features)