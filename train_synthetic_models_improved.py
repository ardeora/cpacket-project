import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    from ctgan import CTGAN, TVAE
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    from sdv.evaluation.single_table import evaluate_quality
    print("Successfully imported CTGAN and SDV libraries")
except ImportError as e:
    print(f"Error importing SDV/CTGAN: {e}")
    print("Please install with: pip install ctgan sdv")
    exit(1)


def create_metadata_from_dataframe(df):
    """Create metadata manually from dataframe."""
    print("Creating metadata from dataframe...")
    
    metadata = SingleTableMetadata()
    
    # Define column types based on analysis
    categorical_cols = ['label', 'activity']
    
    for col in df.columns:
        if col in categorical_cols:
            metadata.add_column(column_name=col, sdtype='categorical')
        else:
            metadata.add_column(column_name=col, sdtype='numerical')
    
    return metadata


def train_ctgan_model(df, metadata, output_dir, epochs=200):
    """Train CTGAN model."""
    print(f"\nTraining CTGAN with {epochs} epochs...")
    
    # Initialize CTGAN
    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=min(500, len(df)//2),  # Adjust batch size based on data size
        verbose=True,
        cuda=False  # Set to True if you have CUDA available
    )
    
    # Train the model
    ctgan.fit(df)
    
    # Save model
    model_path = Path(output_dir) / 'models' / 'ctgan_model.pkl'
    model_path.parent.mkdir(exist_ok=True)
    ctgan.save(model_path)
    print(f"CTGAN model saved to: {model_path}")
    
    return ctgan


def train_tvae_model(df, metadata, output_dir, epochs=200):
    """Train TVAE model."""
    print(f"\nTraining TVAE with {epochs} epochs...")
    
    # Initialize TVAE
    tvae = TVAESynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=min(500, len(df)//2),
        verbose=True,
        cuda=False  # Set to True if you have CUDA available
    )
    
    # Train the model
    tvae.fit(df)
    
    # Save model
    model_path = Path(output_dir) / 'models' / 'tvae_model.pkl'
    model_path.parent.mkdir(exist_ok=True)
    tvae.save(model_path)
    print(f"TVAE model saved to: {model_path}")
    
    return tvae


def generate_synthetic_data(model, model_name, num_rows, output_dir):
    """Generate synthetic data using trained model."""
    print(f"\nGenerating {num_rows} synthetic samples with {model_name}...")
    
    synthetic_data = model.sample(num_rows=num_rows)
    
    # Save synthetic data
    output_path = Path(output_dir) / f'{model_name.lower()}_synthetic_data.csv'
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")
    
    return synthetic_data


def evaluate_synthetic_data(real_data, synthetic_data, model_name, output_dir):
    """Evaluate the quality of synthetic data."""
    print(f"\nEvaluating {model_name} synthetic data quality...")
    
    evaluation_results = {}
    
    # Basic statistics comparison
    print("Comparing basic statistics...")
    stats_comparison = {}
    
    for col in real_data.select_dtypes(include=[np.number]).columns:
        real_stats = {
            'mean': real_data[col].mean(),
            'std': real_data[col].std(),
            'min': real_data[col].min(),
            'max': real_data[col].max(),
            'median': real_data[col].median()
        }
        
        synthetic_stats = {
            'mean': synthetic_data[col].mean(),
            'std': synthetic_data[col].std(),
            'min': synthetic_data[col].min(),
            'max': synthetic_data[col].max(),
            'median': synthetic_data[col].median()
        }
        
        stats_comparison[col] = {
            'real': real_stats,
            'synthetic': synthetic_stats,
            'mean_diff': abs(real_stats['mean'] - synthetic_stats['mean']),
            'std_diff': abs(real_stats['std'] - synthetic_stats['std'])
        }
    
    evaluation_results['statistics_comparison'] = stats_comparison
    
    # Categorical distribution comparison
    print("Comparing categorical distributions...")
    categorical_comparison = {}
    
    for col in real_data.select_dtypes(include=['object']).columns:
        real_dist = real_data[col].value_counts(normalize=True).to_dict()
        synthetic_dist = synthetic_data[col].value_counts(normalize=True).to_dict()
        
        categorical_comparison[col] = {
            'real_distribution': real_dist,
            'synthetic_distribution': synthetic_dist
        }
    
    evaluation_results['categorical_comparison'] = categorical_comparison
    
    # Save evaluation results
    eval_path = Path(output_dir) / f'{model_name.lower()}_evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"Evaluation results saved to: {eval_path}")
    
    return evaluation_results


def create_comparison_visualizations(real_data, ctgan_data, tvae_data, output_dir):
    """Create visualizations comparing real and synthetic data."""
    print("\nCreating comparison visualizations...")
    
    viz_dir = Path(output_dir) / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Select a few key numerical features for visualization
    numerical_cols = real_data.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
    
    # Distribution comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            axes[i].hist(real_data[col], alpha=0.5, label='Real', bins=30, density=True)
            axes[i].hist(ctgan_data[col], alpha=0.5, label='CTGAN', bins=30, density=True)
            axes[i].hist(tvae_data[col], alpha=0.5, label='TVAE', bins=30, density=True)
            axes[i].set_title(f'{col} Distribution')
            axes[i].legend()
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Categorical comparison
    categorical_cols = real_data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(6*len(categorical_cols), 6))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(categorical_cols):
            real_counts = real_data[col].value_counts()
            ctgan_counts = ctgan_data[col].value_counts()
            tvae_counts = tvae_data[col].value_counts()
            
            # Combine all categories
            all_categories = set(real_counts.index) | set(ctgan_counts.index) | set(tvae_counts.index)
            
            x_pos = np.arange(len(all_categories))
            width = 0.25
            
            real_vals = [real_counts.get(cat, 0) for cat in all_categories]
            ctgan_vals = [ctgan_counts.get(cat, 0) for cat in all_categories]
            tvae_vals = [tvae_counts.get(cat, 0) for cat in all_categories]
            
            axes[i].bar(x_pos - width, real_vals, width, label='Real', alpha=0.8)
            axes[i].bar(x_pos, ctgan_vals, width, label='CTGAN', alpha=0.8)
            axes[i].bar(x_pos + width, tvae_vals, width, label='TVAE', alpha=0.8)
            
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(all_categories, rotation=45)
            axes[i].legend()
            axes[i].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'categorical_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {viz_dir}")


def main():
    """Main training function."""
    print("="*60)
    print("CTGAN & TVAE TRAINING FOR TCP FLAG DATASET")
    print("="*60)
    
    # Configuration
    data_path = "datasets/attack-tcp-flag-osyn.csv"
    output_dir = "synthetic_datasets/"
    epochs = 150  # Reduced for faster training, increase for better quality
    
    # Load data
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Dataset columns: {list(df.columns)}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Missing values: {missing_values}")
    
    # Create metadata
    metadata = create_metadata_from_dataframe(df)
    print("Metadata created successfully")
    
    # Train CTGAN
    try:
        ctgan_model = train_ctgan_model(df, metadata, output_dir, epochs)
        ctgan_synthetic = generate_synthetic_data(ctgan_model, "CTGAN", len(df), output_dir)
        ctgan_evaluation = evaluate_synthetic_data(df, ctgan_synthetic, "CTGAN", output_dir)
        print("✓ CTGAN training completed successfully")
    except Exception as e:
        print(f"✗ CTGAN training failed: {e}")
        ctgan_synthetic = None
    
    # Train TVAE
    try:
        tvae_model = train_tvae_model(df, metadata, output_dir, epochs)
        tvae_synthetic = generate_synthetic_data(tvae_model, "TVAE", len(df), output_dir)
        tvae_evaluation = evaluate_synthetic_data(df, tvae_synthetic, "TVAE", output_dir)
        print("✓ TVAE training completed successfully")
    except Exception as e:
        print(f"✗ TVAE training failed: {e}")
        tvae_synthetic = None
    
    # Create comparison visualizations
    if ctgan_synthetic is not None and tvae_synthetic is not None:
        create_comparison_visualizations(df, ctgan_synthetic, tvae_synthetic, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Original dataset: {df.shape[0]} rows")
    if ctgan_synthetic is not None:
        print(f"CTGAN synthetic data: {ctgan_synthetic.shape[0]} rows")
    if tvae_synthetic is not None:
        print(f"TVAE synthetic data: {tvae_synthetic.shape[0]} rows")
    print(f"All outputs saved to: {output_dir}")
    
    print("\nNext steps:")
    print("1. Review the evaluation results in synthetic_datasets/")
    print("2. Check the distribution comparison visualizations")
    print("3. Use the synthetic data for your downstream tasks")
    print("4. Fine-tune hyperparameters if needed")


if __name__ == "__main__":
    main()
