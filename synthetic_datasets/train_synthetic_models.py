
import pandas as pd
import json
from pathlib import Path
from ctgan import CTGAN, TVAE
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality
import warnings
warnings.filterwarnings('ignore')

def train_synthetic_models(data_path, metadata_path, output_dir):
    """
    Train CTGAN and TVAE models for synthetic data generation.
    """
    # Load data and metadata
    df = pd.read_csv(data_path)
    metadata = SingleTableMetadata.load_from_json(metadata_path)
    
    print(f"Training on dataset with shape: {df.shape}")
    
    # Train CTGAN
    print("\nTraining CTGAN...")
    ctgan = CTGANSynthesizer(
        metadata=metadata,
        epochs=300,
        batch_size=500,
        verbose=True
    )
    ctgan.fit(df)
    
    # Save CTGAN model
    ctgan_path = Path(output_dir) / 'models' / 'ctgan_model.pkl'
    ctgan.save(ctgan_path)
    print(f"CTGAN model saved to: {ctgan_path}")
    
    # Generate synthetic data with CTGAN
    print("\nGenerating synthetic data with CTGAN...")
    synthetic_ctgan = ctgan.sample(num_rows=len(df))
    synthetic_ctgan.to_csv(Path(output_dir) / 'ctgan_synthetic_data.csv', index=False)
    
    # Train TVAE
    print("\nTraining TVAE...")
    tvae = TVAESynthesizer(
        metadata=metadata,
        epochs=300,
        batch_size=500,
        verbose=True
    )
    tvae.fit(df)
    
    # Save TVAE model
    tvae_path = Path(output_dir) / 'models' / 'tvae_model.pkl'
    tvae.save(tvae_path)
    print(f"TVAE model saved to: {tvae_path}")
    
    # Generate synthetic data with TVAE
    print("\nGenerating synthetic data with TVAE...")
    synthetic_tvae = tvae.sample(num_rows=len(df))
    synthetic_tvae.to_csv(Path(output_dir) / 'tvae_synthetic_data.csv', index=False)
    
    # Evaluate quality
    print("\nEvaluating synthetic data quality...")
    
    # CTGAN evaluation
    ctgan_quality = evaluate_quality(df, synthetic_ctgan, metadata)
    print(f"CTGAN Quality Score: {ctgan_quality}")
    
    # TVAE evaluation
    tvae_quality = evaluate_quality(df, synthetic_tvae, metadata)
    print(f"TVAE Quality Score: {tvae_quality}")
    
    # Save evaluation results
    evaluation_results = {
        'ctgan_quality': float(ctgan_quality),
        'tvae_quality': float(tvae_quality),
        'original_shape': df.shape,
        'synthetic_ctgan_shape': synthetic_ctgan.shape,
        'synthetic_tvae_shape': synthetic_tvae.shape
    }
    
    with open(Path(output_dir) / 'evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return ctgan, tvae, synthetic_ctgan, synthetic_tvae

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "datasets/attack-tcp-flag-osyn.csv"
    METADATA_PATH = "synthetic_datasets/metadata/table_metadata.json"
    OUTPUT_DIR = "synthetic_datasets/"
    
    # Train models
    train_synthetic_models(DATA_PATH, METADATA_PATH, OUTPUT_DIR)
