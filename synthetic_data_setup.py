import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

try:
    from ctgan import CTGAN, TVAE
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    from sdv.evaluation.single_table import evaluate_quality
except ImportError as e:
    print(f"Error importing SDV/CTGAN: {e}")
    print("Please install with: pip install ctgan sdv")


class SyntheticDataSetup:
    """
    A comprehensive class for setting up metadata and detecting patterns
    for synthetic data generation using CTGAN and TVAE.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'synthetic_datasets/'):
        """
        Initialize the synthetic data setup.
        
        Args:
            data_path: Path to the CSV dataset
            output_dir: Directory to save outputs
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'patterns').mkdir(exist_ok=True)
        
        self.df = None
        self.metadata = None
        self.patterns = {}
        self.feature_types = {}
        
    def load_and_analyze_data(self):
        """Load the dataset and perform initial analysis."""
        print("Loading and analyzing dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {len(self.df.columns)}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Basic statistics
        print("\nDataset Info:")
        print(self.df.info())
        
        return self.df
    
    def detect_column_types(self):
        """
        Automatically detect column types for metadata setup.
        """
        print("\nDetecting column types...")
        
        categorical_cols = []
        numerical_cols = []
        datetime_cols = []
        binary_cols = []
        
        for col in self.df.columns:
            # Check for categorical/label columns
            if col.lower() in ['label', 'activity', 'class', 'target']:
                categorical_cols.append(col)
                continue
                
            # Check unique values
            unique_vals = self.df[col].nunique()
            total_vals = len(self.df[col])
            unique_ratio = unique_vals / total_vals
            
            # Check for binary columns (flags, percentages)
            if unique_vals == 2 or (self.df[col].dtype in ['float64', 'int64'] and 
                                   set(self.df[col].dropna().unique()).issubset({0, 1})):
                binary_cols.append(col)
            elif unique_vals < 10 and unique_ratio < 0.05:
                categorical_cols.append(col)
            elif self.df[col].dtype in ['float64', 'int64']:
                numerical_cols.append(col)
            elif 'time' in col.lower() or 'date' in col.lower():
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        
        self.feature_types = {
            'categorical': categorical_cols,
            'numerical': numerical_cols,
            'binary': binary_cols,
            'datetime': datetime_cols
        }
        
        print("Column type detection results:")
        for type_name, cols in self.feature_types.items():
            print(f"  {type_name}: {len(cols)} columns")
            if len(cols) <= 10:  # Only show if not too many
                print(f"    {cols}")
        
        return self.feature_types
    
    def create_metadata(self):
        """
        Create metadata for SDV/CTGAN training.
        """
        print("\nCreating metadata for synthetic data generation...")
        
        try:
            # Create SingleTableMetadata
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(self.df)
            
            # Update metadata based on our analysis
            for col in self.feature_types['categorical']:
                if col in self.df.columns:
                    self.metadata.update_column(col, sdtype='categorical')
            
            for col in self.feature_types['binary']:
                if col in self.df.columns:
                    self.metadata.update_column(col, sdtype='boolean')
            
            for col in self.feature_types['numerical']:
                if col in self.df.columns:
                    self.metadata.update_column(col, sdtype='numerical')
            
            # Save metadata
            metadata_path = self.output_dir / 'metadata' / 'table_metadata.json'
            self.metadata.save_to_json(metadata_path)
            
            print(f"Metadata saved to: {metadata_path}")
            
            # Create a human-readable version
            metadata_dict = {
                'columns': {},
                'primary_key': None,
                'constraints': []
            }
            
            for column in self.df.columns:
                col_info = {
                    'sdtype': self.metadata.columns[column]['sdtype'],
                    'unique_values': int(self.df[column].nunique()),
                    'null_percentage': float(self.df[column].isnull().mean() * 100),
                    'data_type': str(self.df[column].dtype)
                }
                
                if col_info['sdtype'] == 'numerical':
                    col_info.update({
                        'min': float(self.df[column].min()),
                        'max': float(self.df[column].max()),
                        'mean': float(self.df[column].mean()),
                        'std': float(self.df[column].std())
                    })
                elif col_info['sdtype'] == 'categorical':
                    col_info['categories'] = self.df[column].value_counts().head(10).to_dict()
                
                metadata_dict['columns'][column] = col_info
            
            # Save human-readable metadata
            readable_path = self.output_dir / 'metadata' / 'metadata_summary.json'
            with open(readable_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            
            print(f"Human-readable metadata saved to: {readable_path}")
            
        except Exception as e:
            print(f"Error creating metadata: {e}")
            # Fallback to manual metadata creation
            self._create_manual_metadata()
        
        return self.metadata
    
    def _create_manual_metadata(self):
        """Create metadata manually if automatic detection fails."""
        print("Creating manual metadata...")
        
        metadata_dict = {
            'columns': {},
            'primary_key': None,
            'table_name': 'network_traffic'
        }
        
        for col in self.df.columns:
            if col in self.feature_types['categorical']:
                sdtype = 'categorical'
            elif col in self.feature_types['binary']:
                sdtype = 'boolean'
            elif col in self.feature_types['numerical']:
                sdtype = 'numerical'
            else:
                sdtype = 'categorical'
            
            metadata_dict['columns'][col] = {
                'sdtype': sdtype,
                'pii': False
            }
        
        # Save manual metadata
        manual_path = self.output_dir / 'metadata' / 'manual_metadata.json'
        with open(manual_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"Manual metadata saved to: {manual_path}")
    
    def detect_patterns_and_relationships(self):
        """
        Detect patterns and relationships in the data.
        """
        print("\nDetecting patterns and relationships...")
        
        # 1. Correlation analysis
        self._analyze_correlations()
        
        # 2. Distribution analysis
        self._analyze_distributions()
        
        # 3. Categorical relationships
        self._analyze_categorical_relationships()
        
        # 4. TCP Flag specific patterns
        self._analyze_tcp_flag_patterns()
        
        # 5. Temporal patterns (if any timestamp columns)
        self._analyze_temporal_patterns()
        
        # 6. Clustering analysis
        self._perform_clustering_analysis()
        
        # Save all patterns
        patterns_path = self.output_dir / 'patterns' / 'detected_patterns.json'
        
        # Convert non-serializable objects to strings
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        converted_patterns = convert_for_json(self.patterns)
        
        with open(patterns_path, 'w') as f:
            json.dump(converted_patterns, f, indent=2, default=str)
        
        print(f"Pattern analysis saved to: {patterns_path}")
        
        return self.patterns
    
    def _analyze_correlations(self):
        """Analyze correlations between numerical features."""
        print("  - Analyzing correlations...")
        
        numerical_cols = [col for col in self.feature_types['numerical'] 
                         if col in self.df.columns]
        
        if len(numerical_cols) > 1:
            corr_matrix = self.df[numerical_cols].corr()
            
            # Find high correlations
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            self.patterns['high_correlations'] = high_corr_pairs
            
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations' / 'correlation_heatmap.png', dpi=300)
            plt.close()
    
    def _analyze_distributions(self):
        """Analyze feature distributions."""
        print("  - Analyzing distributions...")
        
        distribution_stats = {}
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            data = self.df[col].dropna()
            
            # Basic stats
            stats_dict = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data)),
                'min': float(data.min()),
                'max': float(data.max()),
                'q25': float(data.quantile(0.25)),
                'q50': float(data.quantile(0.50)),
                'q75': float(data.quantile(0.75))
            }
            
            # Test for normality
            if len(data) > 3:
                _, p_value = stats.normaltest(data)
                stats_dict['normality_p_value'] = float(p_value)
                stats_dict['is_normal'] = p_value > 0.05
            
            distribution_stats[col] = stats_dict
        
        self.patterns['distributions'] = distribution_stats
        
        # Create distribution plots for key features
        numerical_cols = list(distribution_stats.keys())[:6]  # Limit to 6 for visualization
        if numerical_cols:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    self.df[col].hist(bins=30, ax=axes[i])
                    axes[i].set_title(f'{col} Distribution')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations' / 'feature_distributions.png', dpi=300)
            plt.close()
    
    def _analyze_categorical_relationships(self):
        """Analyze relationships between categorical variables."""
        print("  - Analyzing categorical relationships...")
        
        categorical_cols = [col for col in self.feature_types['categorical'] 
                           if col in self.df.columns]
        
        if len(categorical_cols) >= 2:
            relationships = []
            
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i+1:]:
                    # Create contingency table
                    contingency = pd.crosstab(self.df[col1], self.df[col2])
                    
                    # Chi-square test
                    chi2, p_value, dof, expected = chi2_contingency(contingency)
                    
                    relationships.append({
                        'variable1': col1,
                        'variable2': col2,
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significant': p_value < 0.05
                    })
            
            self.patterns['categorical_relationships'] = relationships
    
    def _analyze_tcp_flag_patterns(self):
        """Analyze TCP flag specific patterns."""
        print("  - Analyzing TCP flag patterns...")
        
        tcp_flag_cols = [col for col in self.df.columns 
                        if any(flag in col.lower() for flag in 
                              ['syn', 'rst', 'psh', 'ack', 'fin', 'urg', 'flag'])]
        
        if tcp_flag_cols:
            tcp_patterns = {}
            
            # Analyze flag combinations
            flag_combinations = self.df[tcp_flag_cols].value_counts().head(20)
            tcp_patterns['common_flag_combinations'] = flag_combinations.to_dict()
            
            # Analyze flag correlations with attack types
            if 'label' in self.df.columns or 'activity' in self.df.columns:
                target_col = 'label' if 'label' in self.df.columns else 'activity'
                
                flag_attack_patterns = {}
                for flag_col in tcp_flag_cols:
                    if self.df[flag_col].dtype in ['float64', 'int64']:
                        # Numerical flag analysis
                        flag_by_attack = self.df.groupby(target_col)[flag_col].agg([
                            'mean', 'std', 'min', 'max'
                        ]).to_dict()
                        flag_attack_patterns[flag_col] = flag_by_attack
                
                tcp_patterns['flag_attack_relationships'] = flag_attack_patterns
            
            self.patterns['tcp_flag_patterns'] = tcp_patterns
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns if timestamp columns exist."""
        print("  - Analyzing temporal patterns...")
        
        # Look for timestamp-like columns
        time_cols = [col for col in self.df.columns 
                    if any(term in col.lower() for term in 
                          ['time', 'timestamp', 'iat', 'duration'])]
        
        if time_cols:
            temporal_patterns = {}
            
            for col in time_cols:
                if self.df[col].dtype in ['float64', 'int64']:
                    # Analyze time intervals
                    data = self.df[col].dropna()
                    temporal_patterns[col] = {
                        'mean_interval': float(data.mean()),
                        'std_interval': float(data.std()),
                        'min_interval': float(data.min()),
                        'max_interval': float(data.max()),
                        'unique_values': int(data.nunique())
                    }
            
            self.patterns['temporal_patterns'] = temporal_patterns
    
    def _perform_clustering_analysis(self):
        """Perform clustering analysis to identify data patterns."""
        print("  - Performing clustering analysis...")
        
        # Select numerical features for clustering
        numerical_cols = [col for col in self.feature_types['numerical'] 
                         if col in self.df.columns]
        
        if len(numerical_cols) >= 2:
            # Prepare data
            cluster_data = self.df[numerical_cols].dropna()
            
            if len(cluster_data) > 0:
                # Standardize features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Perform PCA for dimensionality reduction
                pca = PCA(n_components=min(10, len(numerical_cols)))
                pca_data = pca.fit_transform(scaled_data)
                
                # Find optimal number of clusters using elbow method
                inertias = []
                k_range = range(2, min(11, len(cluster_data)//10 + 1))
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(pca_data)
                    inertias.append(kmeans.inertia_)
                
                # Perform clustering with optimal k (simplified: use k=5)
                optimal_k = min(5, len(k_range))
                if optimal_k >= 2:
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(pca_data)
                    
                    # Analyze clusters
                    cluster_analysis = {}
                    for i in range(optimal_k):
                        cluster_mask = cluster_labels == i
                        cluster_stats = {}
                        
                        for col in numerical_cols:
                            cluster_col_data = cluster_data.loc[cluster_mask, col]
                            cluster_stats[col] = {
                                'mean': float(cluster_col_data.mean()),
                                'size': int(cluster_mask.sum())
                            }
                        
                        cluster_analysis[f'cluster_{i}'] = cluster_stats
                    
                    self.patterns['clustering_analysis'] = {
                        'n_clusters': optimal_k,
                        'cluster_stats': cluster_analysis,
                        'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
                    }
    
    def create_training_configurations(self):
        """
        Create training configurations for CTGAN and TVAE.
        """
        print("\nCreating training configurations...")
        
        # CTGAN configuration
        ctgan_config = {
            'model_type': 'CTGAN',
            'epochs': 300,
            'batch_size': 500,
            'generator_dim': (256, 256),
            'discriminator_dim': (256, 256),
            'generator_lr': 2e-4,
            'discriminator_lr': 2e-4,
            'discriminator_steps': 1,
            'log_frequency': True,
            'verbose': True
        }
        
        # TVAE configuration  
        tvae_config = {
            'model_type': 'TVAE',
            'epochs': 300,
            'batch_size': 500,
            'embedding_dim': 128,
            'compress_dims': (128, 128),
            'decompress_dims': (128, 128),
            'l2scale': 1e-5,
            'loss_factor': 2,
            'verbose': True
        }
        
        # Save configurations
        configs = {
            'ctgan': ctgan_config,
            'tvae': tvae_config,
            'data_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'categorical_columns': self.feature_types['categorical'],
                'numerical_columns': self.feature_types['numerical'],
                'binary_columns': self.feature_types['binary']
            }
        }
        
        config_path = self.output_dir / 'metadata' / 'training_configs.json'
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=2)
        
        print(f"Training configurations saved to: {config_path}")
        
        return configs
    
    def generate_sample_training_script(self):
        """
        Generate a sample training script for CTGAN and TVAE.
        """
        training_script = '''
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
    print("\\nTraining CTGAN...")
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
    print("\\nGenerating synthetic data with CTGAN...")
    synthetic_ctgan = ctgan.sample(num_rows=len(df))
    synthetic_ctgan.to_csv(Path(output_dir) / 'ctgan_synthetic_data.csv', index=False)
    
    # Train TVAE
    print("\\nTraining TVAE...")
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
    print("\\nGenerating synthetic data with TVAE...")
    synthetic_tvae = tvae.sample(num_rows=len(df))
    synthetic_tvae.to_csv(Path(output_dir) / 'tvae_synthetic_data.csv', index=False)
    
    # Evaluate quality
    print("\\nEvaluating synthetic data quality...")
    
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
'''
        
        script_path = self.output_dir / 'train_synthetic_models.py'
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        print(f"Sample training script saved to: {script_path}")
        
        return script_path
    
    def create_visualization_dashboard(self):
        """
        Create interactive visualizations for pattern analysis.
        """
        print("\nCreating interactive visualizations...")
        
        # Create correlation matrix plot
        if 'high_correlations' in self.patterns:
            correlations = self.patterns['high_correlations']
            if correlations:
                corr_df = pd.DataFrame(correlations)
                
                fig = px.scatter(corr_df, x='feature1', y='feature2', 
                               size=abs(corr_df['correlation']), 
                               color='correlation',
                               title='High Feature Correlations',
                               color_continuous_scale='RdBu')
                fig.write_html(self.output_dir / 'visualizations' / 'correlations_interactive.html')
        
        # Create distribution comparison
        if self.feature_types['numerical']:
            numerical_cols = self.feature_types['numerical'][:6]  # Limit for performance
            
            fig = make_subplots(rows=2, cols=3, 
                              subplot_titles=numerical_cols)
            
            for i, col in enumerate(numerical_cols):
                row = (i // 3) + 1
                col_idx = (i % 3) + 1
                
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=col),
                    row=row, col=col_idx
                )
            
            fig.update_layout(title_text="Feature Distributions", showlegend=False)
            fig.write_html(self.output_dir / 'visualizations' / 'distributions_interactive.html')
        
        print("Interactive visualizations saved to visualizations/ folder")
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("="*60)
        print("SYNTHETIC DATA SETUP - COMPLETE ANALYSIS")
        print("="*60)
        
        # Step 1: Load and analyze data
        self.load_and_analyze_data()
        
        # Step 2: Detect column types
        self.detect_column_types()
        
        # Step 3: Create metadata
        self.create_metadata()
        
        # Step 4: Detect patterns
        self.detect_patterns_and_relationships()
        
        # Step 5: Create training configurations
        self.create_training_configurations()
        
        # Step 6: Generate training script
        self.generate_sample_training_script()
        
        # Step 7: Create visualizations
        self.create_visualization_dashboard()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All outputs saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Review the metadata and patterns in synthetic_datasets/")
        print("2. Run the generated training script: python synthetic_datasets/train_synthetic_models.py")
        print("3. Evaluate the generated synthetic data quality")
        print("4. Adjust training parameters based on results")
        
        return {
            'metadata': self.metadata,
            'patterns': self.patterns,
            'feature_types': self.feature_types
        }


if __name__ == "__main__":
    # Initialize and run analysis
    setup = SyntheticDataSetup(
        data_path='datasets/attack-tcp-flag-osyn.csv',
        output_dir='synthetic_datasets/'
    )
    
    results = setup.run_complete_analysis()
