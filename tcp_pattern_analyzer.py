import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TCPFlagPatternAnalyzer:
    """
    Advanced pattern analysis specifically for TCP flag data and network traffic.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'synthetic_datasets/'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.patterns_dir = self.output_dir / 'patterns'
        self.visualizations_dir = self.output_dir / 'visualizations'
        
        # Create directories
        self.patterns_dir.mkdir(exist_ok=True)
        self.visualizations_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.tcp_flag_patterns = {}
        self.network_patterns = {}
        self.attack_patterns = {}
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading TCP flag dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def identify_feature_categories(self):
        """Categorize features by their network traffic characteristics."""
        feature_categories = {
            'tcp_flags': [],
            'header_bytes': [],
            'timing_features': [],
            'rate_features': [],
            'window_features': [],
            'packet_counts': [],
            'directional_features': {'forward': [], 'backward': []},
            'labels': []
        }
        
        for col in self.df.columns:
            col_lower = col.lower()
            
            # TCP Flag features
            if any(flag in col_lower for flag in ['syn', 'rst', 'psh', 'ack', 'fin', 'urg', 'flag']):
                feature_categories['tcp_flags'].append(col)
            
            # Header byte features
            elif 'header' in col_lower and 'byte' in col_lower:
                feature_categories['header_bytes'].append(col)
            
            # Timing features (IAT = Inter-Arrival Time)
            elif any(term in col_lower for term in ['iat', 'time', 'duration']):
                feature_categories['timing_features'].append(col)
            
            # Rate features
            elif 'rate' in col_lower:
                feature_categories['rate_features'].append(col)
            
            # Window features
            elif 'win' in col_lower:
                feature_categories['window_features'].append(col)
            
            # Packet count features
            elif any(term in col_lower for term in ['count', 'total']) and 'byte' not in col_lower:
                feature_categories['packet_counts'].append(col)
            
            # Labels
            elif col_lower in ['label', 'activity', 'class', 'target']:
                feature_categories['labels'].append(col)
            
            # Directional features
            if 'fwd' in col_lower or 'forward' in col_lower:
                feature_categories['directional_features']['forward'].append(col)
            elif 'bwd' in col_lower or 'backward' in col_lower:
                feature_categories['directional_features']['backward'].append(col)
        
        return feature_categories
    
    def analyze_tcp_flag_relationships(self):
        """Analyze relationships between different TCP flags."""
        print("Analyzing TCP flag relationships...")
        
        feature_categories = self.identify_feature_categories()
        tcp_flag_cols = feature_categories['tcp_flags']
        
        if not tcp_flag_cols:
            print("No TCP flag columns found!")
            return
        
        flag_analysis = {}
        
        # 1. Flag co-occurrence analysis
        print("  - Analyzing flag co-occurrence patterns...")
        flag_cooccurrence = {}
        
        for i, flag1 in enumerate(tcp_flag_cols):
            for flag2 in tcp_flag_cols[i+1:]:
                # Calculate correlation
                if self.df[flag1].dtype in ['float64', 'int64'] and self.df[flag2].dtype in ['float64', 'int64']:
                    correlation = self.df[flag1].corr(self.df[flag2])
                    
                    # Calculate mutual information
                    try:
                        mi_score = mutual_info_regression(
                            self.df[[flag1]], self.df[flag2], random_state=42
                        )[0]
                    except:
                        mi_score = 0
                    
                    flag_cooccurrence[f"{flag1}_vs_{flag2}"] = {
                        'correlation': float(correlation),
                        'mutual_information': float(mi_score),
                        'flag1_mean': float(self.df[flag1].mean()),
                        'flag2_mean': float(self.df[flag2].mean())
                    }
        
        flag_analysis['cooccurrence'] = flag_cooccurrence
        
        # 2. Flag distribution by attack type
        if 'label' in self.df.columns:
            print("  - Analyzing flag patterns by attack type...")
            flag_by_label = {}
            
            for flag_col in tcp_flag_cols:
                if self.df[flag_col].dtype in ['float64', 'int64']:
                    flag_stats = self.df.groupby('label')[flag_col].agg([
                        'mean', 'std', 'min', 'max', 'count'
                    ]).to_dict()
                    flag_by_label[flag_col] = flag_stats
            
            flag_analysis['by_attack_type'] = flag_by_label
        
        # 3. Flag sequence patterns
        print("  - Identifying flag sequence patterns...")
        flag_combinations = {}
        
        # Get the main flag percentage columns
        flag_pct_cols = [col for col in tcp_flag_cols if 'percentage' in col.lower()]
        
        if flag_pct_cols:
            # Create flag combination signatures
            for idx, row in self.df.head(100).iterrows():  # Sample first 100 for performance
                signature = tuple(row[col] for col in flag_pct_cols[:5])  # Limit to 5 flags
                signature_str = str(signature)
                
                if signature_str not in flag_combinations:
                    flag_combinations[signature_str] = 0
                flag_combinations[signature_str] += 1
            
            # Get top combinations
            top_combinations = dict(sorted(flag_combinations.items(), 
                                         key=lambda x: x[1], reverse=True)[:20])
            flag_analysis['top_combinations'] = top_combinations
        
        self.tcp_flag_patterns = flag_analysis
        return flag_analysis
    
    def analyze_network_traffic_patterns(self):
        """Analyze network traffic patterns beyond TCP flags."""
        print("Analyzing network traffic patterns...")
        
        feature_categories = self.identify_feature_categories()
        traffic_analysis = {}
        
        # 1. Timing pattern analysis
        timing_cols = feature_categories['timing_features']
        if timing_cols:
            print("  - Analyzing timing patterns...")
            timing_patterns = {}
            
            for col in timing_cols:
                if self.df[col].dtype in ['float64', 'int64']:
                    data = self.df[col].dropna()
                    
                    timing_patterns[col] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'std': float(data.std()),
                        'cv': float(data.std() / data.mean()) if data.mean() != 0 else 0,
                        'skewness': float(stats.skew(data)),
                        'kurtosis': float(stats.kurtosis(data)),
                        'zero_percentage': float((data == 0).mean() * 100)
                    }
            
            traffic_analysis['timing_patterns'] = timing_patterns
        
        # 2. Directional flow analysis
        print("  - Analyzing directional flow patterns...")
        fwd_cols = feature_categories['directional_features']['forward']
        bwd_cols = feature_categories['directional_features']['backward']
        
        if fwd_cols and bwd_cols:
            directional_patterns = {}
            
            # Compare forward vs backward statistics
            for fwd_col in fwd_cols[:5]:  # Limit for performance
                # Find corresponding backward column
                fwd_base = fwd_col.replace('fwd_', '').replace('forward_', '')
                potential_bwd_cols = [col for col in bwd_cols 
                                    if fwd_base in col.replace('bwd_', '').replace('backward_', '')]
                
                if potential_bwd_cols:
                    bwd_col = potential_bwd_cols[0]
                    
                    if (self.df[fwd_col].dtype in ['float64', 'int64'] and 
                        self.df[bwd_col].dtype in ['float64', 'int64']):
                        
                        fwd_mean = self.df[fwd_col].mean()
                        bwd_mean = self.df[bwd_col].mean()
                        
                        directional_patterns[f"{fwd_col}_vs_{bwd_col}"] = {
                            'fwd_mean': float(fwd_mean),
                            'bwd_mean': float(bwd_mean),
                            'ratio': float(fwd_mean / bwd_mean) if bwd_mean != 0 else float('inf'),
                            'correlation': float(self.df[fwd_col].corr(self.df[bwd_col]))
                        }
            
            traffic_analysis['directional_patterns'] = directional_patterns
        
        # 3. Rate and volume analysis
        rate_cols = feature_categories['rate_features']
        header_cols = feature_categories['header_bytes']
        
        if rate_cols:
            print("  - Analyzing rate patterns...")
            rate_patterns = {}
            
            for col in rate_cols:
                if self.df[col].dtype in ['float64', 'int64']:
                    data = self.df[col].dropna()
                    
                    rate_patterns[col] = {
                        'mean': float(data.mean()),
                        'max': float(data.max()),
                        'min': float(data.min()),
                        'percentile_95': float(data.quantile(0.95)),
                        'outlier_percentage': float(((data > data.quantile(0.95)) | 
                                                   (data < data.quantile(0.05))).mean() * 100)
                    }
            
            traffic_analysis['rate_patterns'] = rate_patterns
        
        self.network_patterns = traffic_analysis
        return traffic_analysis
    
    def detect_attack_signatures(self):
        """Detect specific attack signatures and patterns."""
        print("Detecting attack signatures...")
        
        if 'label' not in self.df.columns:
            print("No label column found for attack analysis!")
            return {}
        
        attack_analysis = {}
        
        # 1. Feature importance for attack detection
        print("  - Calculating feature importance...")
        
        # Prepare data for ML analysis
        feature_cols = [col for col in self.df.columns 
                       if col not in ['label', 'activity'] and 
                       self.df[col].dtype in ['float64', 'int64']]
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['label']
        
        # Use Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        feature_importance = {}
        for i, col in enumerate(feature_cols):
            feature_importance[col] = float(rf.feature_importances_[i])
        
        # Get top important features
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:20])
        
        attack_analysis['feature_importance'] = top_features
        
        # 2. Attack-specific statistical signatures
        print("  - Analyzing attack-specific signatures...")
        attack_signatures = {}
        
        for attack_type in self.df['label'].unique():
            attack_data = self.df[self.df['label'] == attack_type]
            normal_data = self.df[self.df['label'] != attack_type]
            
            signatures = {}
            
            for col in feature_cols[:15]:  # Limit for performance
                if len(attack_data) > 0 and len(normal_data) > 0:
                    attack_mean = attack_data[col].mean()
                    normal_mean = normal_data[col].mean()
                    
                    # Statistical test
                    try:
                        stat, p_value = stats.ttest_ind(attack_data[col], normal_data[col])
                        
                        signatures[col] = {
                            'attack_mean': float(attack_mean),
                            'normal_mean': float(normal_mean),
                            'difference_ratio': float(attack_mean / normal_mean) if normal_mean != 0 else float('inf'),
                            'statistical_significance': float(p_value),
                            'effect_size': float(abs(attack_mean - normal_mean) / 
                                               ((attack_data[col].std() + normal_data[col].std()) / 2))
                        }
                    except:
                        continue
            
            attack_signatures[attack_type] = signatures
        
        attack_analysis['attack_signatures'] = attack_signatures
        
        self.attack_patterns = attack_analysis
        return attack_analysis
    
    def create_network_graph_analysis(self):
        """Create network graph analysis of feature relationships."""
        print("Creating network graph analysis...")
        
        # Create a graph of high correlations
        G = nx.Graph()
        
        # Load correlation data
        with open(self.patterns_dir / 'detected_patterns.json', 'r') as f:
            patterns = json.load(f)
        
        if 'high_correlations' in patterns:
            correlations = patterns['high_correlations']
            
            for corr in correlations:
                if abs(corr['correlation']) > 0.8:  # Only very high correlations
                    G.add_edge(corr['feature1'], corr['feature2'], 
                             weight=abs(corr['correlation']))
        
        # Calculate network metrics
        network_metrics = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        if len(G.nodes()) > 0:
            # Centrality measures
            centrality = nx.degree_centrality(G)
            network_metrics['most_central_features'] = dict(sorted(centrality.items(), 
                                                                  key=lambda x: x[1], reverse=True)[:10])
        
        return network_metrics
    
    def generate_comprehensive_visualizations(self):
        """Generate comprehensive visualizations for all patterns."""
        print("Generating comprehensive visualizations...")
        
        # 1. TCP Flag correlation heatmap
        feature_categories = self.identify_feature_categories()
        tcp_flag_cols = feature_categories['tcp_flags']
        
        if tcp_flag_cols:
            flag_data = self.df[tcp_flag_cols].select_dtypes(include=[np.number])
            if not flag_data.empty:
                plt.figure(figsize=(12, 10))
                corr_matrix = flag_data.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                           center=0, fmt='.2f', cbar_kws={'label': 'Correlation'})
                plt.title('TCP Flag Correlation Matrix')
                plt.tight_layout()
                plt.savefig(self.visualizations_dir / 'tcp_flag_correlations.png', dpi=300)
                plt.close()
        
        # 2. Attack type distribution
        if 'label' in self.df.columns:
            plt.figure(figsize=(10, 6))
            label_counts = self.df['label'].value_counts()
            plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
            plt.title('Attack Type Distribution')
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'attack_distribution.png', dpi=300)
            plt.close()
        
        # 3. Feature importance visualization
        if hasattr(self, 'attack_patterns') and 'feature_importance' in self.attack_patterns:
            importance = self.attack_patterns['feature_importance']
            top_features = dict(list(importance.items())[:15])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            plt.barh(features, importances)
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Most Important Features for Attack Detection')
            plt.tight_layout()
            plt.savefig(self.visualizations_dir / 'feature_importance.png', dpi=300)
            plt.close()
        
        # 4. Interactive timing pattern visualization
        timing_cols = feature_categories['timing_features']
        if timing_cols:
            timing_data = self.df[timing_cols[:3]].select_dtypes(include=[np.number])  # Limit to 3
            if not timing_data.empty:
                fig = make_subplots(rows=1, cols=len(timing_data.columns),
                                  subplot_titles=timing_data.columns)
                
                for i, col in enumerate(timing_data.columns):
                    fig.add_trace(
                        go.Histogram(x=timing_data[col], name=col),
                        row=1, col=i+1
                    )
                
                fig.update_layout(title_text="Timing Feature Distributions", showlegend=False)
                fig.write_html(self.visualizations_dir / 'timing_patterns_interactive.html')
        
        print(f"Visualizations saved to: {self.visualizations_dir}")
    
    def save_comprehensive_analysis(self):
        """Save all analysis results."""
        comprehensive_analysis = {
            'tcp_flag_patterns': self.tcp_flag_patterns,
            'network_patterns': self.network_patterns,
            'attack_patterns': self.attack_patterns,
            'feature_categories': self.identify_feature_categories(),
            'network_graph_metrics': self.create_network_graph_analysis()
        }
        
        # Save the comprehensive analysis
        output_path = self.patterns_dir / 'comprehensive_tcp_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)
        
        print(f"Comprehensive analysis saved to: {output_path}")
        return comprehensive_analysis
    
    def run_complete_tcp_analysis(self):
        """Run the complete TCP flag and network traffic analysis."""
        print("="*60)
        print("TCP FLAG & NETWORK TRAFFIC PATTERN ANALYSIS")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.analyze_tcp_flag_relationships()
        self.analyze_network_traffic_patterns()
        self.detect_attack_signatures()
        
        # Generate visualizations
        self.generate_comprehensive_visualizations()
        
        # Save comprehensive results
        results = self.save_comprehensive_analysis()
        
        print("\n" + "="*60)
        print("TCP ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Results saved to: {self.patterns_dir}")
        print(f"Visualizations saved to: {self.visualizations_dir}")
        
        # Print summary
        print("\nSUMMARY:")
        if self.tcp_flag_patterns:
            print(f"- TCP flag relationships analyzed: {len(self.tcp_flag_patterns.get('cooccurrence', {}))}")
        if self.attack_patterns:
            print(f"- Attack signatures detected: {len(self.attack_patterns.get('attack_signatures', {}))}")
        if self.network_patterns:
            print(f"- Network traffic patterns: {len(self.network_patterns)}")
        
        return results


if __name__ == "__main__":
    # Run TCP flag analysis
    analyzer = TCPFlagPatternAnalyzer(
        data_path='datasets/attack-tcp-flag-osyn.csv',
        output_dir='synthetic_datasets/'
    )
    
    results = analyzer.run_complete_tcp_analysis()
