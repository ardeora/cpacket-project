#!/usr/bin/env python3
"""
Analysis script for comparing and analyzing trained DDoS classification models.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List
import argparse

class ModelAnalyzer:
    """
    Analyze and compare trained DDoS classification models.
    """
    
    def __init__(self, models_dir: str = 'saved_models/'):
        """Initialize with path to saved models."""
        self.models_dir = Path(models_dir)
        self.results = self._load_all_results()
        
    def _load_all_results(self) -> Dict:
        """Load results from all trained models."""
        results = {}
        
        for results_file in self.models_dir.glob('results_*.json'):
            method = results_file.stem.replace('results_', '')
            if method.endswith('_quick_test'):
                continue  # Skip test results
                
            with open(results_file, 'r') as f:
                results[method] = json.load(f)
        
        return results
    
    def compare_accuracies(self) -> pd.DataFrame:
        """Create a comparison table of model accuracies."""
        comparison_data = []
        
        for method, result in self.results.items():
            report = result['classification_report']
            
            comparison_data.append({
                'Method': method,
                'Overall_Accuracy': result['accuracy'],
                'Attack_F1': report['Attack']['f1-score'],
                'Benign_F1': report['Benign']['f1-score'],
                'Suspicious_F1': report['Suspicious']['f1-score'],
                'Macro_Avg_F1': report['macro avg']['f1-score'],
                'Weighted_Avg_F1': report['weighted avg']['f1-score'],
                'Features_Used': result['n_features']
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Overall_Accuracy', ascending=False)
    
    def plot_performance_comparison(self, save_plot: bool = True):
        """Create comprehensive performance comparison plots."""
        comparison_df = self.compare_accuracies()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Overall Accuracy
        axes[0, 0].bar(comparison_df['Method'], comparison_df['Overall_Accuracy'])
        axes[0, 0].set_title('Overall Accuracy by Method')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Scores by Class
        f1_data = comparison_df[['Method', 'Attack_F1', 'Benign_F1', 'Suspicious_F1']].set_index('Method')
        f1_data.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('F1-Scores by Class and Method')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].legend(title='Class')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Macro vs Weighted F1
        axes[1, 0].scatter(comparison_df['Macro_Avg_F1'], comparison_df['Weighted_Avg_F1'])
        for i, method in enumerate(comparison_df['Method']):
            axes[1, 0].annotate(method, 
                              (comparison_df.iloc[i]['Macro_Avg_F1'], 
                               comparison_df.iloc[i]['Weighted_Avg_F1']))
        axes[1, 0].set_xlabel('Macro Average F1-Score')
        axes[1, 0].set_ylabel('Weighted Average F1-Score')
        axes[1, 0].set_title('Macro vs Weighted F1-Score')
        
        # Performance vs Features
        axes[1, 1].scatter(comparison_df['Features_Used'], comparison_df['Overall_Accuracy'])
        for i, method in enumerate(comparison_df['Method']):
            axes[1, 1].annotate(method, 
                              (comparison_df.iloc[i]['Features_Used'], 
                               comparison_df.iloc[i]['Overall_Accuracy']))
        axes[1, 1].set_xlabel('Number of Features')
        axes[1, 1].set_ylabel('Overall Accuracy')
        axes[1, 1].set_title('Accuracy vs Number of Features')
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.models_dir / 'performance_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison plot saved to: {plot_path}")
        
        plt.show()
    
    def analyze_feature_overlap(self) -> Dict:
        """Analyze feature overlap between different methods."""
        feature_sets = {}
        
        # Load feature lists for each method
        for features_file in self.models_dir.glob('features_*.json'):
            method = features_file.stem.replace('features_', '')
            if method.endswith('_quick_test'):
                continue
                
            with open(features_file, 'r') as f:
                data = json.load(f)
                feature_sets[method] = set(data['features'])
        
        if len(feature_sets) < 2:
            return {"error": "Need at least 2 methods to compare features"}
        
        # Calculate intersections
        methods = list(feature_sets.keys())
        analysis = {
            'total_features_per_method': {method: len(features) for method, features in feature_sets.items()},
            'pairwise_overlap': {},
            'common_features': set.intersection(*feature_sets.values()),
            'unique_features': {}
        }
        
        # Pairwise overlaps
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                overlap = feature_sets[method1] & feature_sets[method2]
                analysis['pairwise_overlap'][f"{method1}_vs_{method2}"] = {
                    'overlap_count': len(overlap),
                    'overlap_features': list(overlap)
                }
        
        # Unique features per method
        for method in methods:
            other_features = set()
            for other_method in methods:
                if other_method != method:
                    other_features.update(feature_sets[other_method])
            
            unique = feature_sets[method] - other_features
            analysis['unique_features'][method] = list(unique)
        
        return analysis
    
    def print_detailed_analysis(self):
        """Print a comprehensive analysis of all models."""
        print("="*80)
        print("DETAILED MODEL ANALYSIS")
        print("="*80)
        
        # Performance comparison
        print("\n1. PERFORMANCE COMPARISON")
        print("-" * 40)
        comparison_df = self.compare_accuracies()
        print(comparison_df.round(4).to_string(index=False))
        
        # Best performing model
        best_method = comparison_df.iloc[0]['Method']
        best_accuracy = comparison_df.iloc[0]['Overall_Accuracy']
        print(f"\n🏆 Best performing method: {best_method} (Accuracy: {best_accuracy:.4f})")
        
        # Feature analysis
        print("\n2. FEATURE ANALYSIS")
        print("-" * 40)
        feature_analysis = self.analyze_feature_overlap()
        
        if 'error' not in feature_analysis:
            print(f"Total unique features across all methods: {len(set().union(*[self.results[m]['classification_report'].keys() for m in self.results.keys() if isinstance(self.results[m]['classification_report'], dict)]))}")
            print(f"Features common to all methods: {len(feature_analysis['common_features'])}")
            
            for method, unique_features in feature_analysis['unique_features'].items():
                print(f"{method}: {len(unique_features)} unique features")
        
        # Class-specific performance
        print("\n3. CLASS-SPECIFIC PERFORMANCE")
        print("-" * 40)
        
        for class_name in ['Attack', 'Benign', 'Suspicious']:
            print(f"\n{class_name} Detection:")
            class_performance = []
            for method, result in self.results.items():
                report = result['classification_report']
                if class_name in report:
                    class_performance.append({
                        'Method': method,
                        'Precision': report[class_name]['precision'],
                        'Recall': report[class_name]['recall'],
                        'F1-Score': report[class_name]['f1-score']
                    })
            
            class_df = pd.DataFrame(class_performance).sort_values('F1-Score', ascending=False)
            print(class_df.round(4).to_string(index=False))
    
    def generate_report(self, output_file: str = None):
        """Generate a comprehensive report and save to file."""
        if output_file is None:
            output_file = self.models_dir / 'analysis_report.txt'
        
        import sys
        from io import StringIO
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        self.print_detailed_analysis()
        
        # Restore stdout and get output
        sys.stdout = old_stdout
        report_content = buffer.getvalue()
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"Detailed analysis report saved to: {output_file}")
        return report_content


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(description='Analyze DDoS Classification Models')
    parser.add_argument('--models-dir', type=str, default='saved_models/',
                       help='Directory containing saved models')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance comparison plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed analysis report')
    parser.add_argument('--output', type=str,
                       help='Output file for report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(args.models_dir)
    
    if not analyzer.results:
        print("No model results found. Please train some models first.")
        return
    
    print(f"Found {len(analyzer.results)} trained models")
    
    # Print basic analysis
    analyzer.print_detailed_analysis()
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating performance comparison plots...")
        analyzer.plot_performance_comparison()
    
    # Generate report if requested
    if args.report:
        print("\nGenerating detailed analysis report...")
        analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()
