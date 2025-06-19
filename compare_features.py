#!/usr/bin/env python3
"""
Feature Selection Comparison Script

Compares the Random Forest selected features with existing feature selection methods
and shows overlaps, differences, and performance implications.
"""

import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set
import numpy as np


class FeatureSelectionComparator:
    """
    Compare different feature selection methods and analyze their effectiveness.
    """
    
    def __init__(self, 
                 features_dir: str = 'features/',
                 rf_features_path: str = 'feature_analysis/top_40_features.json',
                 output_dir: str = 'feature_comparison/'):
        """
        Initialize the comparator.
        
        Args:
            features_dir: Directory with existing feature selection methods
            rf_features_path: Path to Random Forest selected features
            output_dir: Directory to save comparison results
        """
        self.features_dir = Path(features_dir)
        self.rf_features_path = rf_features_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_sets = {}
        self.load_all_feature_sets()
    
    def load_all_feature_sets(self):
        """Load all feature selection methods."""
        print("Loading all feature selection methods...")
        
        # Load existing methods
        for json_file in self.features_dir.glob("*.json"):
            method_name = json_file.stem
            with open(json_file, 'r') as f:
                data = json.load(f)
                features = [f for f in data['features'] if f not in ['label', 'activity']]
                self.feature_sets[method_name] = set(features)
                print(f"  - {method_name}: {len(features)} features")
        
        # Load Random Forest features
        if Path(self.rf_features_path).exists():
            with open(self.rf_features_path, 'r') as f:
                rf_data = json.load(f)
                self.feature_sets['random_forest_all'] = set(rf_data['features'])
                print(f"  - random_forest_all: {len(rf_data['features'])} features")
        else:
            print(f"Warning: Random Forest features not found at {self.rf_features_path}")
    
    def calculate_overlaps(self) -> Dict:
        """Calculate feature overlaps between all methods."""
        print("\nCalculating feature overlaps...")
        
        methods = list(self.feature_sets.keys())
        overlaps = {}
        
        # Pairwise overlaps
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                set1 = self.feature_sets[method1]
                set2 = self.feature_sets[method2]
                
                intersection = set1 & set2
                union = set1 | set2
                
                overlap_pct = len(intersection) / len(union) * 100 if union else 0
                
                overlaps[f"{method1}_vs_{method2}"] = {
                    'intersection_count': len(intersection),
                    'intersection_features': list(intersection),
                    'union_count': len(union),
                    'overlap_percentage': overlap_pct,
                    'method1_unique': list(set1 - set2),
                    'method2_unique': list(set2 - set1)
                }
        
        # Overall statistics
        all_features = set()
        for features in self.feature_sets.values():
            all_features.update(features)
        
        # Features common to all methods
        common_features = set.intersection(*self.feature_sets.values())
        
        overlaps['overall_stats'] = {
            'total_unique_features': len(all_features),
            'common_to_all': list(common_features),
            'common_count': len(common_features)
        }
        
        return overlaps
    
    def create_overlap_matrix(self) -> pd.DataFrame:
        """Create a matrix showing pairwise overlaps."""
        methods = list(self.feature_sets.keys())
        n_methods = len(methods)
        
        # Create overlap matrix
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = 100  # Perfect overlap with itself
                else:
                    set1 = self.feature_sets[method1]
                    set2 = self.feature_sets[method2]
                    intersection = set1 & set2
                    union = set1 | set2
                    overlap_pct = len(intersection) / len(union) * 100 if union else 0
                    overlap_matrix[i, j] = overlap_pct
        
        return pd.DataFrame(overlap_matrix, index=methods, columns=methods)
    
    def analyze_rf_vs_existing(self) -> Dict:
        """Detailed analysis of Random Forest features vs existing methods."""
        if 'random_forest_all' not in self.feature_sets:
            return {"error": "Random Forest features not available"}
        
        rf_features = self.feature_sets['random_forest_all']
        analysis = {
            'rf_feature_count': len(rf_features),
            'comparisons': {}
        }
        
        print("\n🔍 Random Forest Features vs Existing Methods:")
        print("=" * 60)
        
        for method, features in self.feature_sets.items():
            if method == 'random_forest_all':
                continue
            
            intersection = rf_features & features
            rf_unique = rf_features - features
            method_unique = features - rf_features
            
            overlap_pct = len(intersection) / len(rf_features) * 100
            
            analysis['comparisons'][method] = {
                'intersection_count': len(intersection),
                'intersection_features': list(intersection),
                'rf_unique_count': len(rf_unique),
                'rf_unique_features': list(rf_unique),
                'method_unique_count': len(method_unique),
                'method_unique_features': list(method_unique),
                'overlap_percentage': overlap_pct
            }
            
            print(f"\n{method.upper()}:")
            print(f"  Features in common: {len(intersection)}/{len(rf_features)} ({overlap_pct:.1f}%)")
            print(f"  RF unique: {len(rf_unique)}")
            print(f"  {method} unique: {len(method_unique)}")
        
        return analysis
    
    def plot_comparisons(self, save_plots: bool = True):
        """Create visualization plots for feature comparisons."""
        overlap_matrix = self.create_overlap_matrix()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Selection Methods Comparison', fontsize=16)
        
        # 1. Overlap heatmap
        sns.heatmap(overlap_matrix, annot=True, fmt='.1f', cmap='Blues', 
                   ax=axes[0, 0], cbar_kws={'label': 'Overlap %'})
        axes[0, 0].set_title('Pairwise Feature Overlap Matrix')
        axes[0, 0].set_xlabel('Methods')
        axes[0, 0].set_ylabel('Methods')
        
        # 2. Feature set sizes
        method_names = list(self.feature_sets.keys())
        feature_counts = [len(features) for features in self.feature_sets.values()]
        
        colors = ['red' if 'random_forest' in name else 'blue' for name in method_names]
        axes[0, 1].bar(method_names, feature_counts, color=colors, alpha=0.7)
        axes[0, 1].set_title('Number of Features per Method')
        axes[0, 1].set_ylabel('Feature Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Overlap with Random Forest (if available)
        if 'random_forest_all' in self.feature_sets:
            rf_overlaps = []
            other_methods = [m for m in method_names if m != 'random_forest_all']
            
            for method in other_methods:
                rf_features = self.feature_sets['random_forest_all']
                method_features = self.feature_sets[method]
                intersection = rf_features & method_features
                overlap_pct = len(intersection) / len(rf_features) * 100
                rf_overlaps.append(overlap_pct)
            
            axes[1, 0].bar(other_methods, rf_overlaps, color='green', alpha=0.7)
            axes[1, 0].set_title('Overlap with Random Forest Selected Features')
            axes[1, 0].set_ylabel('Overlap Percentage (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Venn diagram-style visualization (for top 3 methods)
        if len(self.feature_sets) >= 3:
            # Create a simple bar chart showing unique vs shared features
            method_subset = list(self.feature_sets.keys())[:3]
            shared_counts = []
            unique_counts = []
            
            for method in method_subset:
                features = self.feature_sets[method]
                other_features = set()
                for other_method in method_subset:
                    if other_method != method:
                        other_features.update(self.feature_sets[other_method])
                
                shared = len(features & other_features)
                unique = len(features - other_features)
                shared_counts.append(shared)
                unique_counts.append(unique)
            
            x = np.arange(len(method_subset))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, unique_counts, width, label='Unique Features', alpha=0.7)
            axes[1, 1].bar(x + width/2, shared_counts, width, label='Shared Features', alpha=0.7)
            axes[1, 1].set_title('Unique vs Shared Features (Top 3 Methods)')
            axes[1, 1].set_ylabel('Feature Count')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(method_subset, rotation=45)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / 'feature_comparison_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nComparison plots saved to: {plot_path}")
        
        plt.show()
    
    def save_comparison_results(self, overlaps: Dict, rf_analysis: Dict):
        """Save all comparison results to files."""
        print(f"\nSaving comparison results to {self.output_dir}...")
        
        # Save overlap analysis
        overlap_path = self.output_dir / 'feature_overlaps.json'
        with open(overlap_path, 'w') as f:
            json.dump(overlaps, f, indent=2)
        
        # Save RF comparison
        if 'error' not in rf_analysis:
            rf_path = self.output_dir / 'rf_vs_existing_analysis.json'
            with open(rf_path, 'w') as f:
                json.dump(rf_analysis, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for method, features in self.feature_sets.items():
            if method != 'random_forest_all':
                rf_features = self.feature_sets.get('random_forest_all', set())
                intersection = features & rf_features if rf_features else set()
                overlap_pct = len(intersection) / len(features) * 100 if features else 0
                
                summary_data.append({
                    'Method': method,
                    'Feature_Count': len(features),
                    'RF_Overlap_Count': len(intersection),
                    'RF_Overlap_Percentage': overlap_pct
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = self.output_dir / 'method_comparison_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary CSV saved to: {summary_path}")
        
        print("✅ All comparison results saved!")
    
    def run_complete_comparison(self):
        """Run the complete feature selection comparison analysis."""
        print("🔄 STARTING FEATURE SELECTION COMPARISON")
        print("=" * 60)
        
        if not self.feature_sets:
            print("❌ No feature sets loaded. Please check input files.")
            return
        
        # Calculate overlaps
        overlaps = self.calculate_overlaps()
        
        # Analyze RF vs existing
        rf_analysis = self.analyze_rf_vs_existing()
        
        # Create visualizations
        self.plot_comparisons()
        
        # Save results
        self.save_comparison_results(overlaps, rf_analysis)
        
        # Print summary
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"   Methods analyzed: {len(self.feature_sets)}")
        print(f"   Total unique features across all methods: {overlaps['overall_stats']['total_unique_features']}")
        print(f"   Features common to ALL methods: {overlaps['overall_stats']['common_count']}")
        
        if 'random_forest_all' in self.feature_sets:
            print(f"\n🤖 Random Forest Feature Analysis:")
            for method, analysis in rf_analysis['comparisons'].items():
                print(f"   vs {method}: {analysis['overlap_percentage']:.1f}% overlap")


def main():
    """Main execution function."""
    comparator = FeatureSelectionComparator()
    comparator.run_complete_comparison()


if __name__ == "__main__":
    main()
