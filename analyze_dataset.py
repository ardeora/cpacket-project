#!/usr/bin/env python3
"""
Dataset Analysis Script for attack-tcp-flag-osyn.csv
Creates a comprehensive analysis webpage with statistics and unique values for each column.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import base64
import io

def load_and_analyze_dataset(csv_path):
    """Load the dataset and perform comprehensive analysis."""
    print(f"Loading dataset from: {csv_path}")
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Basic information
    analysis_results = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        },
        'data_types': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values_count': df.nunique().to_dict(),
        'unique_values': {},
        'statistical_summary': {},
        'categorical_summary': {}
    }
    
    # Get unique values for each column
    print("Analyzing unique values for each column...")
    for col in df.columns:
        unique_vals = df[col].unique()
        # Limit display to reasonable number for readability
        if len(unique_vals) <= 50:
            analysis_results['unique_values'][col] = sorted([str(val) for val in unique_vals if pd.notna(val)])
        else:
            analysis_results['unique_values'][col] = {
                'count': len(unique_vals),
                'sample': sorted([str(val) for val in unique_vals[:20] if pd.notna(val)]),
                'note': f'Showing first 20 out of {len(unique_vals)} unique values'
            }
    
    # Statistical summary for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats_df = df[numerical_cols].describe()
        analysis_results['statistical_summary'] = stats_df.to_dict()
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        analysis_results['categorical_summary'][col] = {
            'value_counts': value_counts.to_dict(),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None
        }
    
    return df, analysis_results

def create_visualizations(df):
    """Create visualizations for the dataset."""
    print("Creating visualizations...")
    
    plots = {}
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Distribution of label column
    if 'label' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        label_counts = df['label'].value_counts()
        ax.bar(label_counts.index, label_counts.values, color='skyblue', edgecolor='navy')
        ax.set_title('Distribution of Labels', fontsize=16, fontweight='bold')
        ax.set_xlabel('Label', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['label_distribution'] = fig_to_base64(fig)
        plt.close()
    
    # 2. Correlation heatmap for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        # Limit to first 20 numerical columns for readability
        cols_to_plot = numerical_cols[:20] if len(numerical_cols) > 20 else numerical_cols
        fig, ax = plt.subplots(figsize=(12, 10))
        correlation_matrix = df[cols_to_plot].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax, fmt='.2f')
        ax.set_title('Correlation Heatmap (Top 20 Numerical Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plots['correlation_heatmap'] = fig_to_base64(fig)
        plt.close()
    
    # 3. Distribution of some key numerical features
    key_features = ['packets_rate', 'fwd_packets_rate', 'bwd_packets_rate', 'mean_header_bytes']
    available_features = [col for col in key_features if col in df.columns]
    
    if available_features:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features[:4]):
            if i < len(axes):
                axes[i].hist(df[feature].dropna(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel(feature, fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plots['feature_distributions'] = fig_to_base64(fig)
        plt.close()
    
    # 4. Activity distribution
    if 'activity' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        activity_counts = df['activity'].value_counts()
        ax.bar(activity_counts.index, activity_counts.values, color='lightgreen', edgecolor='darkgreen')
        ax.set_title('Distribution of Activity Types', fontsize=16, fontweight='bold')
        ax.set_xlabel('Activity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots['activity_distribution'] = fig_to_base64(fig)
        plt.close()
    
    return plots

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return image_base64

def generate_html_report(analysis_results, plots, output_path):
    """Generate a comprehensive HTML report."""
    print(f"Generating HTML report: {output_path}")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Analysis Report - Attack TCP Flag OSYN</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            border-radius: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px 10px 0 0;
            margin: -20px -20px 30px -20px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-top: 3px solid #667eea;
        }}
        
        .stat-card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }}
        
        tr:hover {{
            background: #f5f5f5;
        }}
        
        .unique-values {{
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
        }}
        
        .unique-values h4 {{
            color: #17a2b8;
            margin-bottom: 10px;
        }}
        
        .values-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}
        
        .value-tag {{
            background: #17a2b8;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        
        .plot-container {{
            text-align: center;
            margin: 30px 0;
        }}
        
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .plot-title {{
            font-size: 1.3em;
            color: #333;
            margin-bottom: 15px;
            font-weight: bold;
        }}
        
        .accordion {{
            margin: 20px 0;
        }}
        
        .accordion-item {{
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        
        .accordion-header {{
            background: #f8f9fa;
            padding: 15px;
            cursor: pointer;
            border-radius: 5px;
            font-weight: bold;
            transition: background 0.3s;
        }}
        
        .accordion-header:hover {{
            background: #e9ecef;
        }}
        
        .accordion-content {{
            padding: 15px;
            display: none;
        }}
        
        .accordion-content.active {{
            display: block;
        }}
        
        .timestamp {{
            text-align: center;
            color: #666;
            font-style: italic;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
    <script>
        function toggleAccordion(element) {{
            const content = element.nextElementSibling;
            content.classList.toggle('active');
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Dataset Analysis Report</h1>
            <p>Comprehensive Analysis of attack-tcp-flag-osyn.csv</p>
        </div>
        
        <div class="section">
            <h2>📈 Dataset Overview</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Rows</h3>
                    <div class="stat-value">{analysis_results['basic_info']['total_rows']:,}</div>
                </div>
                <div class="stat-card">
                    <h3>Total Columns</h3>
                    <div class="stat-value">{analysis_results['basic_info']['total_columns']}</div>
                </div>
                <div class="stat-card">
                    <h3>Memory Usage</h3>
                    <div class="stat-value">{analysis_results['basic_info']['memory_usage']}</div>
                </div>
                <div class="stat-card">
                    <h3>Missing Values</h3>
                    <div class="stat-value">{sum(analysis_results['missing_values'].values())}</div>
                </div>
            </div>
        </div>
"""

    # Add visualizations if available
    if plots:
        html_content += """
        <div class="section">
            <h2>📊 Data Visualizations</h2>
"""
        
        plot_titles = {
            'label_distribution': 'Distribution of Labels',
            'activity_distribution': 'Distribution of Activity Types',
            'correlation_heatmap': 'Feature Correlation Heatmap',
            'feature_distributions': 'Key Feature Distributions'
        }
        
        for plot_key, plot_data in plots.items():
            title = plot_titles.get(plot_key, plot_key.replace('_', ' ').title())
            html_content += f"""
            <div class="plot-container">
                <div class="plot-title">{title}</div>
                <img src="data:image/png;base64,{plot_data}" alt="{title}">
            </div>
"""
        
        html_content += "</div>"

    # Add unique values section
    html_content += """
        <div class="section">
            <h2>🔍 Unique Values Analysis</h2>
            <div class="highlight">
                <strong>Note:</strong> This section shows all unique values for each column. For columns with many unique values, only a sample is displayed.
            </div>
            <div class="accordion">
"""

    for col in analysis_results['basic_info']['column_names']:
        unique_data = analysis_results['unique_values'][col]
        unique_count = analysis_results['unique_values_count'][col]
        
        html_content += f"""
                <div class="accordion-item">
                    <div class="accordion-header" onclick="toggleAccordion(this)">
                        📋 {col} ({unique_count} unique values)
                    </div>
                    <div class="accordion-content">
"""
        
        if isinstance(unique_data, dict) and 'sample' in unique_data:
            html_content += f"""
                        <p><strong>Data Type:</strong> {analysis_results['data_types'][col]}</p>
                        <p><strong>Total Unique Values:</strong> {unique_data['count']}</p>
                        <p><strong>Missing Values:</strong> {analysis_results['missing_values'][col]}</p>
                        <div class="unique-values">
                            <h4>Sample Values (First 20):</h4>
                            <div class="values-list">
"""
            for val in unique_data['sample']:
                html_content += f'<span class="value-tag">{val}</span>'
            
            html_content += f"""
                            </div>
                            <p style="margin-top: 10px; font-style: italic;">{unique_data['note']}</p>
                        </div>
"""
        else:
            html_content += f"""
                        <p><strong>Data Type:</strong> {analysis_results['data_types'][col]}</p>
                        <p><strong>Total Unique Values:</strong> {unique_count}</p>
                        <p><strong>Missing Values:</strong> {analysis_results['missing_values'][col]}</p>
                        <div class="unique-values">
                            <h4>All Unique Values:</h4>
                            <div class="values-list">
"""
            for val in unique_data:
                html_content += f'<span class="value-tag">{val}</span>'
            
            html_content += """
                            </div>
                        </div>
"""
        
        html_content += """
                    </div>
                </div>
"""

    html_content += "</div></div>"

    # Add statistical summary
    if analysis_results['statistical_summary']:
        html_content += """
        <div class="section">
            <h2>📊 Statistical Summary</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Count</th>
                            <th>Mean</th>
                            <th>Std</th>
                            <th>Min</th>
                            <th>25%</th>
                            <th>50%</th>
                            <th>75%</th>
                            <th>Max</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for feature, stats in analysis_results['statistical_summary'].items():
            html_content += f"""
                        <tr>
                            <td><strong>{feature}</strong></td>
                            <td>{stats.get('count', 'N/A')}</td>
                            <td>{stats.get('mean', 0):.4f}</td>
                            <td>{stats.get('std', 0):.4f}</td>
                            <td>{stats.get('min', 0):.4f}</td>
                            <td>{stats.get('25%', 0):.4f}</td>
                            <td>{stats.get('50%', 0):.4f}</td>
                            <td>{stats.get('75%', 0):.4f}</td>
                            <td>{stats.get('max', 0):.4f}</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>
"""

    # Add categorical summary
    if analysis_results['categorical_summary']:
        html_content += """
        <div class="section">
            <h2>📝 Categorical Features Summary</h2>
"""
        
        for col, summary in analysis_results['categorical_summary'].items():
            html_content += f"""
            <div class="unique-values">
                <h4>{col}</h4>
                <p><strong>Most Common:</strong> {summary['most_common']}</p>
                <p><strong>Least Common:</strong> {summary['least_common']}</p>
                <div style="margin-top: 10px;">
                    <strong>Value Counts:</strong>
                    <div class="table-container">
                        <table style="margin-top: 10px;">
                            <thead>
                                <tr><th>Value</th><th>Count</th></tr>
                            </thead>
                            <tbody>
"""
            
            for value, count in summary['value_counts'].items():
                html_content += f"<tr><td>{value}</td><td>{count}</td></tr>"
            
            html_content += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
"""
        
        html_content += "</div>"

    # Add footer
    html_content += f"""
        <div class="timestamp">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Analysis completed for {analysis_results['basic_info']['total_rows']:,} records across {analysis_results['basic_info']['total_columns']} features</p>
        </div>
    </div>
</body>
</html>
"""

    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report saved to: {output_path}")

def main():
    """Main function to run the analysis."""
    print("🚀 Starting Dataset Analysis...")
    print("=" * 50)
    
    # Define paths
    csv_path = "datasets/attack-tcp-flag-osyn.csv"
    output_path = "dataset_analysis_report.html"
    
    # Check if dataset exists
    if not Path(csv_path).exists():
        print(f"❌ Error: Dataset not found at {csv_path}")
        return
    
    try:
        # Load and analyze dataset
        df, analysis_results = load_and_analyze_dataset(csv_path)
        
        # Create visualizations
        plots = create_visualizations(df)
        
        # Generate HTML report
        generate_html_report(analysis_results, plots, output_path)
        
        print("=" * 50)
        print("✅ Analysis completed successfully!")
        print(f"📄 Report saved as: {output_path}")
        print(f"📊 Dataset contains {analysis_results['basic_info']['total_rows']:,} rows and {analysis_results['basic_info']['total_columns']} columns")
        print(f"💾 Memory usage: {analysis_results['basic_info']['memory_usage']}")
        print("=" * 50)
        
        # Save analysis results as JSON for future use
        json_output = output_path.replace('.html', '_data.json')
        with open(json_output, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = json.loads(json.dumps(analysis_results, default=str))
            json.dump(serializable_results, f, indent=2)
        print(f"📊 Analysis data saved as: {json_output}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
