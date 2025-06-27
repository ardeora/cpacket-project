#!/usr/bin/env python3
"""
Quick Column Analyzer - View unique values for specific columns
"""

import pandas as pd
import sys

def analyze_column(csv_path, column_name=None):
    """Analyze specific column or list all columns."""
    df = pd.read_csv(csv_path)
    
    if column_name is None:
        print("Available columns:")
        print("=" * 50)
        for i, col in enumerate(df.columns, 1):
            unique_count = df[col].nunique()
            print(f"{i:2d}. {col} ({unique_count} unique values)")
        print("=" * 50)
        print(f"Total columns: {len(df.columns)}")
        print("Usage: python quick_column_analyzer.py <column_name>")
        return
    
    if column_name not in df.columns:
        print(f"❌ Column '{column_name}' not found!")
        print("Available columns:", ", ".join(df.columns))
        return
    
    print(f"Analysis for column: {column_name}")
    print("=" * 60)
    
    # Basic info
    print(f"Data type: {df[column_name].dtype}")
    print(f"Total values: {len(df[column_name])}")
    print(f"Non-null values: {df[column_name].count()}")
    print(f"Null values: {df[column_name].isnull().sum()}")
    print(f"Unique values: {df[column_name].nunique()}")
    print()
    
    # Unique values
    unique_vals = df[column_name].dropna().unique()
    print("Unique values:")
    print("-" * 30)
    
    if len(unique_vals) <= 50:
        for i, val in enumerate(sorted(unique_vals), 1):
            count = (df[column_name] == val).sum()
            percentage = (count / len(df)) * 100
            print(f"{i:3d}. {val} (count: {count}, {percentage:.1f}%)")
    else:
        print(f"Too many unique values ({len(unique_vals)}) to display all.")
        print("First 20 unique values:")
        for i, val in enumerate(sorted(unique_vals)[:20], 1):
            count = (df[column_name] == val).sum()
            percentage = (count / len(df)) * 100
            print(f"{i:3d}. {val} (count: {count}, {percentage:.1f}%)")
        print(f"... and {len(unique_vals) - 20} more values")
    
    # Statistics for numerical columns
    if df[column_name].dtype in ['int64', 'float64']:
        print("\nStatistical Summary:")
        print("-" * 30)
        stats = df[column_name].describe()
        for stat, value in stats.items():
            print(f"{stat}: {value}")

if __name__ == "__main__":
    csv_path = "datasets/attack-tcp-flag-osyn.csv"
    
    if len(sys.argv) > 1:
        column_name = sys.argv[1]
        analyze_column(csv_path, column_name)
    else:
        analyze_column(csv_path)
