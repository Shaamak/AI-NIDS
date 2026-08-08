"""
Phase 2: Dataset Analysis & EDA for UNSW-NB15
---------------------------------------------
This script performs comprehensive Exploratory Data Analysis (EDA) on the UNSW-NB15 dataset.

Concepts Covered:
1. Data Integrity: Missing values, duplicates, unexpected data types.
2. Data Leakage Detection: Identifying columns (like 'id') that could cause artificial inflated metrics.
3. Class Imbalance: Analyzing the distribution of Normal vs. 9 different Attack types.
4. Categorical vs. Numerical Feature Mapping.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")

def load_data():
    train_path = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files missing in {DATA_DIR}. Run download_data.py first!"
        )

    print("[INFO] Loading UNSW-NB15 training and testing CSVs...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"[INFO] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    return df_train, df_test

def analyze_dataset(df: pd.DataFrame, dataset_name: str = "Training Set"):
    print(f"\n=================== {dataset_name.upper()} ANALYSIS ===================")
    
    # 1. Missing Values
    null_counts = df.isnull().sum()
    missing_cols = null_counts[null_counts > 0]
    print(f"\n1. Missing Values Count:")
    if len(missing_cols) == 0:
        print("   -> No missing values detected!")
    else:
        print(missing_cols)

    # 2. Duplicates
    num_duplicates = df.duplicated().sum()
    print(f"\n2. Duplicate Rows: {num_duplicates}")

    # 3. Target Distribution (Binary & Multi-class)
    print(f"\n3. Target Distribution (label - Binary):")
    if 'label' in df.columns:
        print(df['label'].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    print(f"\n4. Multi-class Attack Category Distribution (attack_cat):")
    if 'attack_cat' in df.columns:
        # Standardize attack_cat whitespace
        df['attack_cat'] = df['attack_cat'].str.strip()
        print(df['attack_cat'].value_counts())

    # 5. Feature Types
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\n5. Feature Counts -> Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")
    print(f"   Categorical columns: {cat_cols}")

    # 6. Potential Data Leakage Columns
    leakage_candidates = [col for col in ['id', 'label'] if col in df.columns]
    print(f"\n6. Potential Data Leakage / ID Columns to remove: {leakage_candidates}")

def plot_distributions(df: pd.DataFrame):
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "notebooks", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Clean string column
    df['attack_cat'] = df['attack_cat'].str.strip()

    # Plot 1: Attack Category Frequency
    plt.figure(figsize=(12, 6))
    sns.countplot(y='attack_cat', data=df, order=df['attack_cat'].value_counts().index, palette='viridis')
    plt.title('UNSW-NB15 Attack Category Distribution')
    plt.xlabel('Count')
    plt.ylabel('Attack Category')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'attack_category_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Saved attack distribution plot to {plot_path}")

def main():
    df_train, df_test = load_data()
    analyze_dataset(df_train, "Training Set")
    analyze_dataset(df_test, "Testing Set")
    plot_distributions(df_train)

if __name__ == "__main__":
    main()
