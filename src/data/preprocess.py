"""
Phase 3: Data Preprocessing Pipeline
-----------------------------------
This module builds a reusable, leakage-free preprocessing pipeline.

Important Machine Learning Principles:
1. Data Leakage Prevention: We fit encoders and scalers ONLY on the training set (X_train).
   Then we call .transform() on Validation and Test sets (and API input).
2. Saved Pipeline: We persist the trained scaler, encoders, and feature column ordering
   to `models/preprocessor.pkl` so FastAPI inference performs EXACTLY the same transformations.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

class NIDSPreprocessor:
    def __init__(self):
        self.column_transformer = None
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.categorical_cols = ['proto', 'service', 'state']
        self.drop_cols = ['id', 'label', 'attack_cat']

    def clean_raw_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Strip whitespace in categoricals
        for cat in self.categorical_cols:
            if cat in df.columns:
                df[cat] = df[cat].astype(str).str.strip()
        
        if 'attack_cat' in df.columns:
            df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
            # Handle potential NaNs or blanks in target
            df['attack_cat'] = df['attack_cat'].fillna('Normal')
            df['attack_cat'] = df['attack_cat'].replace({'': 'Normal', 'nan': 'Normal'})

        return df

    def fit_transform(self, train_df: pd.DataFrame):
        train_df = self.clean_raw_df(train_df)

        # Separate Features & Target
        y_train_raw = train_df['attack_cat']
        X_train_raw = train_df.drop(columns=[c for c in self.drop_cols if c in train_df.columns])

        # Identify numerical columns
        numerical_cols = [c for c in X_train_raw.columns if c not in self.categorical_cols]

        # Define ColumnTransformer (Scaler for numericals, OHE for categoricals)
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_cols)
            ]
        )

        # Fit transformer ONLY on X_train
        X_train_proc = self.column_transformer.fit_transform(X_train_raw)

        # Fit LabelEncoder ONLY on y_train
        y_train_proc = self.label_encoder.fit_transform(y_train_raw)

        # Extract generated feature names for traceability
        cat_encoder = self.column_transformer.named_transformers_['cat']
        encoded_cat_names = cat_encoder.get_feature_names_out(self.categorical_cols).tolist()
        self.feature_names = numerical_cols + encoded_cat_names

        print(f"[INFO] Processed Training Data Shape: {X_train_proc.shape}")
        print(f"[INFO] Target Classes ({len(self.label_encoder.classes_)}): {self.label_encoder.classes_}")

        return X_train_proc, y_train_proc

    def transform(self, df: pd.DataFrame, is_test: bool = True):
        df = self.clean_raw_df(df)

        y_proc = None
        if 'attack_cat' in df.columns and is_test:
            # Map unseen labels safely if needed
            known_classes = set(self.label_encoder.classes_)
            df['attack_cat'] = df['attack_cat'].apply(lambda x: x if x in known_classes else 'Normal')
            y_proc = self.label_encoder.transform(df['attack_cat'])

        X_raw = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors='ignore')
        X_proc = self.column_transformer.transform(X_raw)

        return X_proc, y_proc

def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_path = os.path.join(RAW_DIR, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(RAW_DIR, "UNSW_NB15_testing-set.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Raw dataset CSVs not found in data/raw! Run dataset download first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    preprocessor = NIDSPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test, y_test = preprocessor.transform(test_df, is_test=True)

    # Save processed arrays
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "processed_data.npz"),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    # Save Preprocessor Pipeline Object
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[SUCCESS] Preprocessing completed & pipeline saved to {preprocessor_path}")

if __name__ == "__main__":
    run_pipeline()
