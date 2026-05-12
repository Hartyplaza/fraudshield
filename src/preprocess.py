"""
Data preprocessing and feature engineering for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from loguru import logger

from src.config import RAW_DATA_FILE, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def load_raw_data(filepath=RAW_DATA_FILE) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Shape: {df.shape} | Fraud rate: {df[TARGET_COLUMN].mean():.4%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from Time and Amount."""
    df = df.copy()

    # Time features — hour of day and day of week from seconds
    df['hour']       = (df['Time'] // 3600) % 24
    df['day']        = (df['Time'] // 86400).astype(int)

    # Amount features
    df['log_amount'] = np.log1p(df['Amount'])  # log-transform skewed Amount
    df['amount_bin'] = pd.cut(df['Amount'],
                               bins=[0, 10, 50, 200, 1000, 999999],
                               labels=['micro','small','medium','large','xlarge'])

    # Interaction: high amount at unusual hour
    df['high_amount_night'] = ((df['Amount'] > 200) & (df['hour'] < 6)).astype(int)

    # Drop original Time and Amount (replaced by engineered versions)
    df.drop(columns=['Time', 'Amount'], inplace=True)

    return df


def get_feature_groups(df: pd.DataFrame):
    """Identify numeric and categorical columns."""
    exclude = [TARGET_COLUMN]
    numeric_cols     = df.select_dtypes(include=['int64','float64']).columns.tolist()
    numeric_cols     = [c for c in numeric_cols if c not in exclude]
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols=None):
    """
    Build preprocessing pipeline.
    Use RobustScaler for fraud detection — it's less sensitive to outliers
    which are common in fraud data.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  RobustScaler()),  # RobustScaler handles outliers better
    ])

    transformers = [('num', numeric_transformer, numeric_cols)]

    if categorical_cols:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    df = engineer_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    from src.config import PROCESSED_TRAIN, PROCESSED_TEST
    PROCESSED_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy(); train_df[TARGET_COLUMN] = y_train.values
    test_df  = X_test.copy();  test_df[TARGET_COLUMN]  = y_test.values

    train_df.to_csv(PROCESSED_TRAIN, index=False)
    test_df.to_csv(PROCESSED_TEST, index=False)
    logger.info(f"Saved train {train_df.shape} and test {test_df.shape}")
