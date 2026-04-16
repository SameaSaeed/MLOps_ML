# src/features/engineer_dask.py
import dask.dataframe as dd
import numpy as np
from datetime import datetime
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feature-engineering')

def create_features(df):
    """Create new features from existing data (Dask-friendly)."""
    logger.info("Creating new features")
    
    df_featured = df.copy()
    
    # House age
    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df_featured['year_built']
    
    # Price per sqft
    df_featured['price_per_sqft'] = df_featured['price'] / df_featured['sqft']
    
    # Bedroom to bathroom ratio
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    # Replace inf with 0
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info("Features created: house_age, price_per_sqft, bed_bath_ratio")
    return df_featured

def create_preprocessor():
    """Create a preprocessing pipeline."""
    logger.info("Creating preprocessor pipeline")
    
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'price_per_sqft', 'bed_bath_ratio']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    """Full Dask-based feature engineering pipeline."""
    logger.info(f"Loading data from {input_file}")
    
    # Load data using Dask
    df = dd.read_csv(input_file)
    
    # Create features lazily
    df_featured = create_features(df)
    
    # Bring into memory for scikit-learn
    df_featured_pd = df_featured.compute()
    logger.info(f"Created featured dataset with shape: {df_featured_pd.shape}")
    
    # Preprocessing
    preprocessor = create_preprocessor()
    X = df_featured_pd.drop(columns=['price'], errors='ignore')
    y = df_featured_pd['price'] if 'price' in df_featured_pd.columns else None
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Fitted the preprocessor and transformed the features")
    
    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Saved preprocessor to {preprocessor_file}")
    
    # Save preprocessed data
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['price'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Saved fully preprocessed data to {output_file}")
    
    return df_transformed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dask Feature Engineering for Housing Data.')
    parser.add_argument('--input', required=True, help='Path to cleaned CSV file')
    parser.add_argument('--output', required=True, help='Path for output CSV file')
    parser.add_argument('--preprocessor', required=True, help='Path to save preprocessor')
    
    args = parser.parse_args()
    
    run_feature_engineering(args.input, args.output, args.preprocessor)
