import pytest
import pandas as pd
from src.preprocessing import get_preprocessing_pipeline
from src.training import train_model

def test_preprocessing_pipeline(sample_data):
    config = {
        'num_cols': ['A', 'B'],
        'cat_cols': ['cat'],
        'num_impute': 'mean',
        'scaler': 'StandardScaler',
        'cat_impute': 'most_frequent',
        'encoder': 'OneHotEncoder'
    }
    
    pipeline = get_preprocessing_pipeline(config)
    X = sample_data.drop(columns=['target', 'target_reg'])
    X_transformed = pipeline.fit_transform(X)
    
    # Check if shape is correct (100 rows, A, B, cat_x, cat_y, cat_z -> 5 cols)
    assert X_transformed.shape[0] == 100
    assert X_transformed.shape[1] >= 4 # At least

def test_model_training_classification(sample_data):
    config = {
        'drop': ['target_reg'],
        'num_cols': ['A', 'B'],
        'cat_cols': ['cat'],
        'encoder': 'OrdinalEncoder'
    }
    
    results = train_model(sample_data, 'target', 'Classification', ['Logistic Regression'], config, use_cv=False)
    assert 'Logistic Regression' in results
    assert 'accuracy' in results['Logistic Regression']['metrics']

def test_model_training_regression(sample_data):
    config = {
        'drop': ['target'],
        'num_cols': ['A', 'B'],
        'cat_cols': ['cat'],
        'encoder': 'OrdinalEncoder'
    }
    
    results = train_model(sample_data, 'target_reg', 'Regression', ['Linear Regression'], config, use_cv=False)
    assert 'Linear Regression' in results
    assert 'mse' in results['Linear Regression']['metrics']
