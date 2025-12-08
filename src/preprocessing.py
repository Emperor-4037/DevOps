from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd

def get_preprocessing_pipeline(config):
    """
    Builds a ColumnTransformer based on the config dictionary.
    
    Config keys:
    - num_cols: list of numeric column names
    - cat_cols: list of categorical column names
    - num_impute: 'mean', 'median', 'constant'
    - scaler: 'StandardScaler', 'MinMaxScaler', 'None'
    - cat_impute: 'most_frequent', 'constant'
    - encoder: 'OneHotEncoder', 'OrdinalEncoder'
    """
    
    num_transformers = []
    if config.get('num_impute'):
        strategy = config['num_impute']
        num_transformers.append(('imputer', SimpleImputer(strategy=strategy)))
        
    if config.get('scaler') and config['scaler'] != 'None':
        if config['scaler'] == 'StandardScaler':
            num_transformers.append(('scaler', StandardScaler()))
        elif config['scaler'] == 'MinMaxScaler':
            num_transformers.append(('scaler', MinMaxScaler()))
            
    num_pipeline = Pipeline(num_transformers) if num_transformers else 'passthrough'
    
    cat_transformers = []
    if config.get('cat_impute'):
        strategy = config['cat_impute']
        cat_transformers.append(('imputer', SimpleImputer(strategy=strategy, fill_value='missing')))
        
    if config.get('encoder'):
        if config['encoder'] == 'OneHotEncoder':
            cat_transformers.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        else:
            cat_transformers.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            
    cat_pipeline = Pipeline(cat_transformers) if cat_transformers else 'passthrough'
    
    transformers = []
    if config.get('num_cols'):
        transformers.append(('num', num_pipeline, config['num_cols']))
    if config.get('cat_cols'):
        transformers.append(('cat', cat_pipeline, config['cat_cols']))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    return preprocessor
