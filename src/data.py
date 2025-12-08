import pandas as pd
import io

def load_data(file):
    """
    Load data from a CSV or Parquet file buffer/path.
    
    Args:
        file: file-like object or str path
        
    Returns:
        pd.DataFrame
    """
    if isinstance(file, str):
        if file.endswith('.parquet'):
            return pd.read_parquet(file)
        else:
            return pd.read_csv(file)
    
    # For UploadedFile (Streamlit)
    if hasattr(file, 'name'):
        if file.name.endswith('.parquet'):
            return pd.read_parquet(file)
        else:
            return pd.read_csv(file)
            
    # Fallback default assuming csv
    return pd.read_csv(file)
