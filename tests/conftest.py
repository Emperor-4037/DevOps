import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'A': np.random.rand(100),
        'B': np.random.randint(0, 100, 100),
        'cat': np.random.choice(['x', 'y', 'z'], 100),
        'target': np.random.choice([0, 1], 100),
        'target_reg': np.random.rand(100) * 10
    })
    return df
