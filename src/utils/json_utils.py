import numpy as np
import pandas as pd
import json
from typing import Any

def json_serializable(obj: Any) -> Any:
    """
    Convert non-serializable types (numpy, pandas) to standard Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def clean_dict_nans(d: Any) -> Any:
    """
    Recursively remove NaNs from dictionaries and lists, replacing them with None.
    """
    if isinstance(d, dict):
        return {k: clean_dict_nans(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict_nans(v) for v in d]
    elif pd.isna(d):
        return None
    else:
        return d
