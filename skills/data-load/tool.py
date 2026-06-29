from __future__ import annotations

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    try:
        from utils.data_utils import load_data as _load_data
    except Exception as e:
        raise ImportError(f"utils.data_utils.load_data import failed: {e}")
    return _load_data(str(file_path))
