from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def normalize_coordinates(
    df: pd.DataFrame,
    *,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
) -> Dict[str, Any]:
    try:
        from utils.data_utils import normalize_coordinates as _normalize_coordinates
    except Exception as e:
        raise ImportError(f"utils.data_utils.normalize_coordinates import failed: {e}")
    out_df, meta = _normalize_coordinates(df, x_col=x_col, y_col=y_col, lon_col=str(lon_col), lat_col=str(lat_col))
    return {"df": out_df, "meta": meta}
