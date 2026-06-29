from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd


def _fallback_quality(df: pd.DataFrame) -> Dict[str, Any]:
    missing_values: Dict[str, Dict[str, Any]] = {}
    total_rows = int(len(df))
    for col in df.columns:
        miss = int(df[col].isnull().sum())
        if miss <= 0:
            continue
        ratio = (miss / total_rows * 100.0) if total_rows > 0 else 0.0
        missing_values[str(col)] = {"count": miss, "ratio": ratio}
    try:
        duplicates = int(df.duplicated().sum())
    except Exception:
        duplicates = 0
    return {"missing_values": missing_values, "duplicates": duplicates}


def validate_data(df: pd.DataFrame, *, data_analyzer: Any = None, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    log = logger or logging.getLogger("data.validate")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if df.empty:
        return {"ok": False, "reason": "empty_df"}

    try:
        from utils.data_utils import detect_coordinate_columns as _detect_coordinate_columns
    except Exception as e:
        raise ImportError(f"detect_coordinate_columns import failed: {e}")

    coord_x, coord_y = _detect_coordinate_columns(df)

    if data_analyzer is not None and hasattr(data_analyzer, "analyze_data_quality"):
        try:
            quality = data_analyzer.analyze_data_quality(df)
        except Exception:
            quality = _fallback_quality(df)
    else:
        quality = _fallback_quality(df)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    coord_ok = bool(coord_x and coord_y and coord_x in df.columns and coord_y in df.columns)
    issues: List[str] = []
    if not coord_ok:
        issues.append("Coordinate columns were not detected or are incomplete.")
    if quality.get("duplicates", 0) and int(quality.get("duplicates", 0)) > 0:
        issues.append(f"Duplicate rows: {quality.get('duplicates')}")
    missing = quality.get("missing_values") if isinstance(quality.get("missing_values"), dict) else {}
    if missing:
        top_missing = sorted(missing.items(), key=lambda kv: float((kv[1] or {}).get("ratio", 0.0)), reverse=True)[:10]
        issues.append("Top missing-value columns: " + ", ".join([f"{k}({(v or {}).get('ratio')}%)" for k, v in top_missing]))

    payload = {
        "ok": True,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "numeric_columns_count": int(len(numeric_cols)),
        "coordinate_columns": {"x_col": coord_x, "y_col": coord_y, "ok": coord_ok},
        "quality": quality,
        "issues": issues,
    }
    log.info(f"Data validation issues: {len(issues)}")
    return payload
