from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import pandas as pd


def scale_data(
    df: pd.DataFrame,
    *,
    decide: Callable[..., Any],
    scaler: Any = None,
    minmax_scaler: Any = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger("data.scale")
    log.info("Scaling data")
    df_scaled = df.copy()

    scaling_method = decide(
        "Decide the best feature-scaling strategy for the current dataset. Reply with a short text such as StandardScaler or MinMaxScaler.",
        config=config,
    )
    if isinstance(scaling_method, str) and "```python" in scaling_method:
        text_part = scaling_method.split("```python")[0].strip()
        log.info(f"Scaling method decided: {text_part} [code block omitted]")
    else:
        log.info(f"Scaling method: {scaling_method}")

    if scaler is None or minmax_scaler is None:
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler  # type: ignore
        except Exception:
            MinMaxScaler = None
            StandardScaler = None
        if scaler is None and StandardScaler is not None:
            scaler = StandardScaler()
        if minmax_scaler is None and MinMaxScaler is not None:
            minmax_scaler = MinMaxScaler()

    numeric_cols = df_scaled.select_dtypes(include=["int64", "float64"]).columns
    exclude_cols = ["FID", "Ore", "Longitude", "Latitude", "id", "label", "target"]
    cols_to_scale = [
        col
        for col in numeric_cols
        if col not in exclude_cols and not any((ex in str(col).lower() for ex in ["id", "code"]))
    ]
    if len(cols_to_scale) > 0 and scaler is not None and minmax_scaler is not None:
        if isinstance(scaling_method, str) and "minmax" in scaling_method.lower():
            df_scaled[cols_to_scale] = minmax_scaler.fit_transform(df_scaled[cols_to_scale])
            log.info("Applied MinMaxScaler to numeric feature columns")
        else:
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            log.info("Applied StandardScaler to numeric feature columns")

    log.info(f"Scaled data shape: {df_scaled.shape}")
    return df_scaled
