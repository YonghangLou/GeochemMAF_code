from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

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


def clean_data(
    df: pd.DataFrame,
    *,
    decide_json: Callable[..., Dict[str, Any]],
    data_analyzer: Any = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger("data.clean")
    log.info("Cleaning data (strategy-driven mode)")
    df_cleaned = df.copy()

    if data_analyzer is not None and hasattr(data_analyzer, "analyze_data_quality"):
        try:
            quality = data_analyzer.analyze_data_quality(df_cleaned)
        except Exception:
            quality = _fallback_quality(df_cleaned)
    else:
        quality = _fallback_quality(df_cleaned)

    total_missing = int(df_cleaned.isnull().sum().sum())
    total_duplicates = int(quality.get("duplicates", 0) or 0)
    needs_cleaning_missing = total_missing > 0
    needs_cleaning_duplicates = total_duplicates > 0
    if not needs_cleaning_missing and not needs_cleaning_duplicates:
        log.info("Data integrity check passed: no missing values and no duplicate rows were detected.")
        task_context = "The dataset has no missing values and no duplicate rows."
    else:
        log.info(
            f"Data cleaning is required: detected {total_missing} missing values and {total_duplicates} duplicate rows."
        )
        task_context = f"Missing-value summary: {quality.get('missing_values')}\nDuplicate rows: {quality.get('duplicates')}"

    task = (
        "\nAnalyze the following data-quality summary and decide the data-cleaning strategy parameters.\n\n"
        f"Columns: {list(df.columns)}\n"
        f"{task_context}\n\n"
        "Return the cleaning strategy as JSON only, without code.\n"
        "Format:\n"
        "{\n"
        '    "drop_columns_threshold": 0.5,\n'
        '    "impute_strategy_numeric": "median",\n'
        '    "impute_strategy_categorical": "mode",\n'
        '    "remove_duplicates": true,\n'
        '    "handle_outliers": true,\n'
        '    "reasoning": "brief rationale"\n'
        "}\n"
    )
    default_strategy = {
        "drop_columns_threshold": 0.5,
        "impute_strategy_numeric": "median",
        "impute_strategy_categorical": "mode",
        "remove_duplicates": True,
        "handle_outliers": True,
    }
    strategy = decide_json(task, default_strategy, config=config)
    if not isinstance(strategy, dict):
        strategy = default_strategy
    log.info(f"Adopted cleaning strategy: {strategy}")

    try:
        drop_th = float(strategy.get("drop_columns_threshold", 1.0))
    except Exception:
        drop_th = 1.0
    protected_cols = {"fid", "ore", "longitude", "latitude", "id", "label", "target"}
    if drop_th < 1.0 and len(df_cleaned) > 0:
        missing_ratio = df_cleaned.isnull().sum() / len(df_cleaned)
        cols_to_drop = [
            col
            for col in missing_ratio[missing_ratio > drop_th].index
            if str(col).strip().lower() not in protected_cols
        ]
        if len(cols_to_drop) > 0:
            df_cleaned = df_cleaned.drop(columns=cols_to_drop)
            log.info(f"Dropped columns with more than {drop_th * 100}% missing values: {list(cols_to_drop)}")

    if bool(strategy.get("remove_duplicates", True)) and total_duplicates > 0:
        original_len = int(len(df_cleaned))
        df_cleaned = df_cleaned.drop_duplicates()
        log.info(f"Removed {original_len - len(df_cleaned)} duplicate rows")

    for col in df_cleaned.columns:
        if int(df_cleaned[col].isnull().sum()) <= 0:
            continue
        if str(df_cleaned[col].dtype) in {"int64", "float64"}:
            method = str(strategy.get("impute_strategy_numeric", "median"))
            if method == "mean":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif method == "median":
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif method == "zero":
                df_cleaned[col] = df_cleaned[col].fillna(0)
        else:
            method = str(strategy.get("impute_strategy_categorical", "mode"))
            if method == "mode":
                m = df_cleaned[col].mode()
                if not m.empty:
                    df_cleaned[col] = df_cleaned[col].fillna(m.iloc[0])

    if bool(strategy.get("handle_outliers", True)):
        numeric_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns
        exclude_cols = ["FID", "Ore", "Longitude", "Latitude", "id", "label", "target"]
        cols_to_process = [
            col
            for col in numeric_cols
            if col not in exclude_cols and not any((ex in str(col).lower() for ex in ["id", "code"]))
        ]
        for col in cols_to_process:
            q1 = df_cleaned[col].quantile(0.25)
            q3 = df_cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        log.info("Applied outlier clipping with the IQR rule")

    log.info(f"Cleaned data shape: {df_cleaned.shape}")
    return df_cleaned
