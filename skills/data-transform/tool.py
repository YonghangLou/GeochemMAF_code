from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd


def _fallback_distributions(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for col in numeric_cols:
        s = df[col]
        try:
            skewness = float(s.skew())
        except Exception:
            skewness = 0.0
        try:
            min_v = float(s.min())
        except Exception:
            min_v = 0.0
        try:
            max_v = float(s.max())
        except Exception:
            max_v = 0.0
        out[str(col)] = {"skewness": skewness, "min": min_v, "max": max_v}
    return out


def _assess_strict_compositional(df: pd.DataFrame, cols: list[str]) -> Dict[str, Any]:
    if not cols:
        return {"is_strict_compositional": False, "row_sum_cv": None, "non_positive_ratio": None}
    subset = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = subset.dropna(axis=0, how="any")
    if valid.empty:
        return {"is_strict_compositional": False, "row_sum_cv": None, "non_positive_ratio": None}
    vals = valid.to_numpy(dtype=float)
    non_positive_ratio = float((vals <= 0).sum() / vals.size) if vals.size > 0 else 1.0
    row_sums = valid.sum(axis=1).to_numpy(dtype=float)
    mean_sum = float(np.mean(row_sums)) if row_sums.size > 0 else 0.0
    std_sum = float(np.std(row_sums)) if row_sums.size > 0 else 0.0
    if mean_sum == 0:
        row_sum_cv = float("inf")
    else:
        row_sum_cv = abs(std_sum / mean_sum)
    is_strict = bool(non_positive_ratio == 0.0 and row_sum_cv <= 0.01)
    return {
        "is_strict_compositional": is_strict,
        "row_sum_cv": row_sum_cv,
        "non_positive_ratio": non_positive_ratio,
    }


def transform_data(
    df: pd.DataFrame,
    *,
    decide_json: Callable[..., Dict[str, Any]],
    data_analyzer: Any = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger("data.transform")
    log.info("Transforming data (strategy-driven mode)")
    df_transformed = df.copy()

    if data_analyzer is not None and hasattr(data_analyzer, "analyze_distributions"):
        try:
            distributions = data_analyzer.analyze_distributions(df_transformed)
        except Exception:
            distributions = _fallback_distributions(df_transformed)
    else:
        distributions = _fallback_distributions(df_transformed)

    numeric_cols = df_transformed.select_dtypes(include=["int64", "float64"]).columns.tolist()
    protected_cols = {"fid", "ore", "longitude", "latitude", "id", "label", "target"}
    dist_summary: Dict[str, Dict[str, Any]] = {}
    if isinstance(distributions, dict):
        for col, stats_val in distributions.items():
            if (
                col in numeric_cols
                and isinstance(stats_val, dict)
                and str(col).strip().lower() not in protected_cols
                and not any((ex in str(col).lower() for ex in ["id", "code"]))
            ):
                dist_summary[str(col)] = {
                    "skewness": stats_val.get("skewness", 0),
                    "min": stats_val.get("min", 0),
                    "max": stats_val.get("max", 0),
                }

    transformable_cols = sorted(
        [
            col
            for col in numeric_cols
            if str(col).strip().lower() not in protected_cols
            and not any((ex in str(col).lower() for ex in ["id", "code"]))
        ],
        key=lambda x: str(x).lower(),
    )
    compositional_info = _assess_strict_compositional(df_transformed, transformable_cols)
    is_strict_compositional = bool(compositional_info.get("is_strict_compositional"))
    row_sum_cv = compositional_info.get("row_sum_cv")
    non_positive_ratio = compositional_info.get("non_positive_ratio")
    if is_strict_compositional:
        allowed_methods = ["clr", "alr", "ilr"]
    else:
        allowed_methods = ["clr", "alr", "ilr", "box-cox", "log"]
    methods_text = "\n".join([f'{idx + 1}. "{name}"' for idx, name in enumerate(allowed_methods)])
    log.info(
        f"Strict compositional check: is_strict={is_strict_compositional}, "
        f"row_sum_cv={row_sum_cv}, non_positive_ratio={non_positive_ratio}"
    )
    task = (
        "\nBased on the distribution summary below, choose one transformation method for all transformable numeric columns.\n\n"
        f"Number of transformable columns: {len(transformable_cols)}\n"
        f"Distribution summary:\n{dist_summary}\n\n"
        "Strict compositional assessment:\n"
        f"- is_strict_compositional={is_strict_compositional}\n"
        f"- row_sum_cv={row_sum_cv}\n"
        f"- non_positive_ratio={non_positive_ratio}\n\n"
        "Allowed methods. Choose exactly one and do not return none:\n"
        f"{methods_text}\n\n"
        "Return JSON only.\n"
        "Format:\n"
        "{\n"
        '    "global_method": "clr",\n'
        '    "reasoning": "brief rationale"\n'
        "}\n"
        "Constraints:\n"
        "- Use one shared method for all transformable numeric columns.\n"
        "- Do not transform ID columns, code columns, or coordinate columns.\n"
        "- Do not return none.\n"
    )
    default_payload: Dict[str, Any] = {"global_method": "clr", "reasoning": ""}
    strategy_data = decide_json(task, default_payload, config=config)
    if not isinstance(strategy_data, dict):
        strategy_data = default_payload
    selected_method_raw = str(strategy_data.get("global_method", "clr")).strip().lower()
    selected_method = selected_method_raw
    if selected_method in {"none", ""}:
        selected_method = "clr"
    if selected_method not in set(allowed_methods):
        selected_method = "clr"
    reasoning = str(strategy_data.get("reasoning", "")).strip()
    log.info(f"Global transformation method selected: {selected_method} (raw={selected_method_raw})")
    if reasoning:
        log.info(f"Global transformation rationale: {reasoning}")
    log.info(f"Applying the global transformation to {len(transformable_cols)} columns")

    method_to_apply = selected_method
    if method_to_apply == "box-cox":
        try:
            from scipy import stats  # type: ignore
        except Exception:
            stats = None
        if stats is None:
            log.warning("Box-Cox is unavailable. Falling back to CLR.")
            method_to_apply = "clr"
    else:
        stats = None

    if method_to_apply == "clr" and transformable_cols:
        try:
            subset = df_transformed[transformable_cols].astype(float).copy()
            for col in transformable_cols:
                col_min = float(subset[col].min())
                if col_min <= 0:
                    subset[col] = subset[col] + abs(col_min) + 1e-09
            subset = subset.replace(0, 1e-09)
            geom_mean = subset.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
            for col in transformable_cols:
                df_transformed[col] = np.log(subset[col] / geom_mean)
            log.info(f"Applied the global CLR transformation to {len(transformable_cols)} columns")
        except Exception as e:
            log.error(f"Global CLR transformation failed: {e}")
    elif method_to_apply == "alr" and transformable_cols:
        try:
            subset = df_transformed[transformable_cols].astype(float).copy()
            for col in transformable_cols:
                col_min = float(subset[col].min())
                if col_min <= 0:
                    subset[col] = subset[col] + abs(col_min) + 1e-09
            subset = subset.replace(0, 1e-09)
            ref_col = str(transformable_cols[-1])
            ref_vals = subset[ref_col].astype(float).replace(0, 1e-09)
            for col in transformable_cols:
                if str(col) == ref_col:
                    df_transformed[col] = 0.0
                else:
                    df_transformed[col] = np.log(subset[col] / ref_vals)
            log.info(f"Applied the global ALR transformation to {len(transformable_cols)} columns (reference={ref_col})")
        except Exception as e:
            log.error(f"Global ALR transformation failed: {e}")
    elif method_to_apply == "ilr" and transformable_cols:
        try:
            subset = df_transformed[transformable_cols].astype(float).copy()
            for col in transformable_cols:
                col_min = float(subset[col].min())
                if col_min <= 0:
                    subset[col] = subset[col] + abs(col_min) + 1e-09
            subset = subset.replace(0, 1e-09)
            dim = len(transformable_cols)
            if dim < 2:
                geom_mean = subset.apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
                for col in transformable_cols:
                    df_transformed[col] = np.log(subset[col] / geom_mean)
                log.warning("ILR requires at least two columns. Falling back to CLR.")
            else:
                helmert = np.zeros((dim, dim - 1))
                for j in range(1, dim):
                    denom = np.sqrt(j * (j + 1))
                    helmert[:j, j - 1] = 1.0 / denom
                    helmert[j, j - 1] = -float(j) / denom
                log_subset = np.log(subset.to_numpy(dtype=float))
                ilr_coords = log_subset @ helmert
                for idx, col in enumerate(transformable_cols[:-1]):
                    df_transformed[col] = ilr_coords[:, idx]
                df_transformed[transformable_cols[-1]] = 0.0
                log.info(
                    f"Applied the global ILR transformation to {len(transformable_cols)} columns "
                    f"(stored in the first {len(transformable_cols) - 1} columns)"
                )
        except Exception as e:
            log.error(f"Global ILR transformation failed: {e}")
    elif method_to_apply in {"box-cox", "log"}:
        for col in transformable_cols:
            try:
                col_data = df_transformed[col].astype(float)
                if method_to_apply == "box-cox":
                    if float(col_data.min()) <= 0:
                        shift = abs(float(col_data.min())) + 1
                        col_data_shifted = col_data + shift
                    else:
                        col_data_shifted = col_data
                    transformed_data, _ = stats.boxcox(col_data_shifted)  # type: ignore[union-attr]
                    df_transformed[col] = transformed_data
                elif method_to_apply == "log":
                    if float(col_data.min()) <= 0:
                        col_data = col_data + abs(float(col_data.min())) + 1e-09
                    col_data = col_data.replace(0, 1e-09)
                    df_transformed[col] = np.log10(col_data)
            except Exception as e:
                log.warning(f"Failed to apply global {method_to_apply} to {col}. Falling back to CLR: {e}")
                try:
                    col_data = df_transformed[col].astype(float)
                    col_min = float(col_data.min())
                    if col_min <= 0:
                        col_data = col_data + abs(col_min) + 1e-09
                    col_data = col_data.replace(0, 1e-09)
                    col_gm = float(np.exp(np.mean(np.log(col_data))))
                    df_transformed[col] = np.log(col_data / col_gm)
                except Exception as e2:
                    log.error(f"CLR fallback failed for {col}: {e2}")
        if method_to_apply == "log":
            log.info(f"Applied the global log10 transformation to {len(transformable_cols)} columns")

    log.info(f"Transformed data shape: {df_transformed.shape}")
    return df_transformed
