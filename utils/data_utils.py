import os
import json
import tempfile
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple
logger = logging.getLogger(__name__)

_OUTPUT_LANG_MAP = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "chinese": "zh",
    "cn": "zh",
    "en": "en",
    "en-us": "en",
    "en_us": "en",
    "english": "en",
}

_TEXT_ZH_TO_EN: Dict[str, str] = {
    '\u7ecf\u5ea6': "Longitude",
    '\u7eac\u5ea6': "Latitude",
    '\u77ff\u5e8a': "Deposit",
    '\u6807\u7b7e': "Label",
    '\u805a\u7c7b\u7f16\u53f7': "Cluster ID",
    '\u6210\u77ff\u6f5c\u529b\u6982\u7387': "Mineral Potential Probability",
    '\u5df2\u77e5\u77ff\u5e8a': "Known Deposit",
    '\u5df2\u77e5\u77ff\u5e8a\u6837\u672c': "Known Deposit Samples",
    '\u5730\u7403\u5316\u5b66\u5f02\u5e38\u5206\u5e03\u56fe': "Geochemical Anomaly Distribution Map",
    '\u6837\u672c\u805a\u7c7b\u7a7a\u95f4\u7ed3\u679c\u56fe': "Sample Cluster Spatial Map",
    '\u6837\u672c\u6570\uff08\u5bf9\u6570\u523b\u5ea6\uff09': "Sample Count (Log Scale)",
    '\u6837\u672c\u6570': "Sample Count",
    '\u5f97\u5206': "Score",
    '\u5f97\u5206(0-1)': "Score (0-1)",
    '\u5f97\u5206\u4fe1\u53f7(0-1)': "Score Signals (0-1)",
    '\u8bc4\u6d4b\u5f97\u5206\u6982\u89c8': "Evaluation Score Overview",
    '\u6807\u7b7e\u5206\u5e03': "Label Distribution",
    '\u6821\u51c6\u6307\u6807': "Calibration Metrics",
    '\u6392\u5e8f\u6307\u6807': "Ranking Metrics",
    '\u8fc7\u7a0b\u6307\u6807': "Process Metrics",
    '\u673a\u5236\u6307\u6807': "Mechanism Metrics",
    '\u6b21\u6570': "Count",
    '\u667a\u80fd\u4f53\u8c03\u7528\u7edf\u8ba1': "Agent Call Statistics",
    'Token\u6570': "Token Count",
    '\u8f93\u5165(Prompt)': "Input (Prompt)",
    '\u8f93\u51fa(Completion)': "Output (Completion)",
    'Token\u4f7f\u7528\u60c5\u51b5': "Token Usage",
    '\u5143\u7d20': "Element",
    '\u542b\u91cf\u503c': "Concentration",
    '\u6d53\u5ea6': "Concentration",
    '\u6d53\u5ea6\u5206\u5e03': "Concentration Distribution",
    '\u9891\u7387': "Frequency",
    '\u6240\u6709\u5143\u7d20\u542b\u91cf\u5206\u5e03\u7bb1\u7ebf\u56fe': "Boxplot of All Element Concentrations",
    '\u7d2f\u79ef\u9891\u7387\u5206\u5e03': "Cumulative Frequency Distribution",
    'C-A\u65b9\u6cd5\u53cc\u5bf9\u6570\u56fe': "C-A Method Log-Log Plot",
    '\u5143\u7d20\u6d53\u5ea6 (\u5bf9\u6570\u523b\u5ea6)': "Element Concentration (Log Scale)",
    '\u7d2f\u79ef\u9762\u79ef\u767e\u5206\u6bd4 (%) (\u5bf9\u6570\u523b\u5ea6)': "Cumulative Area Percentage (%) (Log Scale)",
    '\u5404\u5143\u7d20\u5f02\u5e38\u6837\u672c\u767e\u5206\u6bd4': "Anomalous Sample Percentage by Element",
    '\u5f02\u5e38\u6837\u672c\u767e\u5206\u6bd4 (%)': "Anomalous Sample Percentage (%)",
    '\u5f02\u5e38\u9608\u503c': "Anomaly Threshold",
    '\u5f02\u5e38\u6837\u672c\u6570': "Anomaly Sample Count",
    '\u5f02\u5e38\u767e\u5206\u6bd4(%)': "Anomaly Percentage (%)",
    '\u9ad8\u503c\u9608\u503c': "High-Value Threshold",
    '\u9ad8\u503c\u6837\u672c\u6570': "High-Value Sample Count",
    '\u9ad8\u503c\u6bd4\u4f8b(%)': "High-Value Percentage (%)",
    '\u5143\u7d20\u7a7a\u95f4\u5206\u5e03\u56fe': "Element Spatial Distribution Map",
    '\u5173\u952e\u5143\u7d20\u5206\u6790': "Key Element Analysis",
    '\u6838\u5fc3\u5143\u7d20\u603bSHAP\u503c': "Total SHAP of Core Elements",
    '\u76f8\u5173\u7cfb\u6570': "Correlation Coefficient",
    '\u56e0\u5b50\u5e8f\u53f7': "Factor Index",
    '\u7279\u5f81\u503c': "Eigenvalue",
    '\u788e\u77f3\u56fe': "Scree Plot",
    '\u56e0\u5b50\u8f7d\u8377\u70ed\u529b\u56fe': "Factor Loading Heatmap",
    '\u5143\u7d20\u5c42\u6b21\u805a\u7c7b\u6811\u72b6\u56fe\uff08\u57fa\u4e8e\u76f8\u5173\u8ddd\u79bb\uff09': "Element Hierarchical Clustering Dendrogram (Correlation Distance)",
    '\u8ddd\u79bb (1 - Pearson r)': "Distance (1 - Pearson r)",
    '\u5c42\u6b21\u805a\u7c7b\u91cd\u6392\u540e\u7684\u76f8\u5173\u6027\u70ed\u529b\u56fe': "Reordered Correlation Heatmap by Hierarchical Clustering",
    '\u5143\u7d20\u76f8\u5173\u6027\u70ed\u529b\u56fe': "Element Correlation Heatmap",
    'API\u8bf7\u6c42\u9891\u7387\u968f\u65f6\u95f4\u53d8\u5316': "API Request Frequency Over Time",
    '\u65f6\u95f4': "Time",
    '\u8bf7\u6c42\u6570': "Request Count",
    '\u8bf7\u6c42\u5e8f\u53f7': "Request Index",
    '\u5355\u6b21\u8bf7\u6c42 Token \u6d88\u8017\u5206\u6790': "Per-request Token Usage Analysis",
    '\u603b\u5206(0-100)': "Overall Score (0-100)",
    '\u7ed3\u679c(\xd7100)': "Outcome (×100)",
    '\u8fc7\u7a0b(\xd7100)': "Process (×100)",
    '\u673a\u5236(\xd7100)': "Mechanism (×100)",
    '\u6b63\u4f8b\u6bd4\u4f8b': "Positive Rate",
    '\u603b\u8ba1': "Total",
    '\u68c0\u67e5\u901a\u8fc7\u7387': "Checks Pass Rate",
    '\u8fd4\u5de5\u7387': "Rework Rate",
    '\u9884\u7b97\u5229\u7528\u7387': "Budget Utilization",
    '\u51b3\u7b56\u7a33\u5b9a\u6027': "Decision Stability",
    '\u4eba\u5de5\u4ecb\u5165\u7387': "HITL Intervention Rate",
    '\u7ed3\u6784\u5316\u5931\u8d25\u7387': "Structured Failure Rate",
    'JSON\u4fee\u590d\u6210\u529f\u7387': "JSON Repair Success Rate",
    '\u53cd\u601d\u5f3a\u5ea6': "Reflection Intensity",
    '\u51b3\u7b56\u8c03\u7528\u6b21\u6570': "Decision Calls",
    '\u7ed3\u6784\u5316\u8c03\u7528\u6b21\u6570': "Structured Calls",
    '\u53cd\u601d\u6587\u672c\u8f6e\u6b21': "Reflection Text Rounds",
    'Brier\u5206\u6570': "Brier Score",
    '\u5bf9\u6570\u635f\u5931': "Log Loss",
    'ECE(10\u7bb1)': "ECE (10 bins)",
    'MCE(10\u7bb1)': "MCE (10 bins)",
    'Top1%\u7cbe\u5ea6': "Top 1% Precision",
    'Top1%\u53ec\u56de': "Top 1% Recall",
    'Top\u6b63\u4f8b\u6570\u7cbe\u5ea6': "Top Pos-count Precision",
    'Top\u6b63\u4f8b\u6570\u53ec\u56de': "Top Pos-count Recall",
    '\u5143\u7d20\u5f02\u5e38\u7a7a\u95f4\u5206\u5e03\u56fe': "Element Anomaly Spatial Distribution Map",
    'C-A\u9608\u503c': "C-A Threshold",
}

_CSV_HEADER_ZH_TO_EN: Dict[str, str] = {
    '\u5143\u7d20\u540d\u79f0': "Element Name",
    '\u5143\u7d20': "Element",
    '\u5f02\u5e38\u9608\u503c': "Anomaly Threshold",
    '\u5f02\u5e38\u6837\u672c\u6570': "Anomaly Sample Count",
    '\u5f02\u5e38\u767e\u5206\u6bd4(%)': "Anomaly Percentage (%)",
    '\u7ecf\u5ea6': "Longitude",
    '\u7eac\u5ea6': "Latitude",
    '\u805a\u7c7b\u7f16\u53f7': "Cluster ID",
    '\u6838\u5fc3\u5143\u7d20\u603bSHAP\u503c': "Total SHAP of Core Elements",
}
_CSV_HEADER_EN_TO_ZH: Dict[str, str] = {v: k for k, v in _CSV_HEADER_ZH_TO_EN.items()}


def normalize_output_language(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "en"
    return _OUTPUT_LANG_MAP.get(raw, "en")


def resolve_output_language(config: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(config, dict):
        cfg = config.get("output_language")
        if cfg is not None:
            return normalize_output_language(cfg)
    env_val = os.environ.get("GEOCHEM_OUTPUT_LANGUAGE") or os.environ.get("AGENTS_OUTPUT_LANGUAGE")
    return normalize_output_language(env_val)


def apply_output_language_env(config: Optional[Dict[str, Any]] = None) -> str:
    lang = resolve_output_language(config=config)
    os.environ["GEOCHEM_OUTPUT_LANGUAGE"] = lang
    os.environ["AGENTS_OUTPUT_LANGUAGE"] = lang
    return lang


def get_bilingual_text(zh_text: str, en_text: str, *, lang: Optional[str] = None) -> str:
    current = normalize_output_language(lang if lang is not None else resolve_output_language())
    return str(en_text) if current == "en" else str(zh_text)


def localize_text(text: Any, *, lang: Optional[str] = None) -> str:
    s = str(text or "")
    current = normalize_output_language(lang if lang is not None else resolve_output_language())
    if current == "en":
        return _TEXT_ZH_TO_EN.get(s, s)
    rev = {v: k for k, v in _TEXT_ZH_TO_EN.items()}
    return rev.get(s, s)


def localize_dataframe_headers(df: pd.DataFrame, *, lang: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    current = normalize_output_language(lang if lang is not None else resolve_output_language())
    mapping = _CSV_HEADER_ZH_TO_EN if current == "en" else _CSV_HEADER_EN_TO_ZH
    out = df.copy()
    out.rename(columns={c: mapping.get(str(c), str(c)) for c in out.columns}, inplace=True)
    return out


def setup_matplotlib_output_style(plt_module: Any = None) -> None:
    try:
        if plt_module is None:
            import matplotlib.pyplot as _plt
            plt_module = _plt
        current_lang = resolve_output_language()
        if str(current_lang).lower() == "zh":
            plt_module.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "sans-serif"]
            plt_module.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
        else:
            plt_module.rcParams["font.family"] = "Times New Roman"
            plt_module.rcParams["font.serif"] = ["Times New Roman"]
        plt_module.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Times New Roman", "Arial", "DejaVu Sans", "sans-serif"]
        plt_module.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        encodings = ['utf-8', 'gbk', 'gb2312', 'ISO-8859-1', 'latin1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read data with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to read data with encoding: {encoding}, trying next...")
                continue
        if df is None:
            raise ValueError(f"Unable to read file: {file_path}. All attempted encodings failed.")
        return df
    elif ext == '.xlsx' or ext == '.xls':
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported data-file format: {ext}")


def detect_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    possible_x_cols = ['\u7ecf\u5ea6', "longitude", "LONGITUDE", "Lon", "lon", "x", "X"]
    possible_y_cols = ['\u7eac\u5ea6', "latitude", "LATITUDE", "Lat", "lat", "y", "Y"]
    x_col = None
    y_col = None
    for col in df.columns:
        if col in possible_x_cols:
            x_col = str(col)
        elif col in possible_y_cols:
            y_col = str(col)
    if x_col is None or y_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = x_col or str(numeric_cols[0])
            y_col = y_col or str(numeric_cols[1])
    return x_col, y_col


def _maybe_swap_lon_lat(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    def _score(a: np.ndarray, a_limit: float, b: np.ndarray, b_limit: float) -> float:
        a_ok = np.nanmean(np.abs(a) <= a_limit) if a.size else 0.0
        b_ok = np.nanmean(np.abs(b) <= b_limit) if b.size else 0.0
        return float(a_ok + b_ok)

    score_as_lon_lat = _score(lon, 180.0, lat, 90.0)
    score_as_lat_lon = _score(lon, 90.0, lat, 180.0)
    if score_as_lat_lon > score_as_lon_lat + 0.1:
        return lat, lon, True
    return lon, lat, False


def normalize_coordinates(
    df: pd.DataFrame,
    *,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    lon_col: str = 'Longitude',
    lat_col: str = 'Latitude',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df, {"ok": False, "reason": "df_invalid_or_empty"}
    x_detected, y_detected = detect_coordinate_columns(df)
    x = str(x_col or x_detected or "").strip()
    y = str(y_col or y_detected or "").strip()
    if not x or not y or x not in df.columns or y not in df.columns:
        return df, {"ok": False, "reason": "coordinate_columns_not_found", "x_col": x, "y_col": y}

    out = df.copy()
    x_values = pd.to_numeric(out[x], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(out[y], errors="coerce").to_numpy(dtype=float)

    lon_values = x_values
    lat_values = y_values
    swapped = False

    lon_values, lat_values, swapped = _maybe_swap_lon_lat(lon_values, lat_values)

    is_projected = False
    if lon_values.size and lat_values.size:
        if np.nanmax(np.abs(lon_values)) > 360.0 or np.nanmax(np.abs(lat_values)) > 180.0:
            is_projected = True

    converted = False
    convert_error = None
    convert_method: Optional[str] = None
    convert_params: Dict[str, Any] = {}
    if is_projected:
        try:
            import pyproj  # type: ignore

            def _finite_sample(a: np.ndarray, b: np.ndarray, k: int = 60) -> Tuple[np.ndarray, np.ndarray]:
                ok = np.isfinite(a) & np.isfinite(b)
                if not np.any(ok):
                    return np.asarray([], dtype=float), np.asarray([], dtype=float)
                idx = np.flatnonzero(ok)
                if idx.size > k:
                    idx = idx[:k]
                return np.asarray(a[idx], dtype=float), np.asarray(b[idx], dtype=float)

            def _score_lonlat(lon: np.ndarray, lat: np.ndarray) -> float:
                ok = np.isfinite(lon) & np.isfinite(lat)
                if not np.any(ok):
                    return -1.0
                lon_ok = (lon[ok] >= -180.0) & (lon[ok] <= 180.0)
                lat_ok = (lat[ok] >= -90.0) & (lat[ok] <= 90.0)
                frac = float(np.mean(lon_ok & lat_ok))
                if frac <= 0:
                    return -1.0
                try:
                    lon_span = float(np.nanmax(lon[ok]) - np.nanmin(lon[ok]))
                    lat_span = float(np.nanmax(lat[ok]) - np.nanmin(lat[ok]))
                except Exception:
                    lon_span = 0.0
                    lat_span = 0.0
                span_bonus = 0.0
                if lon_span > 1e-6:
                    span_bonus += min(0.05, lon_span / 10000.0)
                if lat_span > 1e-6:
                    span_bonus += min(0.05, lat_span / 10000.0)
                return float(frac + span_bonus)

            def _apply(transformer: Any, x_arr: np.ndarray, y_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                lons, lats = transformer.transform(x_arr, y_arr)
                return np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)

            best_score = -1.0
            best_transformer = None
            best_meta: Dict[str, Any] = {}
            best_orientation = "xy"

            x_s, y_s = _finite_sample(lon_values, lat_values, k=60)
            y2_s, x2_s = _finite_sample(lat_values, lon_values, k=60)

            def _consider(*, name: str, transformer: Any, orientation: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
                nonlocal best_score, best_transformer, best_meta, best_orientation
                try:
                    if orientation == "xy":
                        lon_s, lat_s = _apply(transformer, x_s, y_s)
                    else:
                        lon_s, lat_s = _apply(transformer, y2_s, x2_s)
                    sc = _score_lonlat(lon_s, lat_s)
                except Exception:
                    return
                if sc > best_score:
                    best_score = sc
                    best_transformer = transformer
                    best_meta = {"method": name, **(extra_meta or {})}
                    best_orientation = orientation

            val1 = float(x_s[0]) if x_s.size else None
            zone_number = None
            if val1 is not None and 10000000.0 < val1 < 100000000.0:
                try:
                    zone_number = int(str(int(val1))[:2])
                except Exception:
                    zone_number = None
            if zone_number is not None and 1 <= zone_number <= 60:
                central_meridian = zone_number * 6 - 3
                pipeline = (
                    f"+proj=pipeline +step +inv +proj=tmerc +lat_0=0 +lon_0={central_meridian} +k=1 "
                    f"+x_0={zone_number * 1000000 + 500000} +y_0=0 +ellps=GRS80 +units=m "
                    "+step +proj=longlat +ellps=GRS80"
                )
                extra = {"zone_number": int(zone_number), "central_meridian": float(central_meridian)}
                _consider(name="tmerc_zone_prefix", transformer=pyproj.Transformer.from_pipeline(pipeline), orientation="xy", extra_meta=extra)
                _consider(name="tmerc_zone_prefix", transformer=pyproj.Transformer.from_pipeline(pipeline), orientation="yx", extra_meta=extra)

            abs_x = float(np.nanmax(np.abs(x_s))) if x_s.size else 0.0
            abs_y = float(np.nanmax(np.abs(y_s))) if y_s.size else 0.0
            if abs_x and abs_y and (abs_x > 1000.0 or abs_y > 1000.0):
                try:
                    _consider(
                        name="epsg:3857",
                        transformer=pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True),
                        orientation="xy",
                    )
                    _consider(
                        name="epsg:3857",
                        transformer=pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True),
                        orientation="yx",
                    )
                except Exception:
                    pass

            if abs_x and abs_y and abs_x < 2000000.0 and abs_y < 12000000.0:
                for zone in range(1, 61):
                    epsg = f"EPSG:326{zone:02d}"
                    try:
                        tr = pyproj.Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
                    except Exception:
                        continue
                    _consider(name=epsg, transformer=tr, orientation="xy")
                    _consider(name=epsg, transformer=tr, orientation="yx")

            if best_transformer is not None and best_score >= 0.80:
                if best_orientation == "xy":
                    lon_values, lat_values = _apply(best_transformer, lon_values, lat_values)
                else:
                    lon_values, lat_values = _apply(best_transformer, lat_values, lon_values)
                converted = True
                convert_method = str(best_meta.get("method") or "")
                convert_params = {k: v for k, v in best_meta.items() if k != "method"}
        except Exception as e:
            convert_error = str(e)

    out[lon_col] = lon_values
    out[lat_col] = lat_values

    meta: Dict[str, Any] = {
        "ok": True,
        "x_col": x,
        "y_col": y,
        "lon_col": lon_col,
        "lat_col": lat_col,
        "swapped_lon_lat": bool(swapped),
        "is_projected_input": bool(is_projected),
        "converted_to_lon_lat": bool(converted),
    }
    if convert_method:
        meta["convert_method"] = convert_method
    if convert_params:
        meta["convert_params"] = convert_params
    if convert_error:
        meta["convert_error"] = convert_error
    return out, meta


def atomic_write_text(file_path: str, content: str, encoding: str = "utf-8") -> None:
    file_path = os.path.abspath(file_path)
    target_dir = os.path.dirname(file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    suffix = os.path.splitext(file_path)[1] or ".txt"
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=target_dir or None, text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def atomic_write_markdown(file_path: str, content: str, encoding: str = "utf-8") -> None:
    atomic_write_text(file_path=file_path, content=content, encoding=encoding)


def atomic_write_json(obj: Any, file_path: str, encoding: str = "utf-8") -> None:
    content = json.dumps(obj, ensure_ascii=False, indent=2)
    atomic_write_text(file_path=file_path, content=content, encoding=encoding)


def atomic_write_csv(df: pd.DataFrame, file_path: str, **to_csv_kwargs: Any) -> None:
    file_path = os.path.abspath(file_path)
    target_dir = os.path.dirname(file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=target_dir or None)
    os.close(fd)
    try:
        localized_df = localize_dataframe_headers(df)
        localized_df.to_csv(tmp_path, **to_csv_kwargs)
        os.replace(tmp_path, file_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
