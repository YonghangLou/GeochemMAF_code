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
    "经度": "Longitude",
    "纬度": "Latitude",
    "矿床": "Deposit",
    "标签": "Label",
    "聚类编号": "Cluster ID",
    "成矿潜力概率": "Mineral Potential Probability",
    "已知矿床": "Known Deposit",
    "已知矿床样本": "Known Deposit Samples",
    "地球化学异常分布图": "Geochemical Anomaly Distribution Map",
    "样本聚类空间结果图": "Sample Cluster Spatial Map",
    "样本数（对数刻度）": "Sample Count (Log Scale)",
    "样本数": "Sample Count",
    "得分": "Score",
    "得分(0-1)": "Score (0-1)",
    "得分信号(0-1)": "Score Signals (0-1)",
    "评测得分概览": "Evaluation Score Overview",
    "标签分布": "Label Distribution",
    "校准指标": "Calibration Metrics",
    "排序指标": "Ranking Metrics",
    "过程指标": "Process Metrics",
    "机制指标": "Mechanism Metrics",
    "次数": "Count",
    "智能体调用统计": "Agent Call Statistics",
    "Token数": "Token Count",
    "输入(Prompt)": "Input (Prompt)",
    "输出(Completion)": "Output (Completion)",
    "Token使用情况": "Token Usage",
    "元素": "Element",
    "含量值": "Concentration",
    "浓度": "Concentration",
    "浓度分布": "Concentration Distribution",
    "频率": "Frequency",
    "所有元素含量分布箱线图": "Boxplot of All Element Concentrations",
    "累积频率分布": "Cumulative Frequency Distribution",
    "C-A方法双对数图": "C-A Method Log-Log Plot",
    "元素浓度 (对数刻度)": "Element Concentration (Log Scale)",
    "累积面积百分比 (%) (对数刻度)": "Cumulative Area Percentage (%) (Log Scale)",
    "各元素异常样本百分比": "Anomalous Sample Percentage by Element",
    "异常样本百分比 (%)": "Anomalous Sample Percentage (%)",
    "异常阈值": "Anomaly Threshold",
    "异常样本数": "Anomaly Sample Count",
    "异常百分比(%)": "Anomaly Percentage (%)",
    "高值阈值": "High-Value Threshold",
    "高值样本数": "High-Value Sample Count",
    "高值比例(%)": "High-Value Percentage (%)",
    "元素空间分布图": "Element Spatial Distribution Map",
    "关键元素分析": "Key Element Analysis",
    "核心元素总SHAP值": "Total SHAP of Core Elements",
    "相关系数": "Correlation Coefficient",
    "因子序号": "Factor Index",
    "特征值": "Eigenvalue",
    "碎石图": "Scree Plot",
    "因子载荷热力图": "Factor Loading Heatmap",
    "元素层次聚类树状图（基于相关距离）": "Element Hierarchical Clustering Dendrogram (Correlation Distance)",
    "距离 (1 - Pearson r)": "Distance (1 - Pearson r)",
    "层次聚类重排后的相关性热力图": "Reordered Correlation Heatmap by Hierarchical Clustering",
    "元素相关性热力图": "Element Correlation Heatmap",
    "API请求频率随时间变化": "API Request Frequency Over Time",
    "时间": "Time",
    "请求数": "Request Count",
    "请求序号": "Request Index",
    "单次请求 Token 消耗分析": "Per-request Token Usage Analysis",
    "总分(0-100)": "Overall Score (0-100)",
    "结果(×100)": "Outcome (×100)",
    "过程(×100)": "Process (×100)",
    "机制(×100)": "Mechanism (×100)",
    "正例比例": "Positive Rate",
    "总计": "Total",
    "检查通过率": "Checks Pass Rate",
    "返工率": "Rework Rate",
    "预算利用率": "Budget Utilization",
    "决策稳定性": "Decision Stability",
    "人工介入率": "HITL Intervention Rate",
    "结构化失败率": "Structured Failure Rate",
    "JSON修复成功率": "JSON Repair Success Rate",
    "反思强度": "Reflection Intensity",
    "决策调用次数": "Decision Calls",
    "结构化调用次数": "Structured Calls",
    "反思文本轮次": "Reflection Text Rounds",
    "Brier分数": "Brier Score",
    "对数损失": "Log Loss",
    "ECE(10箱)": "ECE (10 bins)",
    "MCE(10箱)": "MCE (10 bins)",
    "Top1%精度": "Top 1% Precision",
    "Top1%召回": "Top 1% Recall",
    "Top正例数精度": "Top Pos-count Precision",
    "Top正例数召回": "Top Pos-count Recall",
    "元素异常空间分布图": "Element Anomaly Spatial Distribution Map",
    "C-A阈值": "C-A Threshold",
}

_CSV_HEADER_ZH_TO_EN: Dict[str, str] = {
    "元素名称": "Element Name",
    "元素": "Element",
    "异常阈值": "Anomaly Threshold",
    "异常样本数": "Anomaly Sample Count",
    "异常百分比(%)": "Anomaly Percentage (%)",
    "经度": "Longitude",
    "纬度": "Latitude",
    "聚类编号": "Cluster ID",
    "核心元素总SHAP值": "Total SHAP of Core Elements",
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
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
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
            raise ValueError(f"无法读取文件: {file_path}，尝试了多种编码均失败")
        return df
    elif ext == '.xlsx' or ext == '.xls':
        return pd.read_excel(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"不支持的数据文件格式: {ext}")


def detect_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, None
    possible_x_cols = ["经度", "longitude", "LONGITUDE", "Lon", "lon", "x", "X"]
    possible_y_cols = ["纬度", "latitude", "LATITUDE", "Lat", "lat", "y", "Y"]
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
    lon_col: str = "经度",
    lat_col: str = "纬度",
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
