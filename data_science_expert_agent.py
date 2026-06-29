import os
import re
import sys
import json
import pandas as pd
import numpy as np
import ctypes
from ctypes import wintypes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.inspection import permutation_importance
from typing import Dict, List, Any, Optional, Tuple
import logging
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import RegularPolygon
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import time
import joblib
from scipy.stats import pearsonr
from itertools import product
try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent
SkillSpec = None
try:
    from .skills.core import SkillSpec as _SkillSpec
    SkillSpec = _SkillSpec
except Exception:
    try:
        from skills.core import SkillSpec as _SkillSpec2
        SkillSpec = _SkillSpec2
    except Exception:
        SkillSpec = None
logger = logging.getLogger(__name__)
try:
    from .utils.data_utils import atomic_write_csv as _atomic_write_csv
    from .utils.data_utils import atomic_write_text as _atomic_write_text
    from .utils.auto_programming import run_auto_programming as _run_auto_programming
    from .utils.data_utils import detect_coordinate_columns as _detect_coordinate_columns
    from .utils.data_utils import get_bilingual_text as _get_bilingual_text
    from .utils.data_utils import localize_dataframe_headers as _localize_dataframe_headers
    from .utils.data_utils import localize_text as _localize_text
    from .utils.data_utils import normalize_coordinates as _normalize_coordinates
    from .utils.data_utils import resolve_output_language as _resolve_output_language
    from .utils.data_utils import setup_matplotlib_output_style as _setup_matplotlib_output_style
except ImportError:
    from utils.data_utils import atomic_write_csv as _atomic_write_csv
    from utils.data_utils import atomic_write_text as _atomic_write_text
    from utils.auto_programming import run_auto_programming as _run_auto_programming
    from utils.data_utils import detect_coordinate_columns as _detect_coordinate_columns
    from utils.data_utils import get_bilingual_text as _get_bilingual_text
    from utils.data_utils import localize_dataframe_headers as _localize_dataframe_headers
    from utils.data_utils import localize_text as _localize_text
    from utils.data_utils import normalize_coordinates as _normalize_coordinates
    from utils.data_utils import resolve_output_language as _resolve_output_language
    from utils.data_utils import setup_matplotlib_output_style as _setup_matplotlib_output_style
try:
    from .data_analyzer import DataAnalyzer
except Exception:
    from data_analyzer import DataAnalyzer

try:
    from minisom import MiniSom
except Exception:
    MiniSom = None
try:
    import shap
except Exception:
    shap = None
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None
chinese_font = None
try:
    font_path = "C:/Windows/Fonts/simhei.ttf"
    chinese_font = FontProperties(fname=font_path)
    plt.rcParams.update({'font.sans-serif': ['SimHei', 'Microsoft YaHei'], 'axes.unicode_minus': False})
except Exception:
    chinese_font = None
try:
    from .result_output_agent import generate_prediction_map_idw
except Exception:
    try:
        from result_output_agent import generate_prediction_map_idw
    except Exception:
        generate_prediction_map_idw = None
try:
    from .utils.llm_utils import get_llm as _get_llm
except Exception:
    try:
        from utils.llm_utils import get_llm as _get_llm
    except Exception:
        _get_llm = None


def _can_stratify_binary(y_values: Any) -> bool:
    try:
        arr = np.asarray(y_values).reshape(-1)
        uniq = set(np.unique(arr).tolist())
        if not (uniq <= {0, 1}):
            return False
        return int(np.sum(arr == 1)) > 0 and int(np.sum(arr == 0)) > 0
    except Exception:
        return False


class SOMQEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        grid_m: Optional[int] = None,
        grid_n: Optional[int] = None,
        n_iter: int = 5000,
        sigma: float = 1.0,
        lr: float = 0.5,
        random_state: int = 42,
        calibrate: bool = True,
        decision_threshold: float = 0.5,
    ):
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.n_iter = n_iter
        self.sigma = sigma
        self.lr = lr
        self.random_state = random_state
        self.calibrate = calibrate
        self.decision_threshold = decision_threshold

    def _resolve_grid(self, n_samples: int) -> Tuple[int, int]:
        m = int(self.grid_m) if self.grid_m is not None else 0
        n = int(self.grid_n) if self.grid_n is not None else 0
        if m <= 0 or n <= 0:
            inner_max = max(float(n_samples), 1.0)
            inner_sqrt = float(np.sqrt(inner_max))
            outer_max = max(4.0, 5.0 * inner_sqrt)
            side = int(round(float(np.sqrt(outer_max))))
            side = max(2, min(25, side))
            m = side
            n = side
        return m, n

    def _as_float_matrix(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float32, copy=True)
        else:
            arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            arr = np.asarray(arr).reshape(int(arr.shape[0]), -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def fit(self, X: Any, y: Any = None):
        X_raw = self._as_float_matrix(X)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.scaler_ = MinMaxScaler()
        X01 = self.scaler_.fit_transform(X_raw)
        n_samples, n_features = X01.shape
        m, n = self._resolve_grid(n_samples)
        self.grid_shape_ = (int(m), int(n))
        rng = np.random.RandomState(int(self.random_state))
        weights = rng.uniform(low=0.0, high=1.0, size=(m, n, n_features)).astype(np.float32)
        xs = np.repeat(np.arange(m, dtype=np.float32), n)
        ys = np.tile(np.arange(n, dtype=np.float32), m)
        coords = np.stack([xs, ys], axis=1)
        self._coords_ = coords

        iters = int(max(1, self.n_iter))
        sigma0 = float(max(1e-3, self.sigma))
        lr0 = float(max(1e-4, self.lr))

        for t in range(iters):
            x = X01[int(rng.randint(0, n_samples))]
            w_flat = weights.reshape(m * n, n_features)
            d2 = np.sum((w_flat - x.reshape(1, -1)) ** 2, axis=1)
            bmu = int(np.argmin(d2))
            bx, by = coords[bmu]
            sigma_t = sigma0 * (1.0 - (float(t) / float(iters))) + 1e-3
            lr_t = lr0 * (1.0 - (float(t) / float(iters))) + 1e-4
            g_d2 = (coords[:, 0] - bx) ** 2 + (coords[:, 1] - by) ** 2
            h = np.exp(-g_d2 / (2.0 * sigma_t * sigma_t)).astype(np.float32)
            w_flat += (lr_t * h).reshape(-1, 1) * (x.reshape(1, -1) - w_flat)
            weights = w_flat.reshape(m, n, n_features)

        self.weights_ = weights.astype(np.float32)

        self.calibrator_ = None
        y_arr = None
        if y is not None:
            y_arr = np.asarray(y).reshape(-1)
        if self.calibrate and y_arr is not None and y_arr.size == n_samples:
            labeled_mask = np.isin(y_arr, [0, 1])
            if int(np.sum(labeled_mask)) > 0:
                y_labeled = y_arr[labeled_mask].astype(int)
                if len(set(np.unique(y_labeled).tolist())) >= 2:
                    qe = self._quantization_error(X01)[labeled_mask].reshape(-1, 1)
                    self.calibrator_ = LogisticRegression(max_iter=200, solver="lbfgs", random_state=int(self.random_state))
                    self.calibrator_.fit(qe, y_labeled)
        return self

    def _quantization_error(self, X01: np.ndarray) -> np.ndarray:
        m, n = self.grid_shape_
        w_flat = self.weights_.reshape(m * n, self.weights_.shape[-1])
        d2 = np.sum((X01[:, None, :] - w_flat[None, :, :]) ** 2, axis=2)
        bmu = np.argmin(d2, axis=1)
        qe = np.sqrt(d2[np.arange(int(X01.shape[0])), bmu])
        return qe.astype(np.float64)

    def decision_function(self, X: Any) -> np.ndarray:
        X_raw = self._as_float_matrix(X)
        X01 = self.scaler_.transform(X_raw)
        return self._quantization_error(X01)

    def predict_proba(self, X: Any) -> np.ndarray:
        scores = self.decision_function(X).reshape(-1, 1)
        if self.calibrator_ is not None:
            p1 = self.calibrator_.predict_proba(scores)[:, 1]
        else:
            s = scores.reshape(-1)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            if s.size <= 1:
                p1 = np.zeros_like(s, dtype=np.float64)
            else:
                lo = float(np.quantile(s, 0.01))
                hi = float(np.quantile(s, 0.99))
                if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
                    lo = float(np.min(s))
                    hi = float(np.max(s))
                denom = float(hi - lo)
                if denom < 1e-12:
                    p1 = np.zeros_like(s, dtype=np.float64)
                else:
                    p1 = np.clip((s - lo) / denom, 0.0, 1.0)
        p0 = 1.0 - np.asarray(p1, dtype=np.float64)
        return np.stack([p0, np.asarray(p1, dtype=np.float64)], axis=1)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(self.decision_threshold)).astype(int)


def _split_features_and_coords(X) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Tuple[str, str]]]:
    if isinstance(X, pd.DataFrame):
        x_col, y_col = _detect_coordinate_columns(X)
        if x_col and y_col and x_col in X.columns and y_col in X.columns:
            coords = np.asarray(X[[x_col, y_col]], dtype=np.float32)
            feats = np.asarray(X.drop(columns=[x_col, y_col]), dtype=np.float32)
            return feats, coords, (str(x_col), str(y_col))
        return np.asarray(X, dtype=np.float32), None, None
    return np.asarray(X, dtype=np.float32), None, None


def _get_process_memory_info_mb() -> Optional[Dict[str, float]]:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        rss = float(getattr(mem, "rss", 0.0))
        vms = float(getattr(mem, "vms", 0.0))
        peak = float(getattr(mem, "peak_wset", 0.0) or getattr(mem, "peak_rss", 0.0))
        private = float(getattr(mem, "private", 0.0) or getattr(mem, "private_bytes", 0.0))
        out = {
            "rss_mb": rss / (1024.0 * 1024.0) if rss else 0.0,
            "vms_mb": vms / (1024.0 * 1024.0) if vms else 0.0,
            "peak_mb": peak / (1024.0 * 1024.0) if peak else 0.0,
            "private_mb": private / (1024.0 * 1024.0) if private else 0.0,
        }
        return out
    except Exception:
        pass
    try:
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return {
                "rss_mb": float(counters.WorkingSetSize) / (1024.0 * 1024.0),
                "vms_mb": float(counters.PagefileUsage) / (1024.0 * 1024.0),
                "peak_mb": float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0),
                "private_mb": float(counters.PrivateUsage) / (1024.0 * 1024.0),
            }
    except Exception:
        return None
    return None


def _log_process_memory(logger_obj: logging.Logger, stage: str) -> None:
    info = _get_process_memory_info_mb()
    if not isinstance(info, dict) or not info:
        return
    parts = []
    for key in ("rss_mb", "peak_mb", "private_mb", "vms_mb"):
        val = info.get(key)
        if isinstance(val, (int, float)) and val > 0:
            parts.append(f"{key}={val:.2f}MB")
    if parts:
        logger_obj.info(f"Memory monitor ({stage}): " + ", ".join(parts))

_MODEL_KEY_ALIASES: Dict[str, str] = {
    "som": "som",
    "self organizing map": "som",
    "self-organizing map": "som",
    "selforganizingmap": "som",
    "self-organizing map": "som",
    "self-organizing map neural network": "som",
}


def _normalize_model_key(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    try:
        s = str(raw).strip()
    except Exception:
        return None
    if not s:
        return None
    low = s.lower().strip(" <>`\"'，。,;；:：")
    return _MODEL_KEY_ALIASES.get(low)


def _extract_ca_anomaly_indices(
    *,
    geology_expert_results: Optional[Dict[str, Any]],
    prefer_elements: Optional[List[str]] = None,
    top_k: int = 12,
) -> List[pd.Index]:
    if not isinstance(geology_expert_results, dict):
        return []
    anomaly_analysis = geology_expert_results.get("anomaly_analysis")
    element_anomalies = anomaly_analysis.get("element_anomalies", {}) if isinstance(anomaly_analysis, dict) else {}
    if not isinstance(element_anomalies, dict) or not element_anomalies:
        return []
    elements: List[str] = []
    if isinstance(prefer_elements, list) and prefer_elements:
        first = str(prefer_elements[0]).strip()
        if first:
            elements = [first]
    out: List[pd.Index] = []
    for e in elements:
        st = element_anomalies.get(e)
        if not isinstance(st, dict):
            continue
        samples = st.get("anomaly_samples")
        if isinstance(samples, list) and samples:
            out.append(pd.Index(samples))
    return out


def _derive_radius_from_ca(
    *,
    data_df: pd.DataFrame,
    geology_expert_results: Optional[Dict[str, Any]],
    prefer_elements: Optional[List[str]] = None,
    k_neighbor: int = 6,
) -> Optional[float]:
    if data_df is None or data_df.empty:
        return None
    x_col, y_col = _detect_coordinate_columns(data_df)
    if not x_col or not y_col or x_col not in data_df.columns or y_col not in data_df.columns:
        return None
    coords = np.asarray(data_df[[x_col, y_col]], dtype=np.float64)
    ok = np.isfinite(coords).all(axis=1)
    if not bool(np.any(ok)):
        return None
    coords_ok = coords[ok]
    idx_ok = data_df.index[ok]

    anomaly_analysis = geology_expert_results.get("anomaly_analysis") if isinstance(geology_expert_results, dict) else None
    element_anomalies = anomaly_analysis.get("element_anomalies", {}) if isinstance(anomaly_analysis, dict) else {}

    elements: List[str] = []
    if isinstance(prefer_elements, list) and prefer_elements:
        first = str(prefer_elements[0]).strip()
        if first and first in data_df.columns:
            elements = [first]
    if not elements and isinstance(geology_expert_results, dict):
        obj = geology_expert_results.get("target_related_elements")
        if isinstance(obj, list) and obj:
            first = str(obj[0]).strip()
            if first and first in data_df.columns:
                elements = [first]
    if not elements:
        return None

    def _box_count_radius(points: np.ndarray, source: str, point_count: int, elem_preview: List[str]) -> Optional[float]:
        if points is None or int(points.shape[0]) < 5:
            return None
        x = points[:, 0]
        y = points[:, 1]
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        lx = float(xmax - xmin)
        ly = float(ymax - ymin)
        span = float(max(lx, ly))
        if not (span > 0):
            return None
        r_min = span / 100.0
        r_max = span / 2.0
        if not (r_min > 0 and r_min < r_max):
            r_min = span / 10.0
            r_max = span / 2.0
            if not (r_min > 0 and r_min < r_max):
                return None
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), num=9)
        r_values_list = [float(x) for x in r_values.tolist()]
        radius = float(r_min)
        try:
            logger.info(
                "Box-counting radius estimate: source=%s, points=%s, elements=%s, r_min=%.6g, r_max=%.6g, r_candidates=%s, pick=%.6g",
                str(source),
                int(point_count),
                ",".join([str(x) for x in elem_preview]) if elem_preview else "NA",
                float(r_min),
                float(r_max),
                ",".join([f"{v:.6g}" for v in r_values_list]),
                float(radius),
            )
        except Exception:
            pass
        return radius if radius > 0 else None

    selected_mask = np.zeros(int(data_df.shape[0]), dtype=bool)
    for e in elements:
        try:
            st = element_anomalies.get(e) if isinstance(element_anomalies, dict) else None
            if not isinstance(st, dict):
                continue
            thr = st.get("threshold")
            try:
                thr_f = float(thr) if thr is not None else None
            except Exception:
                thr_f = None
            if thr_f is None or not (thr_f > 0):
                continue
            v = pd.to_numeric(data_df[e], errors="coerce").astype(float).to_numpy(dtype=np.float64)
            sel = ok & np.isfinite(v) & (v > float(thr_f))
            if int(np.sum(sel)) > 0:
                selected_mask = selected_mask | sel
        except Exception:
            continue
    elem_preview = [str(x) for x in elements[:6]]
    pts = coords[selected_mask] if int(np.sum(selected_mask)) >= 5 else None
    source = "ca_threshold_points"
    if pts is None:
        groups = _extract_ca_anomaly_indices(geology_expert_results=geology_expert_results, prefer_elements=elements, top_k=12)
        union_idx: List[int] = []
        for g in groups:
            try:
                if g is None or len(g) < 1:
                    continue
                keep = idx_ok.intersection(pd.Index(g))
                if len(keep) < 1:
                    continue
                union_idx.extend(idx_ok.get_indexer(keep).tolist())
            except Exception:
                continue
        if union_idx:
            uniq = np.unique(np.asarray(union_idx, dtype=np.int64))
            if int(uniq.size) >= 5:
                pts = coords_ok[uniq]
                source = "ca_anomaly_samples"
    if pts is None or int(pts.shape[0]) < 5:
        return None
    return _box_count_radius(pts, source, int(pts.shape[0]) if pts is not None else 0, elem_preview)


def _summarize_positive_spatial_profile(
    *,
    data_df: pd.DataFrame,
    y: Any,
    radius: Optional[float],
    k_neighbor: int = 6,
) -> Dict[str, Any]:
    if data_df is None or data_df.empty:
        return {"available": False, "reason": "empty_data"}
    x_col, y_col = _detect_coordinate_columns(data_df)
    if not x_col or not y_col or x_col not in data_df.columns or y_col not in data_df.columns:
        return {"available": False, "reason": "no_coord_cols"}
    try:
        y_arr = np.asarray(y)
    except Exception:
        return {"available": False, "reason": "bad_labels"}
    if y_arr.size != int(data_df.shape[0]):
        return {"available": False, "reason": "label_length_mismatch"}
    coords_all = np.asarray(data_df[[x_col, y_col]], dtype=np.float64)
    ok = np.isfinite(coords_all).all(axis=1)
    try:
        pos_mask = (y_arr == 1) & ok
    except Exception:
        pos_mask = ok & (pd.to_numeric(pd.Series(y_arr), errors="coerce").fillna(-999).to_numpy(dtype=float) == 1.0)
    coords_pos = coords_all[pos_mask]
    n_pos = int(coords_pos.shape[0])
    unit_guess = "projected_or_unknown" if (coords_pos.size and (float(np.nanmax(np.abs(coords_pos[:, 0]))) > 360.0 or float(np.nanmax(np.abs(coords_pos[:, 1]))) > 180.0)) else "degree_or_unknown"
    out: Dict[str, Any] = {
        "available": bool(n_pos > 0),
        "coord_cols": [str(x_col), str(y_col)],
        "pos_with_coords": n_pos,
        "coord_unit_guess": unit_guess,
    }
    if n_pos < 2:
        out["reason"] = "pos_with_coords_lt_2"
        return out
    k_eff = int(min(max(int(k_neighbor), 2), n_pos))
    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(coords_pos)
    dists, _ = nn.kneighbors(coords_pos, return_distance=True)
    d = np.asarray(dists[:, 1:], dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size:
        out["pos_knn_dist"] = {
            "k": int(k_eff - 1),
            "median": float(np.median(d)),
            "p10": float(np.quantile(d, 0.10)),
            "p25": float(np.quantile(d, 0.25)),
            "p75": float(np.quantile(d, 0.75)),
            "p90": float(np.quantile(d, 0.90)),
        }
    else:
        out["pos_knn_dist"] = {"k": int(k_eff - 1), "median": None, "p10": None, "p25": None, "p75": None, "p90": None}
    if radius is not None:
        try:
            r = float(radius)
        except Exception:
            r = -1.0
        if r > 0:
            nn_r = NearestNeighbors(radius=r, algorithm="ball_tree")
            nn_r.fit(coords_pos)
            neigh = nn_r.radius_neighbors(coords_pos, return_distance=False)
            counts = np.asarray([max(0, int(len(v)) - 1) for v in neigh], dtype=np.int64)
            out["pos_neighbors_within_radius"] = {
                "radius": float(r),
                "mean": float(np.mean(counts)),
                "median": float(np.median(counts)),
                "pct_has_neighbor": float(np.mean(counts > 0)),
                "max": int(np.max(counts)) if counts.size else 0,
            }
    return out


def _stdin_is_interactive() -> bool:
    try:
        return bool(getattr(sys.stdin, "isatty", lambda: False)())
    except Exception:
        return False


def _input_with_log_prefix(logger_obj: logging.Logger, level: str = "INFO") -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return input(f"{ts} - {logger_obj.name} - {level} - ")


def _hitl_enabled(state: dict) -> bool:
    cfg = state.get("config") if isinstance(state, dict) else None
    if not isinstance(cfg, dict):
        return False
    mode = cfg.get("interaction_mode")
    if isinstance(mode, str):
        return mode.strip().lower() in {"hitl", "human", "human_in_the_loop"}
    return bool(cfg.get("hitl_enabled", False))


def _hitl_disable(state: dict) -> None:
    cfg = state.get("config") if isinstance(state, dict) else None
    if isinstance(cfg, dict):
        cfg["interaction_mode"] = "auto"
class DataScienceExpertAgent(BaseAgent):
    CAPABILITIES = [
        "Data loading: CSV/Excel/JSON -> DataFrame",
        "Data-quality diagnostics: missing values, duplicates, numeric columns, and coordinate-column checks with reports",
        "Data cleaning: missing-value handling, duplicate removal, and IQR-based outlier treatment",
        "Data transformation: log/Box-Cox/square-root/CLR transforms and element-ratio features",
        "Data scaling: StandardScaler / MinMaxScaler",
        "Feature analysis: correlations, factor loadings, and feature-selection suggestions",
        "Modeling and prediction: SOM-based anomaly/potential scoring with quantization error (QE) and thresholding",
        "Model interpretation: QE-score distributions and high-potential sample indexing",
    ]
    def __init__(self, output_dir: str='./output', llm=None):
        role_description = (
            "Responsible for geochemical-data preprocessing, feature engineering, and the selection, training, "
            "and evaluation of prediction models. Choose appropriate preprocessing strategies and predictive "
            "models according to the data characteristics, and provide high-quality processed data and reliable "
            "prediction results for downstream analysis."
        )
        super().__init__('DataScienceExpertAgent', role_description, llm)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.models = {
            "som": SOMQEClassifier(random_state=42),
        }
        self.best_model = None
        self.model_name = None
        self.data_analyzer = DataAnalyzer()
        self._register_skills()
        self.logger.info(f'{self.agent_name} initialized')

    def _register_skills(self) -> None:
        reg = getattr(self, "skills", None)
        if reg is None or SkillSpec is None:
            return
        def _safe_register(spec: SkillSpec, fn) -> None:
            sid = str(getattr(spec, "id", "") or "").strip()
            if not sid:
                return
            try:
                if reg.has(sid):
                    return
            except Exception:
                pass
            reg.register(spec, fn)

        _safe_register(
            SkillSpec(
                id="data.load",
                name="Data loading",
                description="Load CSV/Excel/JSON data into a DataFrame",
                inputs={"file_path": "input data file path"},
                outputs={"df": "loaded DataFrame"},
                tags=("data", "io"),
            ),
            lambda *, ctx, file_path: self.load_data(str(file_path)),
        )
        _safe_register(
            SkillSpec(
                id="data.clean",
                name="Data cleaning",
                description="Handle missing values, duplicate rows, and outliers",
                inputs={"df": "raw input DataFrame"},
                outputs={"df": "cleaned DataFrame"},
                tags=("data", "preprocess"),
            ),
            lambda *, ctx, df: self.clean_data(df, config=ctx.config),
        )
        _safe_register(
            SkillSpec(
                id="data.transform",
                name="Data transformation",
                description="Apply log/Box-Cox/square-root/CLR transforms according to distribution characteristics",
                inputs={"df": "cleaned DataFrame"},
                outputs={"df": "transformed DataFrame"},
                tags=("data", "preprocess"),
            ),
            lambda *, ctx, df: self.transform_data(df, config=ctx.config),
        )
        _safe_register(
            SkillSpec(
                id="data.scale",
                name="Data scaling",
                description="Apply StandardScaler or MinMaxScaler to numeric columns",
                inputs={"df": "transformed DataFrame"},
                outputs={"df": "scaled DataFrame"},
                tags=("data", "preprocess"),
            ),
            lambda *, ctx, df: self.scale_data(df, config=ctx.config),
        )
        _safe_register(
            SkillSpec(
                id="data.validate",
                name="Data validation",
                description="Check missing values, duplicates, numeric columns, and coordinate columns and return diagnostics",
                inputs={"df": "input DataFrame"},
                outputs={"results": "validation-result dictionary"},
                tags=("data", "quality"),
            ),
            lambda *, ctx, df: self.validate_data(df),
        )

    def load_data(self, file_path: str) -> pd.DataFrame:
        fn = self._get_skill_tool_callable("data.load", "tool.py", "load_data")
        if fn is None:
            raise RuntimeError("data.load tool not found: skills/data-load/tool.py:load_data")
        return fn(str(file_path))

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        fn = self._get_skill_tool_callable("data.validate", "tool.py", "validate_data")
        if fn is None:
            raise RuntimeError("data.validate tool not found: skills/data-validate/tool.py:validate_data")
        return fn(df, data_analyzer=self.data_analyzer, logger=self.logger)

    def explain_model(self, model_results: Dict[str, Any], top_k: Optional[int] = None) -> Dict[str, Any]:
        if not isinstance(model_results, dict):
            raise TypeError("model_results must be a dict")
        best_name = model_results.get("best_model_name")
        best_name_display = self._display_model_name(best_name) if best_name else None
        fit_diag = model_results.get("best_model_fit_diagnosis")
        perm = model_results.get("best_model_permutation_importance")
        builtin = model_results.get("best_model_feature_importance")
        top_k_int: Optional[int]
        if top_k is None:
            top_k_int = None
        else:
            top_k_int = int(top_k)
            if top_k_int <= 0:
                top_k_int = None

        def _take(items: Any) -> List[Dict[str, Any]]:
            if not isinstance(items, list):
                return []
            out: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                feat = it.get("feature")
                if feat is None:
                    continue
                try:
                    imp = float(it.get("importance", 0.0))
                except Exception:
                    imp = 0.0
                out.append({"feature": str(feat), "importance": imp})
            if top_k_int is None:
                return out
            return out[:top_k_int]

        perm_top = _take(perm)
        builtin_top = _take(builtin)
        lines: List[str] = []
        lines.append(f"Best model: {best_name_display}" if best_name_display else "Best model: Unknown")
        if isinstance(fit_diag, dict) and fit_diag.get("status"):
            lines.append(f"Fit diagnosis: {fit_diag.get('status')}")
            recs = fit_diag.get("recommendations") or []
            if recs:
                lines.append("Recommendations: " + "; ".join([str(r) for r in recs]))
        if perm_top:
            lines.append("Top permutation importance:")
            lines.extend([f"- {it['feature']}: {it['importance']:.6f}" for it in perm_top])
        elif builtin_top:
            lines.append("Top built-in feature importance:")
            lines.extend([f"- {it['feature']}: {it['importance']:.6f}" for it in builtin_top])
        return {
            "best_model_name": best_name,
            "fit_diagnosis": fit_diag if isinstance(fit_diag, dict) else None,
            "permutation_importance_top": perm_top,
            "feature_importance_top": builtin_top,
            "summary_text": "\n".join(lines),
        }

    def _run_skill_or_fallback(self, ctx: Any, skill_id: str, fallback, **kwargs: Any) -> Any:
        reg = getattr(self, "skills", None)
        if reg is None:
            return fallback(**kwargs)
        try:
            res = reg.run(skill_id, ctx, **kwargs)
        except Exception:
            res = None
        if res is not None and getattr(res, "ok", False):
            return getattr(res, "output", None)
        return fallback(**kwargs)

    def _get_reports_dir(self) -> str:
        reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        return reports_dir

    def _get_model_viz_dir(self) -> str:
        viz_dir = os.path.join(self.output_dir, "model_viz")
        os.makedirs(viz_dir, exist_ok=True)
        return viz_dir

    def _relpath_from_reports(self, path: str) -> str:
        try:
            reports_dir = self._get_reports_dir()
            return os.path.relpath(str(path), reports_dir).replace("\\", "/")
        except Exception:
            return str(path)

    def _display_model_name(self, model_key: Any) -> str:
        if model_key is None:
            return ""
        raw = str(model_key).strip()
        norm = _normalize_model_key(raw)
        display = {
            "som": "SOM",
        }
        if norm and norm in display:
            return display[norm]
        return raw

    def _get_data_science_step_overrides(self, state: dict) -> Dict[str, Any]:
        if not isinstance(state, dict):
            return {}
        try:
            container = state.get("hitl_step_overrides")
            if isinstance(container, dict):
                obj = container.get("data_science_expert")
                if isinstance(obj, dict):
                    return obj
        except Exception:
            return {}
        return {}

    def _maybe_run_auto_programming(self, *, state: dict, df_scaled: pd.DataFrame) -> pd.DataFrame:
        cfg = state.get("config") if isinstance(state, dict) else None
        if isinstance(cfg, dict) and not bool(cfg.get("auto_programming_enabled", True)):
            return df_scaled
        step_overrides = self._get_data_science_step_overrides(state)
        if not step_overrides:
            return df_scaled
        if not bool(step_overrides.get("auto_programming_enabled", False)):
            return df_scaled
        request_text = str(step_overrides.get("programming_request") or "").strip()
        if not request_text:
            return df_scaled
        apply_mode = str(step_overrides.get("programming_apply_mode") or "attach_artifact").strip().lower()
        if apply_mode not in {"replace_df", "attach_artifact"}:
            apply_mode = "attach_artifact"
        timeout_seconds = 90
        if isinstance(cfg, dict):
            try:
                timeout_seconds = int(cfg.get("auto_programming_timeout_seconds", 90))
            except Exception:
                timeout_seconds = 90
        if timeout_seconds < 5:
            timeout_seconds = 5

        auto_prog_dir = os.path.join(self.output_dir, "auto_programming")
        os.makedirs(auto_prog_dir, exist_ok=True)
        result = _run_auto_programming(
            llm=self.llm,
            df=df_scaled,
            request_text=request_text,
            output_dir=auto_prog_dir,
            timeout_seconds=timeout_seconds,
            logger=self.logger,
        )

        state.setdefault("analysis_results", {})
        state["analysis_results"]["auto_programming"] = {
            "ok": bool(result.ok),
            "request": request_text,
            "apply_mode": apply_mode,
            "script_path": result.script_path,
            "work_dir": result.work_dir,
            "input_data_path": result.input_data_path,
            "output_data_path": result.output_data_path,
            "metrics_path": result.metrics_path,
            "stdout_tail": str(result.stdout or "")[-2000:],
            "stderr_tail": str(result.stderr or "")[-2000:],
            "error": str(result.error or ""),
            "metrics": result.metrics if isinstance(result.metrics, dict) else {},
        }
        state.setdefault("artifacts", {})
        if result.script_path:
            state["artifacts"]["auto_programming_script"] = os.path.abspath(result.script_path)
        if result.metrics_path:
            state["artifacts"]["auto_programming_metrics"] = os.path.abspath(result.metrics_path)

        if not result.ok:
            state.setdefault("errors", []).append(f"{self.agent_name}: auto-programming failed - {result.error}")
            state.setdefault("processing_history", []).append(f"{self.agent_name}: auto-programming failed")
            return df_scaled

        state.setdefault("processing_history", []).append(f"{self.agent_name}: auto-programming executed")
        if apply_mode != "replace_df":
            return df_scaled
        try:
            generated_df = pd.read_pickle(result.output_data_path)
            if not isinstance(generated_df, pd.DataFrame):
                raise TypeError("generated output is not a DataFrame")
            state.setdefault("processing_history", []).append(f"{self.agent_name}: auto-programming result replaced the current data")
            return generated_df
        except Exception as e:
            state.setdefault("errors", []).append(f"{self.agent_name}: failed to read the auto-programming result - {e}")
            return df_scaled

    def _plot_roc_curve(
        self,
        *,
        out_path: str,
        title: str,
        curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        y_true: Optional[np.ndarray] = None,
        scores: Optional[np.ndarray] = None,
        legend_style: str = "prefixed",
        force_color: Optional[str] = None,
    ) -> str:
        try:
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
            ax.plot([0, 1], [0, 1], color="#999999", lw=1, linestyle="--")
            palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
            plotted = 0
            if curves is None:
                if y_true is None or scores is None:
                    return ""
                curves = {"ROC": (y_true, scores)}
            for idx, (name, pair) in enumerate((curves or {}).items()):
                try:
                    y_arr, s_arr = pair
                    y_arr = np.asarray(y_arr).astype(int)
                    s_arr = np.asarray(s_arr, dtype=float)
                    if y_arr.size == 0 or s_arr.size == 0 or y_arr.size != s_arr.size:
                        continue
                    if len(set(np.unique(y_arr).tolist())) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_arr, s_arr)
                    roc_auc = auc(fpr, tpr)
                    color = str(force_color).strip() if force_color else palette[int(idx) % len(palette)]
                    legend_prefix = str(name).strip()
                    if legend_prefix.lower() in {"train", "training"}:
                        legend_prefix = "Train"
                    elif legend_prefix.lower() in {"val", "valid", "validation"}:
                        legend_prefix = "Val"
                    elif legend_prefix.lower() in {"test", "testing"}:
                        legend_prefix = "Test"
                    style = str(legend_style or "").strip().lower()
                    if style in {"auc", "auc_only", "only_auc"}:
                        label = f"AUC={roc_auc:.4f}"
                    else:
                        label = f"{legend_prefix}_AUC={roc_auc:.4f}"
                    ax.plot(fpr, tpr, color=color, lw=2, label=label)
                    plotted += 1
                except Exception:
                    continue
            if plotted == 0:
                plt.close(fig)
                return ""
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(str(title))
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the ROC curve: {e}")
            except Exception:
                pass
            return ""

    def _plot_pr_curve(self, *, y_true: np.ndarray, scores: np.ndarray, out_path: str, title: str) -> str:
        try:
            precision, recall, _ = precision_recall_curve(y_true, scores)
            pr_auc = auc(recall, precision)
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
            ax.plot(recall, precision, color="#d62728", lw=2, label=f"AUC={pr_auc:.4f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(str(title))
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the PR curve: {e}")
            except Exception:
                pass
            return ""

    def _plot_confusion_matrix_heatmap(self, *, cm: np.ndarray, out_path: str, title: str) -> str:
        try:
            fig, ax = plt.subplots(figsize=(5.8, 4.8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, linewidths=0.5, linecolor="white")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title(str(title))
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"], rotation=0)
            fig.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the confusion matrix: {e}")
            except Exception:
                pass
            return ""

    def _plot_probability_histogram(self, *, y_true: np.ndarray, scores: np.ndarray, out_path: str, title: str) -> str:
        try:
            y_arr = np.asarray(y_true).astype(int)
            s_arr = np.asarray(scores, dtype=float)
            mask = np.isfinite(s_arr)
            if mask.size != y_arr.size:
                mask = mask[: min(mask.size, y_arr.size)]
            y_arr = y_arr[: mask.size]
            s_arr = s_arr[: mask.size]
            y_arr = y_arr[mask]
            s_arr = s_arr[mask]
            if s_arr.size == 0 or y_arr.size == 0:
                return ""
            pos = s_arr[y_arr == 1]
            neg = s_arr[y_arr == 0]
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
            bins = 30
            if neg.size:
                ax.hist(neg, bins=bins, alpha=0.6, label="Negative (0)", color="#1f77b4", density=True)
            if pos.size:
                ax.hist(pos, bins=bins, alpha=0.6, label="Positive (1)", color="#d62728", density=True)
            ax.set_xlabel("Predicted score")
            ax.set_ylabel("Density")
            ax.set_title(str(title))
            ax.legend(loc="best")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the score distribution: {e}")
            except Exception:
                pass
            return ""

    def _plot_fit_gap(self, *, train_auc: Optional[float], val_auc: Optional[float], out_path: str, title: str) -> str:
        try:
            labels: List[str] = []
            values: List[float] = []
            if train_auc is not None:
                labels.append("train_auc")
                values.append(float(train_auc))
            if val_auc is not None:
                labels.append("val_auc")
                values.append(float(val_auc))
            if not values:
                return ""
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.bar(labels, values, color=["#1f77b4", "#2ca02c"][: len(values)])
            ax.set_ylim(0, 1.02)
            ax.set_ylabel("AUC")
            ax.set_title(str(title))
            for i, v in enumerate(values):
                ax.text(i, min(1.0, v) + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the fit-gap chart: {e}")
            except Exception:
                pass
            return ""

    def _plot_feature_importance_bar(self, *, importance: Any, out_path: str, title: str, top_n: Optional[int] = None, show_legend: bool = True) -> str:
        try:
            if not isinstance(importance, list) or not importance:
                return ""
            df = pd.DataFrame([x for x in importance if isinstance(x, dict)]).copy()
            if df.empty or "feature" not in df.columns or "importance" not in df.columns:
                return ""
            df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
            if "std" in df.columns:
                df["std"] = pd.to_numeric(df["std"], errors="coerce")
            df = df.sort_values("importance", ascending=False)
            if top_n is not None:
                n_keep = int(top_n)
                if n_keep > 0:
                    df = df.head(n_keep)
            df = df.iloc[::-1]
            n_rows = int(len(df))
            fig_h = max(4.8, 0.22 * n_rows + 2.2)
            fig, ax = plt.subplots(figsize=(9.2, fig_h))
            show_std = bool("std" in df.columns and df["std"].notna().any())
            if show_std:
                ax.barh(df["feature"].astype(str), df["importance"].astype(float), xerr=df["std"].astype(float).fillna(0.0), color="#4c78a8", alpha=0.9)
            else:
                ax.barh(df["feature"].astype(str), df["importance"].astype(float), color="#4c78a8", alpha=0.9)
            ax.margins(y=0)
            ax.set_ylim(-0.5, n_rows - 0.5)
            if show_std and bool(show_legend):
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch

                ax.legend(handles=[Patch(facecolor="#4c78a8", alpha=0.9, label="importance"), Line2D([0], [0], color="black", lw=1.2, label="±1 std")], loc="best", frameon=False)
            if n_rows <= 40:
                label_size = 10
            elif n_rows <= 80:
                label_size = 9
            elif n_rows <= 120:
                label_size = 8
            elif n_rows <= 200:
                label_size = 7
            else:
                label_size = 6
            ax.tick_params(axis="y", labelsize=label_size)
            ax.set_xlabel("Importance")
            ax.set_title(str(title))
            ax.grid(True, axis="x", alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the feature-importance chart: {e}")
            except Exception:
                pass
            return ""

    def _plot_learning_curve_from_history(self, *, history: Any, out_path: str, title: str) -> str:
        try:
            if history is None:
                return ""
            epochs: List[int] = []
            train_loss: List[float] = []
            val_loss: List[float] = []
            if isinstance(history, list) and history:
                if all((isinstance(x, (int, float, np.floating)) for x in history)):
                    train_loss = [float(x) for x in history]
                    epochs = list(range(1, len(train_loss) + 1))
                elif all((isinstance(x, dict) for x in history)):
                    for i, row in enumerate(history, 1):
                        epochs.append(int(row.get("epoch", i)))
                        if row.get("train_loss") is not None:
                            try:
                                train_loss.append(float(row.get("train_loss")))
                            except Exception:
                                train_loss.append(np.nan)
                        elif row.get("loss") is not None:
                            try:
                                train_loss.append(float(row.get("loss")))
                            except Exception:
                                train_loss.append(np.nan)
                        else:
                            train_loss.append(np.nan)
                        if row.get("val_loss") is not None:
                            try:
                                val_loss.append(float(row.get("val_loss")))
                            except Exception:
                                val_loss.append(np.nan)
            if not epochs or (not train_loss and not val_loss):
                return ""
            fig, ax = plt.subplots(figsize=(7.6, 5.2))
            if train_loss:
                ax.plot(epochs, train_loss, label="train_loss", color="#1f77b4", lw=2)
            if val_loss:
                ax.plot(epochs[: len(val_loss)], val_loss, label="val_loss", color="#d62728", lw=2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(str(title))
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the learning curve: {e}")
            except Exception:
                pass
            return ""
    def _plot_all_elements_boxplot(self, *, df: pd.DataFrame, out_path: str, title: str) -> str:
        try:
            numeric_df = df.select_dtypes(include=["int64", "float64"])
            if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:
                return ""
            cols: List[str] = []
            data: List[np.ndarray] = []
            for c in numeric_df.columns.tolist():
                s = numeric_df[c].replace([np.inf, -np.inf], np.nan).dropna()
                if s.empty:
                    continue
                cols.append(str(c))
                data.append(s.to_numpy())
            if not cols:
                return ""
            _setup_matplotlib_output_style(plt)
            plt.rcParams["font.family"] = ["Times New Roman", "serif"]
            plt.rcParams["font.serif"] = ["Times New Roman"]
            fig_w = min(28.0, max(10.0, 0.45 * len(cols)))
            fig, ax = plt.subplots(figsize=(fig_w, 6.8))
            ax.boxplot(data, labels=cols, showfliers=True, whis=1.5)
            ax.set_title(str(title), fontsize=22, fontname="Times New Roman")
            ax.tick_params(axis="x", labelrotation=90, labelsize=20)
            ax.tick_params(axis="y", labelsize=20)
            for label in ax.get_xticklabels():
                label.set_fontname("Times New Roman")
            for label in ax.get_yticklabels():
                label.set_fontname("Times New Roman")
            fig.tight_layout()
            fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white", format="png")
            plt.close(fig)
            return out_path
        except Exception as e:
            try:
                self.logger.warning(f"Failed to plot the all-element boxplot: {e}")
            except Exception:
                pass
            return ""
    def run(self, state: dict) -> dict:
        self.logger.info(f'{self.agent_name} running')
        try:
            if 'processing_history' not in state:
                state['processing_history'] = []
            if 'errors' not in state:
                state['errors'] = []
            df = state.get('data')
            if df is None:
                raise ValueError('Data not found. Please load data first.')
            geology_expert_results = state.get('geology_expert_results')
            if not geology_expert_results and 'analysis_results' in state:
                geology_expert_results = state['analysis_results'].get('geology')
            self.logger.debug(f'geology_expert_results present: {bool(geology_expert_results)}')
            self.logger.debug(f"preprocessed_data present: {state.get('preprocessed_data') is not None}")
            if state.get('preprocessed_data') is not None:
                self.logger.debug(f"preprocessed_data type: {type(state.get('preprocessed_data'))}")
            preprocessed_data = state.get('preprocessed_data')
            if preprocessed_data is None:
                preprocessed_data = state.get('processed_data')
            if geology_expert_results:
                self.logger.info('Detected geological-analysis results. Entering the prediction stage...')
                if preprocessed_data is None:
                    self.logger.info('Preprocessed data is missing. Running quick preprocessing again (skipping the analysis report)...')
                    ctx = self.build_skill_context(state=state, config=state.get('config'))
                    df_cleaned = self._run_skill_or_fallback(ctx, "data.clean", lambda df: self.clean_data(df, config=state.get('config')), df=df)
                    df_transformed = self._run_skill_or_fallback(
                        ctx, "data.transform", lambda df: self.transform_data(df, config=state.get('config')), df=df_cleaned
                    )
                    df_scaled = self._run_skill_or_fallback(ctx, "data.scale", lambda df: self.scale_data(df, config=state.get('config')), df=df_transformed)
                else:
                    self.logger.info('Reusing the existing preprocessed data...')
                    df_scaled = preprocessed_data
                df_scaled = self._maybe_run_auto_programming(state=state, df_scaled=df_scaled)
                state['preprocessed_data'] = df_scaled
                state['processed_data'] = df_scaled
                preprocessing_strategy = state.get('preprocessing_strategy', 'Existing strategy reused')
            else:
                self.logger.info('Starting data-feature analysis...')
                self.data_analyzer.clear_cache()
                exclude_cols = ['FID', 'Ore', 'Longitude', 'Latitude']
                geochemical_cols = [col for col in df.columns if col not in exclude_cols]
                df_geochemical = df[geochemical_cols].copy()
                self.logger.info(f'Excluded non-geochemical columns: {exclude_cols}')
                self.logger.info(f'Number of geochemical element columns: {len(geochemical_cols)}')
                quality_analysis = self.data_analyzer.analyze_data_quality(df_geochemical)
                self.logger.info(f"Data-quality analysis completed: {quality_analysis['shape']}")
                distribution_analysis = self.data_analyzer.analyze_distributions(df_geochemical)
                self.logger.info(f'Data-distribution analysis completed for {len(distribution_analysis)} features')
                preprocessing_recommendations = self.data_analyzer.recommend_preprocessing_strategy(df_geochemical)
                self.logger.info(f"Generated {len(preprocessing_recommendations['reasoning'])} preprocessing recommendations")
                data_report = self.data_analyzer.generate_data_report(df_geochemical, include_correlations=False)
                reports_dir = os.path.join(self.output_dir, 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                boxplot_path = os.path.join(reports_dir, "all_elements_boxplot.png")
                boxplot_out = self._plot_all_elements_boxplot(df=df_geochemical, out_path=boxplot_path, title="All elements boxplot")
                boxplot_md = ""
                if boxplot_out:
                    state["all_elements_boxplot_path"] = boxplot_out
                    boxplot_md = "## All-Element Boxplot\n\n![All-Element Boxplot](all_elements_boxplot.png)\n\n"
                original_data_report = data_report
                state["original_data_analysis_report_content"] = original_data_report
                merged_report_path = os.path.join(reports_dir, "data_analysis_report.md")
                merged_report_content = (
                    "# Data Analysis Report\n\n"
                    "## Raw-Data Analysis Report\n\n"
                    f"{original_data_report}\n\n"
                    f"{boxplot_md}"
                )
                _atomic_write_text(merged_report_path, merged_report_content)
                self.logger.info(f"Merged data-analysis report saved to: {merged_report_path}")
                task = f"Based on the following data-analysis results, decide the best preprocessing strategy:\n\nData overview:\n- Shape: {quality_analysis['shape']}\n- Numeric columns: {len(quality_analysis['numeric_columns'])}\n- Columns with missing values: {len(quality_analysis['missing_values'])}\n\nPreprocessing recommendations:\n{chr(10).join(preprocessing_recommendations['reasoning'][:5])}\n\nPlease choose:\n1. Use the recommended standard preprocessing pipeline\n2. Use a custom preprocessing pipeline\n3. Skip some preprocessing steps\n\nBriefly explain your choice and rationale (do not generate code)."
                preprocessing_strategy = self.decide(task, config=state.get('config'))
                self.logger.info(f'Preprocessing strategy decision: {preprocessing_strategy[:200]}...')
                strategy_path = os.path.join(reports_dir, 'preprocessing_strategy.md')
                _atomic_write_text(strategy_path, f'# Preprocessing Strategy\n\n{preprocessing_strategy}\n\n## Data-Analysis Results\n\n{data_report}')
                self.logger.info(f'Preprocessing strategy saved to: {strategy_path}')
                ctx = self.build_skill_context(state=state, config=state.get('config'))
                df_cleaned = self._run_skill_or_fallback(ctx, "data.clean", lambda df: self.clean_data(df, config=state.get('config')), df=df)
                df_transformed = self._run_skill_or_fallback(
                    ctx, "data.transform", lambda df: self.transform_data(df, config=state.get('config')), df=df_cleaned
                )
                df_scaled = self._run_skill_or_fallback(ctx, "data.scale", lambda df: self.scale_data(df, config=state.get('config')), df=df_transformed)
                df_scaled = self._maybe_run_auto_programming(state=state, df_scaled=df_scaled)
                step_overrides = self._get_data_science_step_overrides(state)
                do_post_feature_analysis = not bool(step_overrides.get("skip_post_preprocess_feature_analysis", False))
                parts_obj = step_overrides.get("post_feature_analysis_parts")
                parts = parts_obj if isinstance(parts_obj, dict) else {}
                run_corr = True if "correlation" not in parts else bool(parts.get("correlation"))
                run_hier = True if "hierarchical" not in parts else bool(parts.get("hierarchical"))
                run_factor = True if "factor" not in parts else bool(parts.get("factor"))
                write_report = True if "write_preprocessed_report" not in step_overrides else bool(step_overrides.get("write_preprocessed_report"))
                if _hitl_enabled(state) and _stdin_is_interactive():
                    if do_post_feature_analysis:
                        method_lines: List[str] = ["- Distribution analysis (overview of numeric-feature distributions)"]
                        if run_corr:
                            method_lines.append("- Correlation analysis (identify strongly correlated feature pairs)")
                        if run_hier:
                            method_lines.append("- Hierarchical clustering (group features by similarity)")
                        if run_factor:
                            method_lines.append("- Factor analysis (identify latent common-factor structure)")
                        if write_report:
                            method_lines.append("- Write the report file (data_analysis_report.md, including raw + post-preprocessing results)")
                        else:
                            method_lines.append("- Do not write the report file")
                        self.logger.info(
                            "[INPUT] HITL substep: post-preprocessing feature analysis\n"
                            "Goal: review the post-preprocessing feature structure to support downstream modeling and geological interpretation\n"
                            "Planned methods:\n"
                            + "\n".join(method_lines)
                            + "\n"
                            + f"Current default switches: correlation={run_corr}, hierarchical={run_hier}, factor={run_factor}, write_report={write_report}\n"
                            + "Interaction: Enter=confirm and run; skip=skip; exit=leave HITL; or type natural-language changes to this substep"
                        )
                        cmd = _input_with_log_prefix(self.logger).strip()
                        cmd_norm = cmd.strip().lower()
                        if cmd_norm in {"exit", "quit", "q"}:
                            _hitl_disable(state)
                        elif cmd_norm in {"skip"}:
                            do_post_feature_analysis = False
                            state.setdefault("processing_history", []).append(f"{self.agent_name}: skipped post-preprocessing feature analysis")
                        elif cmd:
                            default_payload = {
                                "intent_summary": "",
                                "assistant_reply": "",
                                "action": "continue",
                                "post_feature_analysis_parts": {"correlation": run_corr, "hierarchical": run_hier, "factor": run_factor},
                                "write_preprocessed_report": write_report,
                                "need_clarification": False,
                                "clarifying_question": "",
                            }
                            prompt = (
                                "You are the HITL substep interpreter for DataScienceExpertAgent."
                                " The current substep is post-preprocessing feature analysis. The user may ask questions or request modifications."
                                "\n\n"
                                "Strict constraints: you may only modify the method/content of the current substep. "
                                "You must not change the overall task objective, request new data sources, or modify the global task plan."
                                " You may only make executable changes within the following whitelist:\n"
                                "- post_feature_analysis_parts.correlation: whether to run correlation analysis\n"
                                "- post_feature_analysis_parts.hierarchical: whether to run hierarchical clustering\n"
                                "- post_feature_analysis_parts.factor: whether to run factor analysis\n"
                                "- write_preprocessed_report: whether to write the post-preprocessing data-analysis report\n"
                                "\n\n"
                                "You must output JSON only. Do not output any other text and do not use Markdown code fences.\n\n"
                                f"Current default settings: {default_payload}\n\n"
                                f"User input: {cmd}\n\n"
                                "Return JSON with the following fields:\n"
                                "- intent_summary: string\n"
                                "- assistant_reply: string\n"
                                "- action: string, must be continue | skip | exit_hitl | modify_then_continue | modify_then_skip\n"
                                "- post_feature_analysis_parts: object, containing booleans for correlation/hierarchical/factor\n"
                                "- write_preprocessed_report: boolean\n"
                                "- need_clarification: boolean\n"
                                "- clarifying_question: string\n"
                            )
                            parsed = self.decide_json(prompt, default_payload, config=state.get("config"))
                            if isinstance(parsed, dict):
                                reply = str(parsed.get("assistant_reply") or "").strip()
                                if reply:
                                    self.logger.info(f"HITL reply: {reply}")
                                action = str(parsed.get("action") or "").strip()
                                if action == "exit_hitl":
                                    _hitl_disable(state)
                                if action in {"skip", "modify_then_skip"}:
                                    do_post_feature_analysis = False
                                    state.setdefault("processing_history", []).append(f"{self.agent_name}: skipped post-preprocessing feature analysis")
                                p = parsed.get("post_feature_analysis_parts")
                                if isinstance(p, dict):
                                    if "correlation" in p:
                                        run_corr = bool(p.get("correlation"))
                                    if "hierarchical" in p:
                                        run_hier = bool(p.get("hierarchical"))
                                    if "factor" in p:
                                        run_factor = bool(p.get("factor"))
                                if "write_preprocessed_report" in parsed:
                                    write_report = bool(parsed.get("write_preprocessed_report"))
                if do_post_feature_analysis:
                    self.logger.info('Reanalyzing post-preprocessing data features...')
                    self.data_analyzer.clear_cache()
                    exclude_cols = ['FID', 'Ore', 'Longitude', 'Latitude', 'id', 'label', 'target']
                    feature_cols_final = [c for c in df_scaled.columns if c not in exclude_cols and df_scaled[c].dtype in ['float64', 'int64']]
                    df_final_features = df_scaled[feature_cols_final]
                    final_distributions = self.data_analyzer.analyze_distributions(df_final_features)
                    final_correlations = self.data_analyzer.analyze_correlations(df_final_features) if run_corr else {}
                    feature_analysis_results = self._analyze_geochemical_features_for_geology(
                        df_scaled,
                        feature_cols_final,
                        run_correlation=run_corr,
                        run_hierarchical=run_hier,
                        run_factor=run_factor,
                    )
                    feature_analysis_results['distributions'] = final_distributions
                    feature_analysis_results['data_analyzer_correlations'] = final_correlations
                    state['feature_analysis_results'] = feature_analysis_results
                    if not state.get('feature_cols'):
                        state['feature_cols'] = feature_cols_final
                    self.logger.info(f'Post-preprocessing feature analysis completed. Updated analysis results for {len(feature_cols_final)} features.')
                    if write_report:
                        preprocessed_data_report = self.data_analyzer.generate_data_report(df_final_features)
                        merged_report_path = os.path.join(reports_dir, "data_analysis_report.md")
                        boxplot_md = ""
                        try:
                            boxplot_md = "## All-Element Boxplot\n\n![All-Element Boxplot](all_elements_boxplot.png)\n\n" if state.get("all_elements_boxplot_path") else ""
                        except Exception:
                            boxplot_md = ""
                        merged_report_content = (
                            "# Data Analysis Report\n\n"
                            "## Raw-Data Analysis Report\n\n"
                            f"{original_data_report}\n\n"
                            f"{boxplot_md}"
                            "## Post-Preprocessing Data Analysis Report\n\n"
                            f"{preprocessed_data_report}\n"
                        )
                        _atomic_write_text(merged_report_path, merged_report_content)
                        self.logger.info(f"Merged data-analysis report saved to: {merged_report_path}")
            geology_expert_results = state.get('geology_expert_results')
            if not geology_expert_results and 'analysis_results' in state:
                geology_expert_results = state['analysis_results'].get('geology')
            if geology_expert_results:
                self.logger.info('Detected geological-analysis results. Starting predictive-model construction...')
                feature_analysis_results = state.get('feature_analysis_results', {})
                feature_cols = state.get('feature_cols', [])
                if not feature_cols:
                    numeric_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    exclude_cols = ['FID', 'Ore', 'Longitude', 'Latitude', 'id', 'label', 'target']
                    feature_cols = [c for c in numeric_cols if c not in exclude_cols and (not any((ex in c.lower() for ex in ['id', 'code'])))]
                    state['feature_cols'] = feature_cols
                target_elements = []
                if isinstance(geology_expert_results, dict):
                    target_elements = geology_expert_results.get('target_related_elements') or []
                    if not target_elements:
                        target_elements = geology_expert_results.get('target_element_selection', {}).get('selected_elements', [])
                if target_elements:
                    filtered_feature_cols = self._filter_feature_cols_by_elements(feature_cols, target_elements)
                    if filtered_feature_cols and len(filtered_feature_cols) < len(feature_cols):
                        self.logger.info(f'Filtered features according to geology-expert elements: {len(feature_cols)} -> {len(filtered_feature_cols)}')
                    if filtered_feature_cols:
                        feature_cols = filtered_feature_cols
                        state['feature_cols'] = feature_cols
                forced_model_key = None
                try:
                    container = state.get('hitl_step_overrides')
                    if isinstance(container, dict):
                        step_overrides_obj = container.get('data_science_expert')
                        if isinstance(step_overrides_obj, dict):
                            forced_model_key = step_overrides_obj.get('model_key')
                except Exception:
                    forced_model_key = None
                prediction_results = self.predict(
                    df_scaled,
                    feature_cols,
                    geology_expert_results,
                    feature_analysis_results,
                    som_reference_data=df,
                    forced_model_key=forced_model_key,
                    config=state.get('config'),
                )
                state['prediction_results'] = prediction_results
                state['processing_history'].append(f'{self.agent_name}: predictive model completed')
                state['next_agent'] = 'result_output'
            else:
                history = state.get('processing_history', [])
                last_history = history[-1] if history else ''
                if 'GeologyExpertAgent' in last_history or 'geological analysis completed' in last_history:
                    self.logger.info('Detected a recent geological-analysis completion without valid outputs. Handing control to the decision center...')
                    state['next_agent'] = 'agent_decision'
                else:
                    self.logger.info('No geological-analysis results were detected. Handing off to the geology expert...')
                    state['processing_history'].append(f'{self.agent_name}: data preprocessing completed (waiting for geological analysis)')
                    state['next_agent'] = 'geology_analysis'
            if isinstance(state.get('prediction_results'), dict) and state.get('next_agent') != 'result_output':
                state['next_agent'] = 'result_output'
            state['preprocessed_data'] = df_scaled
            state['processed_data'] = df_scaled
            state['preprocessing_strategy'] = preprocessing_strategy
            self.logger.info(f'{self.agent_name} completed successfully')
        except Exception as e:
            self.logger.exception(f'{self.agent_name} failed: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'{self.agent_name}: {str(e)}')
            state['processing_history'].append(f'{self.agent_name}: execution failed - {str(e)}')
            state['next_agent'] = 'agent_decision'
        return state

    def _analyze_geochemical_features_for_geology(
        self,
        data: pd.DataFrame,
        element_cols: List[str],
        *,
        run_correlation: bool = True,
        run_hierarchical: bool = True,
        run_factor: bool = True,
    ) -> Dict[str, Any]:
        feature_output_dir = os.path.join(self.output_dir, 'feature_analysis')
        os.makedirs(feature_output_dir, exist_ok=True)
        feature_data = data[element_cols].dropna()
        scaled_df = None
        if run_factor:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(feature_data)
            scaled_df = pd.DataFrame(scaled_data, columns=element_cols, index=feature_data.index)
        results: Dict[str, Any] = {
            'stage': 'data_science',
            'element_count': int(len(element_cols)),
            'sample_count': int(len(feature_data)),
            'correlation_analysis': {},
            'factor_analysis': {},
            'hierarchical_clustering': {},
            'summary': '',
        }
        if run_correlation:
            correlation_output_dir = os.path.join(feature_output_dir, 'correlation_analysis')
            results['correlation_analysis'] = self._analyze_correlations_for_geology(feature_data, element_cols, correlation_output_dir)
        if run_hierarchical:
            hierarchical_output_dir = os.path.join(feature_output_dir, 'hierarchical_clustering')
            os.makedirs(hierarchical_output_dir, exist_ok=True)
            hc = self._perform_hierarchical_clustering_for_geology(feature_data, element_cols, hierarchical_output_dir)
            results['hierarchical_clustering'] = hc
            try:
                hc_path = os.path.join(hierarchical_output_dir, 'hierarchical_clustering_results.json')
                _atomic_write_text(hc_path, json.dumps(hc, ensure_ascii=False, indent=2, default=str))
            except Exception as e:
                self.logger.warning(f'hierarchical clustering results json save failed: {e}')
        if run_factor and scaled_df is not None:
            factor_output_dir = os.path.join(feature_output_dir, 'factor_analysis')
            os.makedirs(factor_output_dir, exist_ok=True)
            results['factor_analysis'] = self._perform_factor_analysis_for_geology(scaled_df, element_cols, factor_output_dir)
        results['summary'] = self._generate_feature_summary_for_geology(results)
        return results

    def _analyze_correlations_for_geology(self, data: pd.DataFrame, element_cols: List[str], output_dir: str) -> Dict[str, Any]:
        lang = _resolve_output_language()
        _setup_matplotlib_output_style(plt)
        results: Dict[str, Any] = {}
        os.makedirs(output_dir, exist_ok=True)
        for method in ['pearson', 'spearman', 'kendall']:
            try:
                corr_matrix = data.corr(method=method)
                corr_matrix_path = os.path.join(output_dir, f'correlation_matrix_{method}.csv')
                _atomic_write_csv(corr_matrix, corr_matrix_path, encoding='utf-8-sig')
                high_correlations: List[Dict[str, Any]] = []
                for i in range(len(element_cols)):
                    for j in range(i + 1, len(element_cols)):
                        elem1 = element_cols[i]
                        elem2 = element_cols[j]
                        corr_value = float(corr_matrix.iloc[i, j])
                        if abs(corr_value) > 0.7:
                            high_correlations.append(
                                {
                                    'elements': f'{elem1}-{elem2}',
                                    'correlation': corr_value,
                                    'strength': 'strong' if abs(corr_value) > 0.85 else 'moderate',
                                }
                            )
                high_corr_df = pd.DataFrame(high_correlations)
                high_corr_path = os.path.join(output_dir, f'high_correlations_{method}.csv')
                _atomic_write_csv(high_corr_df, high_corr_path, index=False, encoding='utf-8-sig')
                results[method] = {
                    'matrix': corr_matrix.to_dict(),
                    'matrix_path': corr_matrix_path,
                    'high_correlations': high_correlations,
                    'high_correlations_path': high_corr_path,
                }
            except Exception as e:
                self.logger.warning(f'{method} correlation failed: {e}')
        try:
            corr_for_plot = data[element_cols].corr(method='pearson') if element_cols else data.corr(method='pearson')
            n_elements = int(corr_for_plot.shape[0])
            figsize_w = min(32, max(12, 0.35 * n_elements + 6))
            figsize_h = min(28, max(10, 0.32 * n_elements + 5))
            annot = n_elements <= 20
            font_mul = 2.0
            title_fontsize = int(14 * font_mul)
            tick_fontsize = 24
            cbar_tick_fontsize = 24
            cbar_label_fontsize = 24
            annot_fontsize = 24
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            with sns.axes_style('white'):
                with sns.plotting_context('talk', font_scale=1.1):
                    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                    fig.patch.set_facecolor('white')
                    ax = sns.heatmap(
                        corr_for_plot,
                        annot=annot,
                        annot_kws={"size": annot_fontsize} if annot else None,
                        cmap=cmap,
                        fmt='.2f',
                        linewidths=0.5,
                        linecolor='white',
                        vmin=-1,
                        vmax=1,
                        center=0,
                        square=n_elements <= 25,
                        cbar_kws={'shrink': 0.82, 'pad': 0.02, 'aspect': 30},
                    )
                    ax.set_title(_localize_text('Element Correlation Heatmap', lang=lang), fontsize=title_fontsize, pad=18)
                    rotate = 45 if n_elements <= 22 else 90
                    ax.tick_params(axis='x', labelrotation=rotate, labelsize=tick_fontsize)
                    ax.tick_params(axis='y', labelrotation=0, labelsize=tick_fontsize)
                    for label in ax.get_xticklabels():
                        label.set_ha('right')
                        label.set_rotation_mode('anchor')
                        label.set_fontsize(tick_fontsize)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(tick_fontsize)
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
                    cbar.set_label(_get_bilingual_text('Correlation Coefficient', 'Correlation Coefficient', lang=lang), fontsize=cbar_label_fontsize)
                    plt.tight_layout(pad=2.0)
                    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
                    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white', format='png')
                    plt.close(fig)
            results['heatmap_path'] = heatmap_path
        except Exception as e:
            self.logger.warning(f'correlation heatmap failed: {e}')
        return results

    def _perform_factor_analysis_for_geology(self, scaled_data: pd.DataFrame, element_cols: List[str], output_dir: str) -> Dict[str, Any]:
        lang = _resolve_output_language()
        _setup_matplotlib_output_style(plt)
        n_features = int(len(element_cols)) if element_cols else 0
        if n_features <= 0:
            return {'factor_count': 0, 'factor_loadings': {}}
        os.makedirs(output_dir, exist_ok=True)
        try:
            X = scaled_data[element_cols].to_numpy(dtype=float, copy=False) if isinstance(scaled_data, pd.DataFrame) else np.asarray(scaled_data, dtype=float)
        except Exception:
            X = np.asarray(scaled_data, dtype=float)
        n_samples = int(X.shape[0]) if X is not None and hasattr(X, "shape") and len(getattr(X, "shape", ())) >= 2 else 0
        max_components = int(min(n_samples, n_features)) if n_samples > 0 else int(n_features)
        if max_components <= 0:
            return {'factor_count': 0, 'factor_loadings': {}}
        max_factors = int(min(8, max_components))
        full_pca = PCA(n_components=max_components)
        full_pca.fit(X)
        eigenvalues = np.asarray(getattr(full_pca, "explained_variance_", []), dtype=float)
        evr = np.asarray(getattr(full_pca, "explained_variance_ratio_", []), dtype=float)
        eigenvalues = eigenvalues[:max_components]
        evr = evr[:max_components]
        k_kaiser = 0
        elbow_k = 0
        try:
            v = eigenvalues[:max_factors].copy()
            if v.size >= 3:
                first = v[:-1] - v[1:]
                second = first[:-1] - first[1:]
                elbow_k = int(np.argmax(second) + 2) if second.size else 0
        except Exception:
            elbow_k = 0
        if 1 <= elbow_k <= max_factors:
            n_factors = int(elbow_k)
            selection_method = "scree_elbow"
        else:
            n_factors = int(min(4, max_factors))
            selection_method = "fixed_cap"
        if n_factors <= 0:
            return {'factor_count': 0, 'factor_loadings': {}}
        if n_factors > max_components:
            n_factors = int(max_components)
        results: Dict[str, Any] = {
            'factor_count': int(n_factors),
            'factor_selection_method': str(selection_method),
            'kaiser_factor_count': int(k_kaiser),
            'scree_elbow_factor_count': int(elbow_k),
            'eigenvalues': [float(x) for x in eigenvalues[:max_factors].tolist()] if eigenvalues.size else [],
            'explained_variance_ratio': [float(x) for x in evr[:max_factors].tolist()] if evr.size else [],
        }
        try:
            xs = np.arange(1, int(min(max_factors, eigenvalues.size)) + 1, dtype=int)
            ys = eigenvalues[: len(xs)]
            title_fontsize = 18
            label_fontsize = 16
            tick_fontsize = 16
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('white')
            ax.plot(xs, ys, marker='o', linewidth=2)
            ax.set_xlabel('Factor Index', fontsize=label_fontsize)
            ax.set_ylabel(_localize_text('Eigenvalue', lang=lang), fontsize=label_fontsize)
            ax.set_title(_localize_text('Scree Plot', lang=lang), fontsize=title_fontsize)
            ax.tick_params(axis='both', labelsize=tick_fontsize)
            ax.grid(True, linestyle='--', alpha=0.35)
            if 1 <= n_factors <= len(xs):
                ax.axvline(x=n_factors, color='red', linestyle='--', linewidth=1.6)
            plt.tight_layout()
            scree_path = os.path.join(output_dir, 'scree_plot.png')
            fig.savefig(scree_path, dpi=200, bbox_inches='tight', facecolor='white', format='png')
            plt.close(fig)
            results['scree_plot_path'] = scree_path
        except Exception as e:
            self.logger.warning(f'scree plot failed: {e}')
        pca = PCA(n_components=n_factors)
        pca.fit(X)
        factor_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loading_df = pd.DataFrame(factor_loadings, columns=[f'Factor{i + 1}' for i in range(n_factors)], index=element_cols)
        results['factor_loadings'] = loading_df.to_dict()
        try:
            scores = pca.transform(X)
            if isinstance(scaled_data, pd.DataFrame):
                sample_index = scaled_data.index
            else:
                sample_index = pd.RangeIndex(start=0, stop=int(scores.shape[0]))
            score_df = pd.DataFrame(scores, columns=[f'Factor{i + 1}' for i in range(n_factors)], index=sample_index)
            score_df.insert(0, 'sample_index', sample_index)
            factor_scores_path = os.path.join(output_dir, 'factor_scores.csv')
            _atomic_write_csv(score_df, factor_scores_path, index=False, encoding='utf-8-sig')
            results['factor_scores_path'] = factor_scores_path
        except Exception as e:
            self.logger.warning(f'factor scores export failed: {e}')
        try:
            n_rows = int(loading_df.shape[0])
            n_cols = int(loading_df.shape[1])
            fig_w = min(30, max(12, 0.42 * n_cols + 10))
            fig_h = min(28, max(10, 0.28 * n_rows + 6))
            title_fontsize = 28
            tick_fontsize = 24 if n_rows <= 35 else 18
            cbar_tick_fontsize = 22
            cbar_label_fontsize = 24
            annot = True
            annot_fontsize = int(max(18, min(28, 480 / max(1, n_rows, n_cols))))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor('white')
            ax = sns.heatmap(
                loading_df,
                annot=annot,
                annot_kws={"size": annot_fontsize},
                cmap='RdBu_r',
                fmt='.2f',
                linewidths=0.5,
                center=0,
                cbar_kws={'shrink': 0.82, 'pad': 0.02, 'aspect': 30},
            )
            ax.set_title(_localize_text('Factor-Loading Heatmap', lang=lang), fontsize=title_fontsize, pad=18)
            ax.tick_params(axis='x', labelrotation=0, labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelrotation=0, labelsize=tick_fontsize)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
            cbar.set_label(_get_bilingual_text('Factor Loading', 'Factor Loading', lang=lang), fontsize=cbar_label_fontsize)
            plt.tight_layout(pad=2.0)
            factor_loading_path = os.path.join(output_dir, 'factor_loadings_heatmap.png')
            fig.savefig(factor_loading_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            plt.close(fig)
            results['factor_loading_plot_path'] = factor_loading_path
        except Exception as e:
            self.logger.warning(f'factor loading heatmap failed: {e}')
        return results

    def _perform_hierarchical_clustering_for_geology(self, data: pd.DataFrame, element_cols: List[str], output_dir: str) -> Dict[str, Any]:
        lang = _resolve_output_language()
        _setup_matplotlib_output_style(plt)
        results: Dict[str, Any] = {
            'method': 'average',
            'distance_metric': '1-pearson',
            'cluster_count': 0,
            'cluster_labels': {},
            'clusters': {},
        }
        os.makedirs(output_dir, exist_ok=True)
        if not element_cols or len(element_cols) < 2:
            return results
        try:
            from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
            from scipy.spatial.distance import squareform
        except Exception as e:
            self.logger.warning(f'hierarchical clustering skipped (scipy missing): {e}')
            return results

        try:
            corr = data[element_cols].corr(method='pearson').fillna(0.0).clip(-1.0, 1.0)
            dist = (1.0 - corr).astype(float)
            np.fill_diagonal(dist.values, 0.0)
            condensed = squareform(dist.values, checks=False)
            Z = linkage(condensed, method=str(results['method']))
            order_idx = leaves_list(Z).tolist()
            ordered_cols = [str(corr.index[int(i)]) for i in order_idx]

            n = int(len(element_cols))
            cluster_count = int(min(10, max(3, round(float(np.sqrt(n))))))
            labels = fcluster(Z, t=cluster_count, criterion='maxclust')
            cluster_labels: Dict[str, int] = {str(corr.index[i]): int(labels[i]) for i in range(n)}
            clusters: Dict[str, List[str]] = {}
            for name, cid in cluster_labels.items():
                clusters.setdefault(str(cid), []).append(str(name))

            results['cluster_count'] = cluster_count
            results['cluster_labels'] = cluster_labels
            results['clusters'] = clusters
            results['ordered_elements'] = ordered_cols
        except Exception as e:
            self.logger.warning(f'hierarchical clustering failed: {e}')
            return results

        try:
            n_elements = int(len(element_cols))
            fig_w = min(34, max(12, 0.28 * n_elements + 10))
            fig_h = 11
            leaf_font_size = (12 if n_elements <= 35 else (11 if n_elements <= 60 else 10)) * 2
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.patch.set_facecolor('white')
            dendrogram(
                Z,
                labels=[str(x) for x in corr.index.tolist()],
                leaf_rotation=90 if n_elements > 18 else 45,
                leaf_font_size=leaf_font_size,
                ax=ax,
            )
            is_zh = str(lang).lower().startswith("zh")
            if is_zh and chinese_font is not None:
                ax.set_title("Hierarchical Clustering Dendrogram", fontsize=28, pad=14, fontproperties=chinese_font)
                ax.set_ylabel("Distance", fontsize=28, fontproperties=chinese_font)
            elif is_zh:
                ax.set_title("Hierarchical Clustering Dendrogram", fontsize=28, pad=14)
                ax.set_ylabel("Distance", fontsize=28)
            else:
                ax.set_title("Hierarchical Clustering Dendrogram", fontsize=28, pad=14)
                ax.set_ylabel("Distance", fontsize=28)
            ax.tick_params(axis="y", labelsize=28)
            for label in ax.get_xticklabels():
                label.set_fontsize(leaf_font_size)
            for label in ax.get_yticklabels():
                label.set_fontsize(26)
            plt.tight_layout(pad=2.0)
            dendro_path = os.path.join(output_dir, 'hierarchical_clustering_dendrogram.png')
            fig.savefig(dendro_path, dpi=150, bbox_inches='tight', facecolor='white', format='png')
            plt.close(fig)
            results['dendrogram_path'] = dendro_path
        except Exception as e:
            self.logger.warning(f'hierarchical dendrogram plot failed: {e}')

        try:
            corr_ord = corr.loc[ordered_cols, ordered_cols]
            n_elements = int(corr_ord.shape[0])
            figsize_w = min(32, max(12, 0.35 * n_elements + 6))
            figsize_h = min(28, max(10, 0.32 * n_elements + 5))
            annot = n_elements <= 20
            tick_size = (12 if n_elements <= 22 else (11 if n_elements <= 35 else 10)) * 2
            cbar_tick_size = (11 if n_elements <= 35 else 10) * 2
            annot_size = (11 if n_elements <= 16 else 10) * 2
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            with sns.axes_style('white'):
                with sns.plotting_context('talk', font_scale=1.15):
                    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                    fig.patch.set_facecolor('white')
                    ax = sns.heatmap(
                        corr_ord,
                        annot=annot,
                        annot_kws={"size": annot_size} if annot else None,
                        cmap=cmap,
                        fmt='.2f',
                        linewidths=0.5,
                        linecolor='white',
                        vmin=-1,
                        vmax=1,
                        center=0,
                        square=n_elements <= 25,
                        cbar_kws={'shrink': 0.82, 'pad': 0.02, 'aspect': 30},
                    )
                    ax.set_title(_localize_text('Hierarchically Reordered Correlation Heatmap', lang=lang), fontsize=28, pad=18)
                    rotate = 45 if n_elements <= 22 else 90
                    ax.tick_params(axis='x', labelrotation=rotate, labelsize=tick_size)
                    ax.tick_params(axis='y', labelrotation=0, labelsize=tick_size)
                    for label in ax.get_xticklabels():
                        label.set_ha('right')
                        label.set_rotation_mode('anchor')
                        label.set_fontsize(tick_size)
                    for label in ax.get_yticklabels():
                        label.set_fontsize(tick_size)
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=cbar_tick_size)
                    cbar.set_label(_get_bilingual_text('Correlation Coefficient', 'Correlation Coefficient', lang=lang), fontsize=24)
                    plt.tight_layout(pad=2.0)
                    heatmap_path = os.path.join(output_dir, 'hierarchical_clustering_correlation_heatmap.png')
                    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight', facecolor='white', format='png')
                    plt.close(fig)
            results['reordered_correlation_heatmap_path'] = heatmap_path
        except Exception as e:
            self.logger.warning(f'hierarchical clustering heatmap failed: {e}')
        return results

    def _generate_feature_summary_for_geology(self, results: Dict[str, Any]) -> str:
        stage = str(results.get('stage', 'unknown'))
        summary = f"Feature-analysis results ({stage} stage):\n"
        summary += f"- Number of analyzed elements: {int(results.get('element_count', 0))}\n"
        summary += f"- Number of valid samples: {int(results.get('sample_count', 0))}\n\n"
        pearson_corr = results.get('correlation_analysis', {}).get('pearson', {})
        high_correlations = pearson_corr.get('high_correlations', [])
        summary += f'- Number of strong correlation pairs (>0.7): {len(high_correlations) if isinstance(high_correlations, list) else 0}\n'
        factor_loadings = (results.get('factor_analysis') or {}).get('factor_loadings', {})
        factor_count = (results.get('factor_analysis') or {}).get('factor_count', 0)
        if factor_loadings:
            try:
                method = (results.get('factor_analysis') or {}).get('factor_selection_method', '')
                method_text = f' (factor-count selection: {method})' if method else ''
                summary += f'- Factor analysis identified {int(factor_count)} major factors{method_text}, reflecting different geological processes\n'
            except Exception:
                summary += '- Factor analysis identified major factors that reflect different geological processes\n'
        hc = results.get('hierarchical_clustering') or {}
        if isinstance(hc, dict):
            try:
                cc = int(hc.get('cluster_count', 0) or 0)
            except Exception:
                cc = 0
            if cc > 0:
                summary += f'- Number of hierarchical clusters (maxclust): {cc}\n'
        return summary
    def clean_data(self, df: pd.DataFrame, config: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        fn = self._get_skill_tool_callable("data.clean", "tool.py", "clean_data")
        if fn is None:
            raise RuntimeError("data.clean tool not found: skills/data-clean/tool.py:clean_data")
        return fn(
            df,
            decide_json=self.decide_json,
            data_analyzer=self.data_analyzer,
            logger=self.logger,
            config=config,
        )
    def transform_data(self, df: pd.DataFrame, config: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        fn = self._get_skill_tool_callable("data.transform", "tool.py", "transform_data")
        if fn is None:
            raise RuntimeError("data.transform tool not found: skills/data-transform/tool.py:transform_data")
        return fn(
            df,
            decide_json=self.decide_json,
            data_analyzer=self.data_analyzer,
            logger=self.logger,
            config=config,
        )
    def scale_data(self, df: pd.DataFrame, config: Optional[Dict[str, Any]]=None) -> pd.DataFrame:
        fn = self._get_skill_tool_callable("data.scale", "tool.py", "scale_data")
        if fn is None:
            raise RuntimeError("data.scale tool not found: skills/data-scale/tool.py:scale_data")
        return fn(
            df,
            decide=self.decide,
            scaler=self.scaler,
            minmax_scaler=self.minmax_scaler,
            logger=self.logger,
            config=config,
        )
    def clr_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Applying CLR transformation')
        df_clr = df.copy()
        numeric_cols = df_clr.select_dtypes(include=['int64', 'float64']).columns
        df_clr[numeric_cols] = df_clr[numeric_cols].replace(0, 1e-09)
        geom_mean = df_clr[numeric_cols].apply(lambda x: np.exp(np.mean(np.log(x))), axis=1)
        for col in numeric_cols:
            df_clr[col] = np.log(df_clr[col] / geom_mean)
        self.logger.info(f'CLR transformed data shape: {df_clr.shape}')
        return df_clr

    def _expand_positive_samples_if_needed(
        self,
        *,
        data: pd.DataFrame,
        y: pd.Series,
        positive_label: int = 1,
        negative_label: int = 0,
        ratio_threshold: float = 0.01,
        k_neighbors: int = 8,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        y_series = pd.Series(y, index=data.index).copy()
        n_total = int(len(y_series))
        pos_idx = y_series[y_series == int(positive_label)].index
        n_pos = int(len(pos_idx))
        meta: Dict[str, Any] = {
            "enabled": True,
            "triggered": False,
            "ratio_threshold": float(ratio_threshold),
            "k_neighbors": int(k_neighbors),
            "candidate_label": int(negative_label),
            "total_samples": n_total,
            "original_positive_count": n_pos,
        }
        if n_total <= 0 or n_pos <= 0:
            meta["reason"] = "no_positive_or_empty"
            return (y_series, meta)
        if float(n_pos) > float(ratio_threshold) * float(n_total):
            meta["reason"] = "ratio_above_threshold"
            return (y_series, meta)

        x_col, y_col = _detect_coordinate_columns(data)
        if not x_col or not y_col or x_col not in data.columns or y_col not in data.columns:
            meta["reason"] = "coordinate_columns_not_found"
            meta["coord_cols"] = {"x": x_col, "y": y_col}
            return (y_series, meta)

        coords_x = pd.to_numeric(data[x_col], errors="coerce").to_numpy(dtype=float)
        coords_y = pd.to_numeric(data[y_col], errors="coerce").to_numpy(dtype=float)
        coords = np.stack([coords_x, coords_y], axis=1)
        has_coord = np.isfinite(coords).all(axis=1)
        idx_all = np.asarray(data.index)

        cand_mask = (np.asarray(y_series) == int(negative_label)) & has_coord
        cand_idx = idx_all[cand_mask]
        cand_coords = coords[cand_mask]

        pos_mask = (np.asarray(y_series) == int(positive_label)) & has_coord
        pos_with_coord_idx = idx_all[pos_mask]
        pos_with_coord_coords = coords[pos_mask]

        meta["coord_cols"] = {"x": str(x_col), "y": str(y_col)}
        meta["positive_with_coords"] = int(len(pos_with_coord_idx))
        meta["unlabeled_with_coords"] = int(len(cand_idx))
        meta["negative_with_coords"] = int(len(cand_idx))
        meta["positive_without_coords"] = int(n_pos - int(len(pos_with_coord_idx)))

        if int(len(cand_idx)) == 0 or int(len(pos_with_coord_idx)) == 0:
            meta["reason"] = "no_candidates_or_positive_coords"
            return (y_series, meta)

        expanded: set = set(pos_idx.tolist())
        k_eff = max(1, int(k_neighbors))
        for p in pos_with_coord_coords:
            diff = cand_coords - p.reshape(1, 2)
            dist2 = np.einsum("ij,ij->i", diff, diff)
            if dist2.size == 0:
                continue
            k_take = min(k_eff, int(dist2.size))
            nn_pos = np.argpartition(dist2, k_take - 1)[:k_take]
            for j in nn_pos.tolist():
                expanded.add(cand_idx[int(j)])

        y_series.loc[pd.Index(list(expanded))] = int(positive_label)
        meta["triggered"] = True
        meta["expanded_positive_count"] = int((y_series == int(positive_label)).sum())
        meta["added_positive_count"] = int(meta["expanded_positive_count"] - n_pos)
        return (y_series, meta)
    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: list,
        geology_expert_results: Dict,
        feature_analysis_results: Dict,
        previous_predictions: Optional[Dict] = None,
        *,
        som_reference_data: Optional[pd.DataFrame] = None,
        forced_model_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.logger.info('Starting predictive modeling...')
        learning_mode = ""
        if isinstance(config, dict):
            raw_mode = config.get("learning_mode")
            s = str(raw_mode or "").strip().lower()
            if s in {"supervised", "1"}:
                learning_mode = "supervised"
            elif s in {"unsupervised", "2"}:
                learning_mode = "unsupervised"
            elif s in {"self_supervised", "self-supervised", "3"}:
                learning_mode = "self_supervised"
            if not learning_mode:
                learning_mode = "supervised"
                config["learning_mode"] = learning_mode
        else:
            learning_mode = "supervised"
        exclude_cols = {'FID', 'Ore', 'Longitude', 'Latitude', 'id', 'label', 'target'}
        feature_cols = [c for c in feature_cols if c in data.columns and c not in exclude_cols and (not any((ex in str(c).lower() for ex in ['id', 'code'])))]
        target_variable, label_meta = self._define_labels_by_learning_mode(data=data, geology_expert_results=geology_expert_results, learning_mode=learning_mode, config=config)
        try:
            y_arr = np.asarray(target_variable)
            valid_mask = y_arr != -1
            y_valid = y_arr[valid_mask]
            pos_count = int(np.sum(y_valid == 1))
            neg_count = int(np.sum(y_valid == 0))
            excluded_count = int(np.sum(~valid_mask))
        except Exception:
            pos_count = None
            neg_count = None
            excluded_count = None
        graph_topology_ready = False
        graph_topology_info: Dict[str, Any] = {}
        positive_spatial_profile: Dict[str, Any] = {}
        try:
            x_col, y_col = _detect_coordinate_columns(data)
            has_coords = bool(x_col and y_col and x_col in data.columns and y_col in data.columns)
            if has_coords:
                coords = np.asarray(data[[x_col, y_col]], dtype=np.float64)
                ok = np.isfinite(coords).all(axis=1)
                graph_topology_info = {"coord_cols": [str(x_col), str(y_col)], "has_coords": bool(np.any(ok))}
            positive_spatial_profile = _summarize_positive_spatial_profile(data_df=data, y=target_variable, radius=None, k_neighbor=6)
        except Exception:
            graph_topology_ready = False
            graph_topology_info = {}
            positive_spatial_profile = {}
        mode_to_models = {"supervised": ["som"], "self_supervised": ["som"], "unsupervised": ["som"]}
        allowed_models = mode_to_models.get(learning_mode, ["som"])
        all_models = {"som": {"tuning": "Optional: grid_m/grid_n/n_iter/sigma/lr (automatic tuning is currently disabled by default)", "metric": "roc_auc"}}
        available_models = {k: v for k, v in all_models.items() if k in allowed_models}
        sampling_plan: Dict[str, Any] = {
            "train_test_split": {"test_size": 0.3, "stratify": True, "random_state": 42},
            "train_downsampling": {"enabled": False},
            "note": "SOM(QE) is trained with the full training split. If both positive and negative labels (0/1) are available, QE->LogisticRegression is used for probability calibration.",
        }
        if learning_mode == "self_supervised":
            sampling_plan = {
                "train_test_split": {"test_size": 0.3, "stratify": True, "random_state": 42},
                "pseudo_labeling": {"enabled": False},
                "note": "Only SOM(QE) scoring is used in the current configuration; pseudo-label iteration is disabled.",
            }
        try:
            if pos_count is not None and neg_count is not None:
                train_pos_est = int(round(float(pos_count) * 0.7))
                train_neg_est = int(round(float(neg_count) * 0.7))
                if bool((sampling_plan.get("train_downsampling") or {}).get("enabled")):
                    used_neg = int(round(float(train_pos_est) * float(sampling_plan["train_downsampling"]["negative_to_positive_ratio"])))
                    used_neg = min(used_neg, train_neg_est)
                    sampling_plan["estimated_counts"] = {
                        "raw_valid": {"positive": int(pos_count), "negative": int(neg_count), "excluded": int(excluded_count or 0)},
                        "train_estimated_before_downsampling": {"positive": train_pos_est, "negative": train_neg_est},
                        "train_estimated_after_downsampling": {"positive": train_pos_est, "negative": used_neg},
                    }
        except Exception:
            pass
        display_names = {"som": "SOM(QE)"}
        list_lines = [f"- {display_names.get(k, k)}" for k in allowed_models if k in display_names]
        task = (
            "You are a mineralization-potential predictive-modeling expert. The current system retains only the SOM(QE) model, so focus on whether it is appropriate for this dataset and whether tuning is necessary.\n"
            "Important: the data have already been preprocessed and standardized before this step. Do not discuss or recommend any preprocessing strategy.\n\n"
            "Sample-and-label note: the positive/negative label definition has already been determined by the selected learning mode (positive=1, negative=0, excluded=-1). Refer to the contextual field labeling_rule for details. Do not propose or design a new labeling rule.\n\n"
            "Current learning mode: "
            + str(learning_mode)
            + "\n\n"
            "Current model (the only available option):\n"
            + "\n".join(list_lines)
            + "\n\n"
            "Current evaluation protocol:\n"
            "- Primary metric: ROC-AUC (evaluate it when Ore(0/1) labels exist; otherwise output QE scores for relative ranking only)\n"
            "- Validation scheme: train/holdout = 70/30 with stratification when feasible\n\n"
            "Required output format (must be followed strictly, and every field must be filled):\n"
            "1) First line: model_key: <som>\n"
            "2) Second line: metric: roc_auc\n"
            "3) Third line: tuning: <whether tuning is needed and why>\n"
            "4) Fourth line: suitability: <why SOM is suitable for the current dataset>\n"
            "5) Next line: reasoning: <2-5 bullet-style points explaining why this model is selected, explicitly referencing data characteristics or class distribution>\n"
            "6) Next line: reliability: <1-3 points covering failure cases or backup options for this choice>"
        )
        geo_summary_text = geology_expert_results.get('summary') if isinstance(geology_expert_results, dict) else None
        if not geo_summary_text:
            geo_summary_text = str(geology_expert_results)
        geo_summary_text = ' '.join(str(geo_summary_text).split())
        if len(geo_summary_text) > 800:
            geo_summary_text = geo_summary_text[:800] + '...(truncated)'
        feature_summary_text = feature_analysis_results.get('summary') if isinstance(feature_analysis_results, dict) else None
        if not feature_summary_text:
            try:
                if isinstance(feature_analysis_results, dict):
                    feature_summary_text = f'keys={list(feature_analysis_results.keys())[:20]}'
                else:
                    feature_summary_text = str(feature_analysis_results)
            except Exception:
                feature_summary_text = str(feature_analysis_results)
        feature_summary_text = ' '.join(str(feature_summary_text).split())
        if len(feature_summary_text) > 800:
            feature_summary_text = feature_summary_text[:800] + '...(truncated)'
        gt = {'has_precomputed_distance': graph_topology_ready, 'node_count': int(data.shape[0]), 'node_names': []}
        if graph_topology_info:
            try:
                gt.update(graph_topology_info)
            except Exception:
                pass
        if positive_spatial_profile:
            try:
                gt["positive_spatial_profile"] = positive_spatial_profile
            except Exception:
                pass
        context = {
            'data_shape': data.shape,
            'feature_cols': feature_cols,
            'feature_count': int(len(feature_cols)),
            'class_distribution': {'positive': pos_count, 'negative': neg_count, 'excluded': excluded_count},
            'labeling_rule': label_meta,
            'training_sampling_plan': sampling_plan,
            'available_models': available_models,
            'graph_topology': gt,
            'geology_summary': geo_summary_text,
            'feature_analysis_summary': feature_summary_text,
        }
        forced_key_norm = _normalize_model_key(forced_model_key)
        if forced_key_norm and forced_key_norm in available_models and forced_key_norm in allowed_models:
            self.logger.info(f"HITL: forced model selection applied: {self._display_model_name(forced_key_norm)}")
            model_selection = f"model_key: {forced_key_norm}"
            model_selection_cot_steps: List[str] = []
        else:
            model_selection = ""
            model_selection_cot_steps = []
            try:
                cot_out = self.decide_json_cot(task, context=context, config=config, max_steps=12, default_final="")
                model_selection = str(cot_out.get("final") or "").strip()
                cot_steps_obj = cot_out.get("cot_steps")
                if isinstance(cot_steps_obj, list):
                    model_selection_cot_steps = [str(x).strip() for x in cot_steps_obj if isinstance(x, str) and str(x).strip()]
            except Exception:
                model_selection = ""
                model_selection_cot_steps = []
            if not model_selection:
                model_selection = self.decide(task, context, config=config)
        model_selection_text = model_selection.split('```python')[0].strip() if '```python' in model_selection else model_selection
        model_selection_preview = ' '.join(str(model_selection_text).split())
        if len(model_selection_preview) > 400:
            model_selection_preview = model_selection_preview[:400] + '...(truncated)'
        self.logger.info(f'Model-selection decision (summary): {model_selection_preview}')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        model_selection_md_content = ""
        try:
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            extra_lines: List[str] = []
            if model_selection_cot_steps:
                extra_lines.extend(["", "## Chain Of Thought (CoT)", ""])
                for i, s in enumerate(model_selection_cot_steps[:12], start=1):
                    extra_lines.append(f"{i}. {str(s).strip()}")
                extra_lines.append("")
            model_selection_md_content = "\n".join(
                [
                    "# Model Selection Output",
                    "",
                    f"- Timestamp: {timestamp}",
                    f"- Data shape: {data.shape}",
                    f"- Feature count: {len(feature_cols)}",
                    "",
                    "## Output Content",
                    "",
                    str(model_selection).strip(),
                    "",
                    *extra_lines,
                ]
            )
        except Exception as e:
            self.logger.warning(f'Failed to build the model-selection output content: {e}')
        prepared_data = self._prepare_prediction_data(data, geology_expert_results, feature_analysis_results)
        selected_features = feature_cols.copy()
        geo_summary = geology_expert_results.get('summary', str(geology_expert_results)[:500])
        task = f'Assess whether the current feature list is sufficient for accurate mineralization-potential prediction. Current features: {selected_features}. Data shape: {data.shape}. Geological-analysis summary: {geo_summary}'
        feature_decision = self.decide(task, config=config)
        if '```python' in feature_decision:
            text_part = feature_decision.split('```python')[0].strip()
            self.logger.info(f'Feature decision: {text_part} [generated code has been saved]')
        else:
            self.logger.info(f'Feature decision: {feature_decision}')
        _log_process_memory(self.logger, "before training")
        model_results = self._train_and_evaluate_models(
            data[selected_features],
            target_variable,
            model_selection,
            learning_mode=learning_mode,
            allowed_models=allowed_models,
            geology_expert_results=geology_expert_results,
            full_data_for_graph=prepared_data,
        )
        _log_process_memory(self.logger, "after training")
        if 'best_model' in model_results:
            predictions = self._generate_predictions(data[selected_features], model_results['best_model'], selected_features, previous_predictions, target_variable)
        else:
            predictions = {'probabilities': [0.5] * len(data), 'predictions': [0] * len(data), 'feature_importance': {feature: 0.0 for feature in selected_features}, 'message': 'No suitable best model was found', 'high_potential_count': 0}
        if isinstance(predictions, dict) and (not predictions.get('feature_importance')):
            fallback_importance = model_results.get('best_model_feature_importance')
            if isinstance(fallback_importance, list) and fallback_importance:
                predictions['feature_importance'] = fallback_importance
        try:
            model_results["_dataset_info"] = {
                "learning_mode": str(learning_mode),
                "data_shape": tuple(data.shape),
                "feature_count": int(len(selected_features)),
                "class_distribution": {"positive": pos_count, "negative": neg_count, "excluded": excluded_count},
                "label_meta": label_meta,
                "training_sampling_plan": sampling_plan,
            }
        except Exception:
            pass
        evaluation_report = self._generate_evaluation_report(model_results)
        evaluation_report_path = ''
        try:
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            evaluation_report_path = os.path.join(reports_dir, 'model_selection_and_evaluation.md')
            merged_content = (model_selection_md_content.strip() + "\n\n---\n\n" + str(evaluation_report).strip() + "\n").lstrip()
            _atomic_write_text(evaluation_report_path, merged_content)
            self.logger.info(f'Model-selection and evaluation report saved to: {evaluation_report_path}')
        except Exception as e:
            self.logger.warning(f'Failed to save the model-selection and evaluation report: {e}')
        predictions_summary = {k: v for k, v in predictions.items() if k not in ['probabilities', 'predictions', 'confidence', 'high_potential_indices']}
        predictions_summary['sample_count'] = len(predictions.get('predictions', []))
        predictions_summary['positive_count'] = sum(predictions.get('predictions', []))
        model_results_summary = {}
        if 'best_model_name' in model_results:
            model_results_summary['best_model_name'] = model_results['best_model_name']
            model_results_summary['best_model_score'] = model_results['best_model_score']
            if 'best_model' in model_results and 'classification_report' in model_results[model_results['best_model_name']]:
                model_results_summary['classification_report'] = model_results[model_results['best_model_name']]['classification_report']
        else:
            model_results_summary = str(model_results)[:1000]
        features_str = str(selected_features)
        if len(features_str) > 2000:
            features_str = features_str[:2000] + '...(truncated)'
        task = f'Generate a detailed model interpretation based on the following information:\n1. Model-results summary: {model_results_summary}\n2. Prediction-results summary: {predictions_summary}\n3. Selected features: {features_str}'
        model_explanation = self.decide(task, config=config)
        self.logger.info('Model interpretation generated.')
        som_cluster_analysis = None
        som_all_elements_analysis = None
        som_filtered_elements_analysis = None
        som_geology_interpretation = None
        mineral_qe_analysis = None
        try:
            som_input_data = None
            som_data_source = "preprocessed"
            som_use_raw_data_enabled = bool((config or {}).get("som_use_raw_data_enabled", False))
            som_all_elements_enabled = bool((config or {}).get("som_all_elements_enabled", False))
            som_result_root = os.path.join(self.output_dir, "SOM result")

            raw_candidate = som_reference_data.copy() if isinstance(som_reference_data, pd.DataFrame) else None
            preprocessed_candidate = data.copy() if isinstance(data, pd.DataFrame) else None

            raw_x_col, raw_y_col = _detect_coordinate_columns(raw_candidate) if isinstance(raw_candidate, pd.DataFrame) else (None, None)
            pre_x_col, pre_y_col = _detect_coordinate_columns(preprocessed_candidate) if isinstance(preprocessed_candidate, pd.DataFrame) else (None, None)
            raw_has_coords = bool(
                isinstance(raw_candidate, pd.DataFrame)
                and raw_x_col
                and raw_y_col
                and raw_x_col in raw_candidate.columns
                and raw_y_col in raw_candidate.columns
            )
            pre_has_coords = bool(
                isinstance(preprocessed_candidate, pd.DataFrame)
                and pre_x_col
                and pre_y_col
                and pre_x_col in preprocessed_candidate.columns
                and pre_y_col in preprocessed_candidate.columns
            )
            raw_has_features = bool(isinstance(raw_candidate, pd.DataFrame) and all(c in raw_candidate.columns for c in selected_features))
            pre_has_features = bool(
                isinstance(preprocessed_candidate, pd.DataFrame) and all(c in preprocessed_candidate.columns for c in selected_features)
            )
            raw_ready = bool(raw_has_coords and raw_has_features)
            pre_ready = bool(pre_has_coords and pre_has_features)

            if pre_ready:
                som_input_data = preprocessed_candidate
                som_data_source = "preprocessed"
                self.logger.info("SOM input source: preprocessed data")
                if som_use_raw_data_enabled and raw_ready:
                    self.logger.info("SOM raw-data auxiliary branch enabled: an additional raw-data branch will be run alongside the default preprocessed branch")
                elif som_use_raw_data_enabled and not raw_ready:
                    self.logger.warning("SOM raw-data input is enabled, but the raw data lack coordinate columns or feature columns; only the preprocessed branch will be run")
            elif som_use_raw_data_enabled and raw_ready:
                som_input_data = raw_candidate
                som_data_source = "raw"
                self.logger.warning("The preprocessed data lack coordinate columns or feature columns; falling back to the raw-data-only branch")
            else:
                self.logger.warning("The SOM input data lack coordinate columns or feature columns; the SOM clustering branch has been skipped")

            x_col, y_col = _detect_coordinate_columns(som_input_data) if isinstance(som_input_data, pd.DataFrame) else (None, None)
            if x_col and y_col and x_col in som_input_data.columns and y_col in som_input_data.columns:
                dual_source_mode = bool(som_use_raw_data_enabled and raw_ready and pre_ready)

                def _infer_ore_elements(elements: list[str], run_df: pd.DataFrame) -> list[str]:
                    ore_elements: list[str] = []
                    try:
                        ore_elements_obj = geology_expert_results.get("target_related_elements") if isinstance(geology_expert_results, dict) else None
                        if isinstance(ore_elements_obj, list):
                            ore_elements = [str(x) for x in ore_elements_obj if str(x) in run_df.columns]
                    except Exception:
                        ore_elements = []
                    if not ore_elements:
                        ore_elements = elements[:5]
                    return ore_elements

                def _collect_som_elements(run_df: pd.DataFrame) -> tuple[list[str], list[str]]:
                    filtered_elements = [c for c in selected_features if c in run_df.columns]
                    try:
                        numeric_cols = run_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                    except Exception:
                        numeric_cols = []
                    exclude_cols = {"FID", "Ore", "Longitude", "Latitude", "id", "label", "target"}
                    all_elements = [
                        c
                        for c in numeric_cols
                        if c not in exclude_cols and (not any((ex in str(c).lower() for ex in ["id", "code"])))
                    ]
                    return all_elements, filtered_elements

                def _run_som_for_source(
                    *,
                    source_name: str,
                    run_df: pd.DataFrame,
                    run_x_col: str,
                    run_y_col: str,
                ) -> Dict[str, Any]:
                    all_dirname = f"som_{source_name}_all_elements" if dual_source_mode else "som_all_elements"
                    filtered_dirname = f"som_{source_name}_filtered_elements" if dual_source_mode else "som_filtered_elements"
                    run_elements_all, run_elements_filtered = _collect_som_elements(run_df)
                    run_elements_all_for_run = run_elements_all if som_all_elements_enabled else []
                    run_all_analysis = None
                    run_filtered_analysis = None
                    if run_elements_all_for_run:
                        run_all_analysis = run_som_cluster_analysis_from_df(
                            df=run_df,
                            output_dir=os.path.join(som_result_root, all_dirname),
                            x_col=str(run_x_col),
                            y_col=str(run_y_col),
                            elements=run_elements_all_for_run,
                            ore_elements=_infer_ore_elements(run_elements_all_for_run, run_df),
                            apply_log_all=False,
                            enable_shap=False,
                        )
                    else:
                        self.logger.info(f"SOM all-elements experiment disabled; skipping the {source_name} all_elements run")
                    if run_elements_filtered:
                        run_filtered_analysis = run_som_cluster_analysis_from_df(
                            df=run_df,
                            output_dir=os.path.join(som_result_root, filtered_dirname),
                            x_col=str(run_x_col),
                            y_col=str(run_y_col),
                            elements=run_elements_filtered,
                            ore_elements=_infer_ore_elements(run_elements_filtered, run_df),
                            apply_log_all=False,
                            enable_shap=False,
                        )
                    return {
                        "all_elements": run_all_analysis,
                        "filtered_elements": run_filtered_analysis,
                        "elements_all": run_elements_all_for_run,
                        "elements_filtered": run_elements_filtered,
                        "x_col": str(run_x_col),
                        "y_col": str(run_y_col),
                        "source": source_name,
                    }

                primary_run = _run_som_for_source(
                    source_name=som_data_source,
                    run_df=som_input_data,
                    run_x_col=str(x_col),
                    run_y_col=str(y_col),
                )
                som_all_elements_analysis = primary_run.get("all_elements")
                som_filtered_elements_analysis = primary_run.get("filtered_elements")
                elements_all_for_run = primary_run.get("elements_all") or []
                elements_for_som_filtered = primary_run.get("elements_filtered") or []

                som_cluster_analysis = {
                    "all_elements": som_all_elements_analysis,
                    "filtered_elements": som_filtered_elements_analysis,
                    "elements_all": elements_all_for_run,
                    "elements_filtered": elements_for_som_filtered,
                    "som_use_raw_data_enabled": som_use_raw_data_enabled,
                    "som_data_source": som_data_source,
                }
                if dual_source_mode:
                    extra_sources: Dict[str, Any] = {}
                    if str(som_data_source) == "raw":
                        extra_df = preprocessed_candidate
                        extra_source = "preprocessed"
                        extra_x_col, extra_y_col = str(pre_x_col), str(pre_y_col)
                    else:
                        extra_df = raw_candidate
                        extra_source = "raw"
                        extra_x_col, extra_y_col = str(raw_x_col), str(raw_y_col)
                    if isinstance(extra_df, pd.DataFrame) and extra_x_col and extra_y_col:
                        extra_sources[extra_source] = _run_som_for_source(
                            source_name=extra_source,
                            run_df=extra_df,
                            run_x_col=extra_x_col,
                            run_y_col=extra_y_col,
                        )
                    som_cluster_analysis["dual_source_runs"] = extra_sources

                def _label_distribution(labels_obj: Any) -> Dict[str, int]:
                    try:
                        labels = [int(x) for x in (labels_obj or [])]
                    except Exception:
                        labels = []
                    dist: Dict[str, int] = {}
                    for v in labels:
                        k = str(v)
                        dist[k] = int(dist.get(k, 0)) + 1
                    return dist

                def _build_som_interp_context(*, tag: str, analysis_obj: Any, elements: list[str]) -> Dict[str, Any]:
                    analysis = analysis_obj if isinstance(analysis_obj, dict) else {}
                    return {
                        "run_tag": str(tag),
                        "coord_cols": [str(x_col), str(y_col)],
                        "element_count": int(len(elements)),
                        "elements_preview": [str(x) for x in elements[:30]],
                        "qe": analysis.get("qe"),
                        "te": analysis.get("te"),
                        "final_k": analysis.get("final_k"),
                        "elbow_k": (analysis.get("k_suggestion") or {}).get("elbow_k") if isinstance(analysis.get("k_suggestion"), dict) else None,
                        "cluster_label_distribution": _label_distribution(analysis.get("sample_labels")),
                        "artifacts_dir": analysis.get("artifacts_dir"),
                    }

                def _write_som_interp(*, analysis_obj: Any, text: str) -> str:
                    analysis = analysis_obj if isinstance(analysis_obj, dict) else {}
                    out_dir = analysis.get("output_dir")
                    if not out_dir:
                        return ""
                    try:
                        path = os.path.join(str(out_dir), "geological_interpretation.md")
                        _atomic_write_text(path, str(text).strip() + "\n")
                        return path
                    except Exception:
                        return ""

                som_geology_interpretation = {"all_elements": {}, "filtered_elements": {}}
                try:
                    if isinstance(som_all_elements_analysis, dict):
                        ctx = _build_som_interp_context(tag="all_elements", analysis_obj=som_all_elements_analysis, elements=elements_all_for_run)
                        prompt = (
                            "You are a geochemical mineral-exploration interpretation expert. Based on the following SOM clustering summary, provide a geological interpretation and exploration recommendations for this run.\n"
                            "Requirements:\n"
                            "1) Output plain English prose only. Do not return JSON and do not use Markdown code blocks. Short section headings are allowed.\n"
                            "2) You must mention: the size of the element set, the number of clusters and/or zoning meaning, the relationship to target mineralization-related elements, and recommendations for the next validation step.\n"
                            "3) When uncertainty exists, expressions such as 'may' or 'is likely to' are allowed.\n\n"
                            f"SOM run summary: {ctx}"
                        )
                        text = self.decide(prompt, config=config)
                        path = _write_som_interp(analysis_obj=som_all_elements_analysis, text=text)
                        som_geology_interpretation["all_elements"] = {"text": str(text).strip(), "path": str(path)}
                    if isinstance(som_filtered_elements_analysis, dict):
                        ctx = _build_som_interp_context(
                            tag="filtered_elements", analysis_obj=som_filtered_elements_analysis, elements=elements_for_som_filtered
                        )
                        prompt = (
                            "You are a geochemical mineral-exploration interpretation expert. Based on the following SOM clustering summary, provide a geological interpretation and exploration recommendations for this run.\n"
                            "Requirements:\n"
                            "1) Output plain English prose only. Do not return JSON and do not use Markdown code blocks. Short section headings are allowed.\n"
                            "2) You must mention: the analytical focus introduced by the filtered element set, the number of clusters and/or zoning meaning, the relationship to target mineralization-related elements, and recommendations for the next validation step.\n"
                            "3) When uncertainty exists, expressions such as 'may' or 'is likely to' are allowed.\n\n"
                            f"SOM run summary: {ctx}"
                        )
                        text = self.decide(prompt, config=config)
                        path = _write_som_interp(analysis_obj=som_filtered_elements_analysis, text=text)
                        som_geology_interpretation["filtered_elements"] = {"text": str(text).strip(), "path": str(path)}
                except Exception:
                    try:
                        som_geology_interpretation = {"all_elements": {}, "filtered_elements": {}}
                    except Exception:
                        som_geology_interpretation = None

                if elements_for_som_filtered or elements_all_for_run:
                    sample_id_col = "FID" if "FID" in som_input_data.columns else None
                    threshold_percentile = 95.0
                    mineral_shp_path = None
                    buffer_dist = 3000.0
                    use_smote = True
                    if isinstance(config, dict):
                        try:
                            threshold_percentile = float(config.get("threshold_percentile", threshold_percentile))
                        except Exception:
                            threshold_percentile = 95.0
                        shp_obj = config.get("mineral_shp") or config.get("mineral_shp_path")
                        if shp_obj:
                            mineral_shp_path = str(shp_obj)
                        try:
                            buffer_dist = float(config.get("buffer_dist", buffer_dist))
                        except Exception:
                            buffer_dist = 3000.0
                        if "use_smote" in config:
                            use_smote = bool(config.get("use_smote"))
                    def _fallback_som_output_dir(source_name: str, run_tag: str) -> str:
                        run_tag_norm = "all_elements" if str(run_tag) == "all_elements" else "filtered_elements"
                        if dual_source_mode:
                            return os.path.join(som_result_root, f"som_{source_name}_{run_tag_norm}")
                        return os.path.join(som_result_root, f"som_{run_tag_norm}")

                    def _run_qe_for_container(run_container: Dict[str, Any], run_df: pd.DataFrame, run_x_col: str, run_y_col: str) -> None:
                        nonlocal mineral_qe_analysis
                        source_name = str(run_container.get("source") or som_data_source)
                        for run_tag, elements_key in (("all_elements", "elements_all"), ("filtered_elements", "elements_filtered")):
                            run_analysis = run_container.get(run_tag)
                            run_elements = run_container.get(elements_key) or []
                            if not isinstance(run_analysis, dict) or not run_elements:
                                continue
                            mineral_elements = [str(x) for x in run_elements if str(x).strip()]
                            if not mineral_elements:
                                continue
                            run_output_dir = run_analysis.get("output_dir")
                            if not run_output_dir:
                                run_output_dir = _fallback_som_output_dir(source_name, run_tag)
                            run_cluster_results_dir = run_analysis.get("artifacts_dir")
                            if not run_cluster_results_dir:
                                run_cluster_results_dir = os.path.join(str(run_output_dir), "Sample_Cluster_Results")
                            run_qe_analysis = run_mineral_qe_analysis_from_df(
                                df=run_df,
                                output_dir=str(run_output_dir),
                                cluster_results_dir=run_cluster_results_dir,
                                qe_dirname="qe",
                                x_col=str(run_x_col),
                                y_col=str(run_y_col),
                                sample_id_col=sample_id_col,
                                mineral_elements=mineral_elements,
                                threshold_percentile=threshold_percentile,
                                mineral_shp_path=mineral_shp_path,
                                buffer_dist=buffer_dist,
                                use_smote=use_smote,
                                llm=self.llm,
                            )
                            run_analysis["qe_analysis"] = run_qe_analysis
                            if run_tag == "filtered_elements":
                                mineral_qe_analysis = run_qe_analysis

                    _run_qe_for_container(som_cluster_analysis, som_input_data, str(x_col), str(y_col))
                    dual_source_runs = som_cluster_analysis.get("dual_source_runs")
                    if isinstance(dual_source_runs, dict):
                        for source_name, run_container_obj in dual_source_runs.items():
                            if not isinstance(run_container_obj, dict):
                                continue
                            if str(source_name) == "raw":
                                run_df = raw_candidate
                                run_x_col, run_y_col = raw_x_col, raw_y_col
                            else:
                                run_df = preprocessed_candidate
                                run_x_col, run_y_col = pre_x_col, pre_y_col
                            if not isinstance(run_df, pd.DataFrame) or not run_x_col or not run_y_col:
                                continue
                            _run_qe_for_container(run_container_obj, run_df, str(run_x_col), str(run_y_col))
        except Exception as e:
            try:
                self.logger.warning(f"Failed to generate SOM clustering/QE artifacts: {e}")
            except Exception:
                pass

        results = {'prepared_data': prepared_data, 'selected_features': selected_features, 'model_results': model_results, 'predictions': predictions, 'summary': self._generate_summary(model_results, predictions), 'model_selection': model_selection, 'model_selection_cot_steps': model_selection_cot_steps, 'model_explanation': model_explanation, 'evaluation_report': evaluation_report, 'evaluation_report_path': evaluation_report_path, 'som_cluster_analysis': som_cluster_analysis, 'som_geology_interpretation': som_geology_interpretation, 'mineral_qe_analysis': mineral_qe_analysis}
        self.add_memory({'action': 'prediction', 'result': results})
        return results
    def _filter_feature_cols_by_elements(self, feature_cols: List[str], target_elements: List[str]) -> List[str]:
        if not target_elements:
            return feature_cols
        selected = []
        for col in feature_cols:
            col_lower = str(col).lower()
            for elem in target_elements:
                elem_lower = str(elem).lower()
                pattern = r'(?<![a-z0-9])' + re.escape(elem_lower) + r'(?![a-z0-9])'
                if elem_lower == col_lower or re.search(pattern, col_lower):
                    selected.append(col)
                    break
        return selected if selected else feature_cols
    def _prepare_prediction_data(self, data: pd.DataFrame, geology_expert_results: Dict, feature_analysis_results: Dict) -> pd.DataFrame:
        prepared_df = data.copy()
        if 'anomaly_analysis' in geology_expert_results:
            pass
        return prepared_df

    def _compute_ca_anomaly_score(self, *, data: pd.DataFrame, geology_expert_results: Dict[str, Any], fallback_feature_cols: Optional[List[str]] = None) -> Tuple[pd.Series, List[str]]:
        elements: List[str] = []
        try:
            target_elements_obj = geology_expert_results.get("target_related_elements") if isinstance(geology_expert_results, dict) else None
            if isinstance(target_elements_obj, list):
                elements = [str(x) for x in target_elements_obj if str(x) in data.columns]
        except Exception:
            elements = []
        anomaly_analysis = geology_expert_results.get("anomaly_analysis") if isinstance(geology_expert_results, dict) else None
        element_anomalies = anomaly_analysis.get("element_anomalies", {}) if isinstance(anomaly_analysis, dict) else {}
        if not elements:
            try:
                ranked: List[Tuple[str, float]] = []
                if isinstance(element_anomalies, dict):
                    for k, v in element_anomalies.items():
                        if str(k) not in data.columns:
                            continue
                        if not isinstance(v, dict):
                            continue
                        try:
                            pct = float(v.get("anomaly_percentage", 0.0))
                        except Exception:
                            pct = 0.0
                        ranked.append((str(k), pct))
                ranked.sort(key=lambda x: x[1], reverse=True)
                elements = [k for k, _ in ranked[:12]]
            except Exception:
                elements = []
        if not elements and fallback_feature_cols:
            elements = [str(x) for x in fallback_feature_cols[:12] if str(x) in data.columns]
        if not elements:
            return (pd.Series(np.zeros(len(data), dtype=float), index=data.index), [])
        acc = np.zeros(len(data), dtype=np.float64)
        used = 0
        for e in elements:
            stats = element_anomalies.get(e) if isinstance(element_anomalies, dict) else None
            if not isinstance(stats, dict):
                continue
            thr = stats.get("threshold")
            try:
                thr_f = float(thr)
            except Exception:
                continue
            if not (thr_f > 0.0):
                continue
            x = pd.to_numeric(data[e], errors="coerce").astype(float)
            ratio = (x / float(thr_f)).to_numpy(dtype=np.float64)
            score_e = np.where(ratio >= 1.0, ratio * ratio, ratio)
            score_e = np.nan_to_num(score_e, nan=0.0, posinf=0.0, neginf=0.0)
            acc += score_e
            used += 1
        if used <= 0:
            return (pd.Series(np.zeros(len(data), dtype=float), index=data.index), [])
        score = acc / float(used)
        return (pd.Series(score, index=data.index), elements)

    def _labels_by_score(self, *, score: pd.Series, pos_q: float, neg_q: float, min_pos: int, min_neg: int) -> pd.Series:
        s = pd.to_numeric(score, errors="coerce").astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        fill_val = float(np.nanmedian(np.asarray(s, dtype=float))) if np.isfinite(np.nanmedian(np.asarray(s, dtype=float))) else 0.0
        s = s.fillna(fill_val)
        y = pd.Series([-1] * len(s), index=s.index, dtype=int)
        try:
            thr_pos = float(s.quantile(float(pos_q)))
        except Exception:
            thr_pos = float(np.nanmax(np.asarray(s, dtype=float)))
        try:
            thr_neg = float(s.quantile(float(neg_q)))
        except Exception:
            thr_neg = float(np.nanmin(np.asarray(s, dtype=float)))
        pos_mask = s >= thr_pos
        neg_mask = s <= thr_neg
        if int(pos_mask.sum()) < int(min_pos):
            top_idx = s.sort_values(ascending=False).head(int(min_pos)).index
            pos_mask = s.index.isin(top_idx)
        if int(neg_mask.sum()) < int(min_neg):
            bot_idx = s.sort_values(ascending=True).head(int(min_neg)).index
            neg_mask = s.index.isin(bot_idx)
        overlap = pos_mask & neg_mask
        if bool(overlap.any()):
            overlap_idx = s.index[overlap]
            order = s.loc[overlap_idx].sort_values(ascending=False).index
            keep_pos = set(order[: max(1, int(len(order) / 2))])
            pos_mask = pos_mask & (~s.index.isin(overlap_idx) | s.index.isin(list(keep_pos)))
            neg_mask = neg_mask & (~s.index.isin(overlap_idx) | ~s.index.isin(list(keep_pos)))
        y.loc[neg_mask] = 0
        y.loc[pos_mask] = 1
        return y

    def _define_labels_supervised(self, *, data: pd.DataFrame, geology_expert_results: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, Dict[str, Any]]:
        if "Ore" in data.columns:
            ore = pd.to_numeric(data["Ore"], errors="coerce")
            y = pd.Series([-1] * len(data), index=data.index, dtype=int)
            known = ore.notna()
            y.loc[known] = 0
            y.loc[ore == 1] = 1
            y2, expansion_meta = self._expand_positive_samples_if_needed(data=data, y=y)
            meta = {
                "learning_mode": "supervised",
                "label_rule": "Ore==1 is positive and all other known Ore values are negative. Missing Ore is treated as excluded (-1). When the positive-sample ratio is too low, additional positives are expanded from negative samples (0) using coordinate-neighbor proximity.",
                "label_source": "Ore",
                "positive_expansion": expansion_meta,
            }
            return (y2, meta)
        potential = []
        try:
            potential = geology_expert_results.get("potential_areas", []) if isinstance(geology_expert_results, dict) else []
        except Exception:
            potential = []
        y = pd.Series([0] * len(data), index=data.index, dtype=int)
        if isinstance(potential, list) and potential:
            try:
                pos_idx = [data.index[int(i)] for i in potential if int(i) >= 0 and int(i) < len(data)]
                y.loc[pos_idx] = 1
            except Exception:
                pass
        return (y, {"learning_mode": "supervised", "label_rule": "Fallback rule: potential_areas are positive and all remaining samples are negative when Ore is unavailable.", "label_source": "geology_expert_results.potential_areas"})

    def _define_labels_self_supervised(self, *, data: pd.DataFrame, geology_expert_results: Dict[str, Any], fallback_feature_cols: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, Dict[str, Any]]:
        if "Ore" in data.columns:
            ore = pd.to_numeric(data["Ore"], errors="coerce")
            y = pd.Series([-1] * len(data), index=data.index, dtype=int)
            known = ore.notna()
            y.loc[known] = 0
            y.loc[ore == 1] = 1
            meta = {
                "learning_mode": "self_supervised",
                "label_rule": "Initial stage: randomly sample the same number of negatives from label=0 as the positive samples. Training stage: generate pseudo-labels from high-confidence model predictions and iteratively expand the training set. All remaining samples are excluded (-1).",
                "label_source": "Ore",
            }
            return (y, meta)
        y = pd.Series([-1] * len(data), index=data.index, dtype=int)
        return (
            y,
            {
                "learning_mode": "self_supervised",
                "label_rule": "Without Ore labels, negative sampling from label=0 cannot be initialized; all samples are treated as unlabeled and excluded (-1).",
                "label_source": "none",
            },
        )

    def _define_labels_unsupervised(self, *, data: pd.DataFrame, geology_expert_results: Dict[str, Any], fallback_feature_cols: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, Dict[str, Any]]:
        if "Ore" in data.columns:
            ore = pd.to_numeric(data["Ore"], errors="coerce").fillna(0).astype(int)
            y = ore.where(ore == 1, 0).astype(int)
            return (y, {"learning_mode": "unsupervised", "label_rule": "When Ore labels exist, Ore==1 is positive and all remaining Ore values are negative. Labels are not used for training and are retained only for evaluation and stratified splitting.", "label_source": "Ore"})
        y = pd.Series([0] * len(data), index=data.index, dtype=int)
        return (y, {"learning_mode": "unsupervised", "label_rule": "Without Ore labels, all samples are treated as unlabeled for unsupervised training and no labels are used in evaluation.", "label_source": "none"})

    def _define_labels_by_learning_mode(
        self,
        *,
        data: pd.DataFrame,
        geology_expert_results: Dict[str, Any],
        learning_mode: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        mode = str(learning_mode or "").strip().lower()
        fallback_feature_cols = None
        try:
            exclude_cols = {'FID', 'Ore', 'Longitude', 'Latitude', 'id', 'label', 'target'}
            fallback_feature_cols = [c for c in data.columns if c not in exclude_cols]
        except Exception:
            fallback_feature_cols = None
        if mode == "unsupervised":
            return self._define_labels_unsupervised(data=data, geology_expert_results=geology_expert_results, fallback_feature_cols=fallback_feature_cols, config=config)
        if mode == "self_supervised":
            return self._define_labels_self_supervised(data=data, geology_expert_results=geology_expert_results, fallback_feature_cols=fallback_feature_cols, config=config)
        return self._define_labels_supervised(data=data, geology_expert_results=geology_expert_results, config=config)
    def _create_target_variable(self, data: pd.DataFrame, geology_expert_results: Dict) -> pd.Series:
        self.logger.info('Creating target variable...')
        if 'Ore' not in data.columns:
            self.logger.warning('Warning: the Ore column is missing; a zero-only target array will be returned')
            return pd.Series([0] * len(data))
        target = data['Ore'].astype(int)
        target = target.where(target == 1, 0)
        pos_count = (target == 1).sum()
        neg_count = (target == 0).sum()
        self.logger.info(f'Target variable created. Positive samples: {pos_count}, negative samples: {neg_count}')
        return target
    def _train_and_evaluate_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_selection: str,
        *,
        learning_mode: Optional[str] = None,
        allowed_models: Optional[List[str]] = None,
        geology_expert_results: Optional[Dict[str, Any]] = None,
        full_data_for_graph: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        viz_dir = self._get_model_viz_dir()
        results: Dict[str, Any] = {"model_viz_dir": viz_dir}

        def _get_scores(m: Any, x: pd.DataFrame) -> Optional[np.ndarray]:
            if hasattr(m, "predict_proba"):
                proba = m.predict_proba(x)
                try:
                    proba_arr = np.asarray(proba)
                except Exception:
                    proba_arr = proba
                if np.ndim(proba_arr) == 2 and getattr(proba_arr, "shape", (0, 0))[1] >= 2:
                    return np.asarray(proba_arr[:, 1])
                if np.ndim(proba_arr) == 1:
                    return np.asarray(proba_arr)
            if hasattr(m, "decision_function"):
                decision = np.asarray(m.decision_function(x))
                return decision.reshape(-1)
            return None

        def _downsample_training_set(
            x_train: pd.DataFrame,
            y_train: pd.Series,
            *,
            positive_label: int = 1,
            negative_label: int = 0,
            negative_to_positive_ratio: float = 1.0,
            min_negative: int = 0,
            random_state: int = 42,
        ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
            y_series = pd.Series(y_train, index=x_train.index)
            pos_idx = y_series[y_series == positive_label].index
            neg_idx = y_series[y_series == negative_label].index
            n_pos = int(len(pos_idx))
            n_neg = int(len(neg_idx))
            if n_pos == 0 or n_neg == 0:
                return x_train, y_series, {"train_pos": n_pos, "train_neg": n_neg, "train_used_pos": n_pos, "train_used_neg": n_neg}
            target_neg = int(round(n_pos * float(negative_to_positive_ratio)))
            if int(min_negative) > 0:
                target_neg = max(target_neg, int(min_negative))
            target_neg = min(target_neg, n_neg)
            sampled_neg_idx = y_series.loc[neg_idx].sample(n=target_neg, random_state=random_state).index
            keep_idx = pos_idx.union(sampled_neg_idx)
            x_used = x_train.loc[keep_idx].copy()
            y_used = y_series.loc[keep_idx].copy()
            return x_used, y_used, {"train_pos": n_pos, "train_neg": n_neg, "train_used_pos": int((y_used == positive_label).sum()), "train_used_neg": int((y_used == negative_label).sum())}

        mode_lower = str(learning_mode or "").strip().lower()
        holdout_role = "validation" if mode_lower == "supervised" else "test"

        X_all = X.copy()
        y_all = pd.Series(y, index=X_all.index) if not isinstance(y, pd.Series) else y.reindex(X_all.index)
        y_all = y_all.fillna(-1).astype(int)

        labeled_mask = y_all.isin([0, 1])
        X_labeled = X_all[labeled_mask].copy()
        y_labeled = y_all[labeled_mask].copy().astype(int)

        X_train_labeled = X_labeled
        y_train_labeled = y_labeled
        X_test = X_labeled.iloc[:0].copy()
        y_test = y_labeled.iloc[:0].copy()
        if len(X_labeled) > 0:
            can_stratify = _can_stratify_binary(y_labeled)
            try:
                X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(
                    X_labeled,
                    y_labeled,
                    test_size=0.3,
                    random_state=42,
                    stratify=y_labeled if can_stratify else None,
                )
            except Exception:
                X_train_labeled, X_test, y_train_labeled, y_test = train_test_split(X_labeled, y_labeled, test_size=0.3, random_state=42)

        X_train_used = X_train_labeled
        y_train_used = y_train_labeled
        train_sampling_stats = {
            "train_pos": int((pd.Series(y_train_labeled) == 1).sum()) if len(y_train_labeled) else 0,
            "train_neg": int((pd.Series(y_train_labeled) == 0).sum()) if len(y_train_labeled) else 0,
            "train_used_pos": int((pd.Series(y_train_labeled) == 1).sum()) if len(y_train_labeled) else 0,
            "train_used_neg": int((pd.Series(y_train_labeled) == 0).sum()) if len(y_train_labeled) else 0,
        }
        if mode_lower == "supervised" and len(X_train_labeled) > 0 and _can_stratify_binary(y_train_labeled):
            X_train_used, y_train_used, train_sampling_stats = _downsample_training_set(X_train_labeled, y_train_labeled, negative_to_positive_ratio=1.0, random_state=42)

        results["train_test_split"] = {
            "train_size": int(len(X_train_labeled)),
            "train_used_size": int(len(X_train_used)),
            "test_size": int(len(X_test)),
            "val_size": int(len(X_test)),
            "holdout_role": str(holdout_role),
            "all_size": int(len(X_all)),
            "train_pos": int(train_sampling_stats.get("train_pos", 0)),
            "train_neg": int(train_sampling_stats.get("train_neg", 0)),
            "train_used_pos": int(train_sampling_stats.get("train_used_pos", 0)),
            "train_used_neg": int(train_sampling_stats.get("train_used_neg", 0)),
        }

        y_fit = pd.Series([-1] * len(y_all), index=y_all.index, dtype=int)
        if len(y_train_used) > 0:
            y_fit.loc[y_train_used.index] = pd.Series(y_train_used, index=y_train_used.index).astype(int)

        estimator = clone(self.models["som"])
        estimator.fit(X_all, y_fit)

        train_scores = _get_scores(estimator, X_train_used if len(X_train_used) > 0 else X_all)
        holdout_scores = _get_scores(estimator, X_test) if len(X_test) > 0 else None

        train_metrics: Dict[str, Any] = {}
        test_metrics: Dict[str, Any] = {}
        train_cm = None
        cm = None

        if train_scores is not None and len(y_train_used) > 0 and _can_stratify_binary(y_train_used):
            y_train_arr = np.asarray(y_train_used).astype(int)
            s = np.asarray(train_scores, dtype=float).reshape(-1)
            train_metrics["roc_auc"] = float(roc_auc_score(y_train_arr, s))
            train_metrics["pr_auc"] = float(average_precision_score(y_train_arr, s))
            preds_train = (s >= 0.5).astype(int)
            train_metrics["accuracy"] = float(accuracy_score(y_train_arr, preds_train))
            train_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_train_arr, preds_train))
            train_metrics["precision_pos"] = float(precision_score(y_train_arr, preds_train, zero_division=0))
            train_metrics["recall_pos"] = float(recall_score(y_train_arr, preds_train, zero_division=0))
            train_metrics["f1_pos"] = float(f1_score(y_train_arr, preds_train, zero_division=0))
            train_metrics["kappa"] = float(cohen_kappa_score(y_train_arr, preds_train))
            train_cm = confusion_matrix(y_train_arr, preds_train).tolist()

        if holdout_scores is not None and len(y_test) > 0 and _can_stratify_binary(y_test):
            y_test_arr = np.asarray(y_test).astype(int)
            s = np.asarray(holdout_scores, dtype=float).reshape(-1)
            test_metrics["roc_auc"] = float(roc_auc_score(y_test_arr, s))
            test_metrics["pr_auc"] = float(average_precision_score(y_test_arr, s))
            preds_holdout = (s >= 0.5).astype(int)
            test_metrics["accuracy"] = float(accuracy_score(y_test_arr, preds_holdout))
            test_metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test_arr, preds_holdout))
            test_metrics["precision_pos"] = float(precision_score(y_test_arr, preds_holdout, zero_division=0))
            test_metrics["recall_pos"] = float(recall_score(y_test_arr, preds_holdout, zero_division=0))
            test_metrics["f1_pos"] = float(f1_score(y_test_arr, preds_holdout, zero_division=0))
            test_metrics["kappa"] = float(cohen_kappa_score(y_test_arr, preds_holdout))
            cm = confusion_matrix(y_test_arr, preds_holdout).tolist()

        fit_status = "unknown"
        auc_gap = None
        if train_metrics.get("roc_auc") is not None and test_metrics.get("roc_auc") is not None:
            try:
                auc_gap = float(train_metrics["roc_auc"]) - float(test_metrics["roc_auc"])
                if auc_gap > 0.05:
                    fit_status = "overfit"
                elif float(train_metrics["roc_auc"]) < 0.6 and float(test_metrics["roc_auc"]) < 0.6:
                    fit_status = "underfit"
                else:
                    fit_status = "ok"
            except Exception:
                fit_status = "unknown"
                auc_gap = None
        recommendations: List[str] = []
        if fit_status == "overfit":
            recommendations = ["Decrease the grid size (grid_m/grid_n)", "Increase sigma", "Reduce n_iter", "Increase decision_threshold"]
        elif fit_status == "underfit":
            recommendations = ["Increase the grid size (grid_m/grid_n)", "Decrease sigma", "Increase n_iter", "Decrease decision_threshold"]
        fit_diagnosis = {"status": fit_status, "auc_gap": auc_gap, "recommendations": recommendations}

        som_result: Dict[str, Any] = {
            "model": estimator,
            "tuned": False,
            "tuning_info": None,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_confusion_matrix": train_cm,
            "confusion_matrix": cm,
            "fit_diagnosis": fit_diagnosis,
            "holdout_scores": holdout_scores.tolist() if isinstance(holdout_scores, np.ndarray) else (list(holdout_scores) if holdout_scores is not None else None),
            "train_scores": train_scores.tolist() if isinstance(train_scores, np.ndarray) else (list(train_scores) if train_scores is not None else None),
        }
        results["som"] = som_result
        best_model = estimator
        best_model_name = "som"
        best_roc = test_metrics.get("roc_auc") if isinstance(test_metrics, dict) else None
        best_score = float(best_roc) if best_roc is not None else 0.0
        best_model_metric = "roc_auc"

        results["best_model"] = best_model
        results["best_model_name"] = best_model_name
        results["best_model_score"] = best_score
        results["best_model_metric"] = best_model_metric
        results["best_model_fit_diagnosis"] = fit_diagnosis

        try:
            best_details = results.get(best_model_name)
            if isinstance(best_details, dict):
                artifacts = best_details.get("artifacts") if isinstance(best_details.get("artifacts"), dict) else {}
                holdout_label = "Val" if str(holdout_role).strip().lower() == "validation" else "Test"
                model_display = self._display_model_name(best_model_name)

                if holdout_scores is not None and len(y_test) > 0 and _can_stratify_binary(y_test):
                    try:
                        roc_path = os.path.join(viz_dir, "best_model_roc_curve.png")
                        pr_path = os.path.join(viz_dir, "best_model_pr_curve.png")
                        prob_path = os.path.join(viz_dir, "best_model_score_distribution.png")
                        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                        if train_scores is not None and len(y_train_used) > 0 and _can_stratify_binary(y_train_used):
                            curves["Train"] = (np.asarray(y_train_used).astype(int), np.asarray(train_scores, dtype=float))
                        curves[holdout_label] = (np.asarray(y_test).astype(int), np.asarray(holdout_scores, dtype=float))
                        plotted = self._plot_roc_curve(out_path=roc_path, title=f"{model_display} ROC Curves", curves=curves)
                        if plotted:
                            artifacts["roc_curve_path"] = plotted
                        try:
                            y_holdout_np = np.asarray(y_test).astype(int)
                            holdout_score_np = np.asarray(holdout_scores, dtype=float)
                            artifacts["pr_curve_path"] = self._plot_pr_curve(y_true=y_holdout_np, scores=holdout_score_np, out_path=pr_path, title=f"{model_display} PR Curve ({holdout_label})")
                            artifacts["score_hist_path"] = self._plot_probability_histogram(y_true=y_holdout_np, scores=holdout_score_np, out_path=prob_path, title=f"{model_display} Score Distribution")
                        except Exception:
                            pass
                    except Exception:
                        pass

                if train_cm is not None:
                    try:
                        cm_train_path = os.path.join(viz_dir, "best_model_train_confusion_matrix.png")
                        artifacts["train_confusion_matrix_path"] = self._plot_confusion_matrix_heatmap(
                            cm=np.asarray(train_cm), out_path=cm_train_path, title=f"{model_display} Train Confusion Matrix"
                        )
                    except Exception:
                        pass
                if cm is not None:
                    try:
                        cm_path = os.path.join(viz_dir, "best_model_confusion_matrix.png")
                        cm_title = f"{model_display} Validation Confusion Matrix" if holdout_label == "Val" else f"{model_display} Test Confusion Matrix"
                        artifacts["confusion_matrix_path"] = self._plot_confusion_matrix_heatmap(cm=np.asarray(cm), out_path=cm_path, title=cm_title)
                    except Exception:
                        pass

                predict_cm = None
                try:
                    y_all_arr = np.asarray(y_all)
                    mask_all = np.isin(y_all_arr, [0, 1])
                    if bool(np.any(mask_all)):
                        scores_all = _get_scores(best_model, X_all)
                        if scores_all is not None:
                            s_all = np.asarray(scores_all, dtype=float).reshape(-1)
                            if s_all.size == y_all_arr.size:
                                y_true_all = np.asarray(y_all_arr)[mask_all].astype(int)
                                preds_all = (np.asarray(s_all)[mask_all] >= 0.5).astype(int)
                                predict_cm = confusion_matrix(y_true_all, preds_all).tolist()
                except Exception:
                    predict_cm = None

                if predict_cm is not None:
                    try:
                        cm_predict_path = os.path.join(viz_dir, "best_model_predict_confusion_matrix.png")
                        artifacts["predict_confusion_matrix_path"] = self._plot_confusion_matrix_heatmap(
                            cm=np.asarray(predict_cm), out_path=cm_predict_path, title=f"{model_display} Predict Confusion Matrix"
                        )
                        best_details["predict_confusion_matrix"] = predict_cm
                    except Exception:
                        pass

                if train_metrics.get("roc_auc") is not None and test_metrics.get("roc_auc") is not None:
                    try:
                        fit_gap_path = os.path.join(viz_dir, "best_model_fit_gap.png")
                        artifacts["fit_gap_path"] = self._plot_fit_gap(
                            train_auc=train_metrics.get("roc_auc"), val_auc=test_metrics.get("roc_auc"), out_path=fit_gap_path, title=f"{model_display} Fit Gap (AUC)"
                        )
                    except Exception:
                        pass

                if artifacts:
                    best_details["artifacts"] = artifacts
                    results[best_model_name] = best_details
        except Exception:
            pass

        try:
            if len(X_test) > 0 and _can_stratify_binary(y_test):
                perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=1)
                perm_values = np.asarray(perm.importances_mean, dtype=float)
                perm_std = None
                try:
                    perm_std = np.asarray(perm.importances_std, dtype=float)
                except Exception:
                    perm_std = None
                order = np.argsort(perm_values)[::-1]
                perm_importance: List[Dict[str, Any]] = []
                for idx in order.tolist():
                    row: Dict[str, Any] = {"feature": str(X_test.columns[int(idx)]), "importance": float(perm_values[int(idx)])}
                    if perm_std is not None and int(idx) < int(len(perm_std)):
                        row["std"] = float(perm_std[int(idx)])
                    perm_importance.append(row)
                if isinstance(results.get(best_model_name), dict):
                    results[best_model_name]["permutation_importance"] = perm_importance
                results["best_model_permutation_importance"] = perm_importance
                results["best_model_feature_importance"] = perm_importance
                artifacts = {}
                best_details = results.get(best_model_name)
                if isinstance(best_details, dict) and isinstance(best_details.get("artifacts"), dict):
                    artifacts = dict(best_details.get("artifacts") or {})
                if perm_importance:
                    perm_path = os.path.join(viz_dir, "best_model_permutation_importance.png")
                    plotted = self._plot_feature_importance_bar(
                        importance=perm_importance, out_path=perm_path, title=f"{self._display_model_name(best_model_name)} Permutation Importance", top_n=None, show_legend=False
                    )
                    if plotted:
                        artifacts["permutation_importance_path"] = plotted
                if isinstance(best_details, dict):
                    best_details["artifacts"] = artifacts
                    results[best_model_name] = best_details
        except Exception as e:
            self.logger.warning(f"Failed to compute permutation feature importance: {e}")

        self.best_model = best_model
        self.model_name = best_model_name
        return results
    def _generate_evaluation_report(self, model_results: Dict[str, Any]) -> str:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines: List[str] = []
        lines.append('# Model Evaluation Report')
        lines.append('')
        lines.append(f'- Generated at: {timestamp}')
        split = model_results.get('train_test_split')
        holdout_label = "Test set"
        if isinstance(split, dict):
            role = str(split.get("holdout_role") or "").strip().lower()
            if role == "validation":
                holdout_label = "Validation set"
            holdout_size = split.get("val_size") if holdout_label == "Validation set" else split.get("test_size")
            lines.append(f"- Training set (full): {split.get('train_size')}; training set (after sampling): {split.get('train_used_size')}; {holdout_label}: {holdout_size}")
            lines.append(f"- Training-class counts (full): pos={split.get('train_pos')} neg={split.get('train_neg')}; training-class counts (after sampling): pos={split.get('train_used_pos')} neg={split.get('train_used_neg')}")
            if split.get("all_size") is not None:
                lines.append(f"- Prediction set (full dataset): {split.get('all_size')}")
        dataset_info = model_results.get("_dataset_info")
        if isinstance(dataset_info, dict):
            lines.append("")
            lines.append("## Dataset Overview")
            learning_mode = dataset_info.get("learning_mode")
            if learning_mode:
                lines.append(f"- Learning mode: {learning_mode}")
            ds_shape = dataset_info.get("data_shape")
            if ds_shape:
                lines.append(f"- Data shape: {ds_shape}")
            feature_count = dataset_info.get("feature_count")
            if feature_count is not None:
                lines.append(f"- Feature count: {feature_count}")
            dist = dataset_info.get("class_distribution")
            if isinstance(dist, dict):
                lines.append(f"- Effective sample distribution (after excluding excluded=-1): pos={dist.get('positive')} neg={dist.get('negative')}; excluded={dist.get('excluded')}")
            label_meta = dataset_info.get("label_meta")
            if isinstance(label_meta, dict):
                label_source = label_meta.get("label_source")
                label_rule = label_meta.get("label_rule")
                if label_source:
                    lines.append(f"- Label source: {label_source}")
                if label_rule:
                    lines.append(f"- Label rule: {label_rule}")
                exp = label_meta.get("positive_expansion")
                if isinstance(exp, dict) and exp.get("enabled"):
                    trig = bool(exp.get("triggered"))
                    coord_cols = exp.get("coord_cols")
                    coord_text = ""
                    if isinstance(coord_cols, dict) and coord_cols.get("x") and coord_cols.get("y"):
                        coord_text = f", coordinate columns=({coord_cols.get('x')},{coord_cols.get('y')})"
                    if trig:
                        lines.append(
                            f"- Positive-sample expansion: triggered (threshold={exp.get('ratio_threshold')}, k={exp.get('k_neighbors')})"
                            f"; original positives={exp.get('original_positive_count')}, added positives={exp.get('added_positive_count')}, expanded positives={exp.get('expanded_positive_count')}{coord_text}"
                        )
                    else:
                        lines.append(f"- Positive-sample expansion: not triggered (reason={exp.get('reason')}){coord_text}")
        lines.append('')
        lines.append('## Selection-And-Tuning Procedure (DataScienceExpertAgent)')
        lines.append('- Data splitting: remove samples with excluded=-1 first, then perform a 70/30 split on effective samples (training/holdout, random_state=42) with stratification when possible.')
        lines.append('- Model: SOM(QE). SOM is trained on the training subset, and quantization error (QE) is used as the anomaly/mineralization-tendency score.')
        lines.append('- Probability output: when the training set contains both positive and negative samples (0/1), QE->LogisticRegression is used for calibration; otherwise a relative QE-based score in the range 0~1 is reported for ranking.')
        lines.append('')
        best_name = model_results.get('best_model_name')
        best_name_display = self._display_model_name(best_name) if best_name else None
        best_score = model_results.get('best_model_score')
        best_metric = model_results.get('best_model_metric', 'roc_auc')
        if best_name:
            lines.append('## Best Model')
            lines.append(f'- Model: {best_name_display}')
            if best_score is not None:
                metric_name = 'Cross-validation ROC-AUC' if 'roc' in str(best_metric).lower() else 'Cross-validation score'
                lines.append(f'- {metric_name}: {best_score:.4f}')
            best_details = model_results.get(best_name)
            if isinstance(best_details, dict):
                final_params = None
                try:
                    model_obj = best_details.get("model")
                    if model_obj is not None and hasattr(model_obj, "get_params"):
                        final_params = dict(model_obj.get_params(deep=False) or {})
                except Exception:
                    final_params = None
                best_params = None
                best_params_source = None
                sec = best_details.get('secondary_tuning')
                if isinstance(sec, dict) and sec.get('activated') and sec.get('selected') and isinstance(sec.get('best_params'), dict) and sec.get('best_params'):
                    best_params = sec.get('best_params')
                    best_params_source = 'secondary_tuning'
                tuning = best_details.get('tuning')
                if best_params is None and isinstance(tuning, dict) and isinstance(tuning.get('best_params'), dict) and tuning.get('best_params'):
                    best_params = tuning.get('best_params')
                    best_params_source = str(tuning.get("strategy") or "tuning")
                if isinstance(best_params, dict) and best_params:
                    lines.append(f"- Best parameters ({best_params_source}): {best_params}")
                if isinstance(tuning, dict) and tuning.get("best_cv_score") is not None:
                    try:
                        lines.append(f"- Selection-stage score (best_cv_score): {float(tuning.get('best_cv_score')):.4f}")
                    except Exception:
                        pass
                if isinstance(final_params, dict) and final_params:
                    key_order = []
                    if best_name == "som":
                        key_order = ["grid_m", "grid_n", "n_iter", "sigma", "lr", "calibrate", "decision_threshold", "random_state"]
                    picked = {k: final_params.get(k) for k in key_order if k in final_params}
                    if picked:
                        lines.append(f"- Final model key parameters: {picked}")
                train_metrics = best_details.get('train_metrics') if isinstance(best_details.get('train_metrics'), dict) else {}
                test_metrics = best_details.get('test_metrics') if isinstance(best_details.get('test_metrics'), dict) else {}
                train_parts = []
                test_parts = []
                if train_metrics.get('roc_auc') is not None:
                    train_parts.append(f"ROC-AUC={float(train_metrics.get('roc_auc')):.4f}")
                if train_metrics.get('pr_auc') is not None:
                    train_parts.append(f"PR-AUC={float(train_metrics.get('pr_auc')):.4f}")
                if train_metrics.get('accuracy') is not None:
                    train_parts.append(f"Accuracy={float(train_metrics.get('accuracy')):.4f}")
                if train_metrics.get('balanced_accuracy') is not None:
                    train_parts.append(f"BalancedAcc={float(train_metrics.get('balanced_accuracy')):.4f}")
                if train_metrics.get('precision_pos') is not None:
                    train_parts.append(f"Precision(positive)={float(train_metrics.get('precision_pos')):.4f}")
                if train_metrics.get('recall_pos') is not None:
                    train_parts.append(f"Recall(positive)={float(train_metrics.get('recall_pos')):.4f}")
                if train_metrics.get('f1_pos') is not None:
                    train_parts.append(f"F1(positive)={float(train_metrics.get('f1_pos')):.4f}")
                if train_metrics.get('kappa') is not None:
                    train_parts.append(f"Kappa={float(train_metrics.get('kappa')):.4f}")
                if train_parts:
                    lines.append(f"- Training-set metrics: {'; '.join(train_parts)}")
                if test_metrics.get('roc_auc') is not None:
                    test_parts.append(f"ROC-AUC={float(test_metrics.get('roc_auc')):.4f}")
                if test_metrics.get('pr_auc') is not None:
                    test_parts.append(f"PR-AUC={float(test_metrics.get('pr_auc')):.4f}")
                if test_metrics.get('accuracy') is not None:
                    test_parts.append(f"Accuracy={float(test_metrics.get('accuracy')):.4f}")
                if test_metrics.get('balanced_accuracy') is not None:
                    test_parts.append(f"BalancedAcc={float(test_metrics.get('balanced_accuracy')):.4f}")
                if test_metrics.get('precision_pos') is not None:
                    test_parts.append(f"Precision(positive)={float(test_metrics.get('precision_pos')):.4f}")
                if test_metrics.get('recall_pos') is not None:
                    test_parts.append(f"Recall(positive)={float(test_metrics.get('recall_pos')):.4f}")
                if test_metrics.get('f1_pos') is not None:
                    test_parts.append(f"F1(positive)={float(test_metrics.get('f1_pos')):.4f}")
                if test_metrics.get('kappa') is not None:
                    test_parts.append(f"Kappa={float(test_metrics.get('kappa')):.4f}")
                if test_parts:
                    lines.append(f"- {holdout_label} metrics: {'; '.join(test_parts)}")
                train_cm = best_details.get('train_confusion_matrix')
                if isinstance(train_cm, list) and len(train_cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in train_cm):
                    lines.append(f"- Training-set confusion matrix ([[TN, FP], [FN, TP]]): {train_cm}")
                cm = best_details.get('confusion_matrix')
                if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
                    lines.append(f"- {holdout_label} confusion matrix ([[TN, FP], [FN, TP]]): {cm}")
                predict_cm = best_details.get('predict_confusion_matrix')
                if isinstance(predict_cm, list) and len(predict_cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in predict_cm):
                    lines.append(f"- Prediction-set confusion matrix ([[TN, FP], [FN, TP]]): {predict_cm}")
                artifacts = best_details.get("artifacts") if isinstance(best_details.get("artifacts"), dict) else {}
                if artifacts:
                    lines.append("")
                    lines.append("## Visualization Artifacts")
                    holdout_cm_label = "Validation confusion matrix" if str(holdout_label) == "Val" else "Test confusion matrix"
                    show_order = [
                        ("roc_curve_path", "ROC curve"),
                        ("predict_roc_curve_path", "Prediction-set ROC curve"),
                        ("train_confusion_matrix_path", "Training-set confusion matrix"),
                        ("confusion_matrix_path", holdout_cm_label),
                        ("predict_confusion_matrix_path", "Prediction-set confusion matrix"),
                        ("score_hist_path", "Prediction-score distribution"),
                        ("fit_gap_path", "Fit gap (AUC)"),
                        ("learning_curve_path", "Learning curve"),
                        ("feature_importance_path", "Feature importance"),
                        ("permutation_importance_path", "Permutation importance"),
                    ]
                    for key, label in show_order:
                        p = artifacts.get(key)
                        if not p:
                            continue
                        rel = self._relpath_from_reports(str(p))
                        if str(rel).lower().endswith(".png"):
                            lines.append(f"### {label}")
                            lines.append(f"![{label}]({rel})")
                            lines.append("")
                        else:
                            lines.append(f"- {label}: {rel}")
                    tuning_path = artifacts.get("tuning_cv_results_path")
                    if tuning_path:
                        rel = self._relpath_from_reports(str(tuning_path))
                        lines.append(f"- Tuning details: {rel}")
            top_imp = model_results.get('best_model_feature_importance')
            if isinstance(top_imp, list) and top_imp:
                lines.append('')
                lines.append('### Feature Importance (All)')
                for item in top_imp:
                    if not isinstance(item, dict):
                        continue
                    feature = item.get('feature')
                    if feature is None:
                        continue
                    try:
                        importance = float(item.get('importance', 0.0))
                    except Exception:
                        importance = 0.0
                    lines.append(f'- {feature}: {importance:.6f}')
            fit_diag = model_results.get('best_model_fit_diagnosis')
            if isinstance(fit_diag, dict):
                status = fit_diag.get('status')
                if status:
                    lines.append(f'- Fit diagnosis: {status}')
                recs = fit_diag.get('recommendations') or []
                if recs:
                    rec_text = '; '.join([str(r) for r in recs[:5]])
                    lines.append(f'- Recommendations: {rec_text}')
            if isinstance(best_details, dict):
                sec = best_details.get('secondary_tuning')
                if isinstance(sec, dict) and sec.get('activated'):
                    selected = bool(sec.get('selected'))
                    init = sec.get('initial') if isinstance(sec.get('initial'), dict) else {}
                    final = sec.get('final') if isinstance(sec.get('final'), dict) else {}
                    init_status = init.get('status')
                    final_status = final.get('status')
                    status_text = ''
                    if init_status or final_status:
                        status_text = f" ({init_status} -> {final_status})"
                    lines.append(f"- Iterative tuning: {'adopted' if selected else 'attempted but not adopted'}{status_text}")
                    if sec.get('best_cv_score') is not None:
                        try:
                            lines.append(f"- Iterative-tuning Ref-AUC: {float(sec.get('best_cv_score')):.4f}")
                        except Exception:
                            pass
                    best_params = sec.get('best_params')
                    if isinstance(best_params, dict) and best_params:
                        lines.append(f"- Iterative-tuning parameters: {best_params}")
                    reason = sec.get("selected_reason")
                    if reason:
                        lines.append(f"- Iterative-tuning decision: {reason}")
                    tuning_path = sec.get("tuning_cv_results_path")
                    if tuning_path:
                        try:
                            rel = self._relpath_from_reports(str(tuning_path))
                            lines.append(f"- Iterative-tuning details: {rel}")
                        except Exception:
                            pass
                    rounds = sec.get("rounds")
                    if isinstance(rounds, list) and rounds:
                        lines.append(f"- Iterative-tuning rounds: {len(rounds)}")
                        for r in rounds[:5]:
                            if not isinstance(r, dict):
                                continue
                            rd = r.get("round")
                            st = r.get("status")
                            val_auc = r.get("val_auc")
                            sel_auc = r.get("selection_auc")
                            tr_auc = r.get("train_roc_auc")
                            gap = r.get("auc_gap")
                            parts = []
                            if st is not None:
                                parts.append(f"status={st}")
                            if val_auc is not None:
                                try:
                                    parts.append(f"val_auc={float(val_auc):.4f}")
                                except Exception:
                                    pass
                            if sel_auc is not None:
                                try:
                                    parts.append(f"selection_auc={float(sel_auc):.4f}")
                                except Exception:
                                    pass
                            if tr_auc is not None:
                                try:
                                    parts.append(f"train_auc={float(tr_auc):.4f}")
                                except Exception:
                                    pass
                            if gap is not None:
                                try:
                                    parts.append(f"gap={float(gap):.4f}")
                                except Exception:
                                    pass
                            if parts:
                                lines.append(f"  - Round {rd}: " + ", ".join(parts))
            lines.append('')
        candidates = [k for k in model_results.keys() if k in self.models]
        if candidates:
            lines.append('## Metrics By Model')
            for name in candidates:
                res = model_results.get(name)
                lines.append(f"### {self._display_model_name(name)}")
                if not isinstance(res, dict) or res.get('error'):
                    err = ''
                    if isinstance(res, dict):
                        err = res.get('error') or res.get('message') or ''
                    lines.append(f'- Training failed: {err}'.strip())
                    lines.append('')
                    continue
                tuning = res.get("tuning") if isinstance(res.get("tuning"), dict) else {}
                sec = res.get("secondary_tuning") if isinstance(res.get("secondary_tuning"), dict) else {}
                if isinstance(sec, dict) and sec.get("activated") and sec.get("selected") and isinstance(sec.get("best_params"), dict) and sec.get("best_params"):
                    lines.append(f"- Best parameters (iterative tuning): {sec.get('best_params')}")
                elif isinstance(tuning, dict) and isinstance(tuning.get("best_params"), dict) and tuning.get("best_params"):
                    strategy = str(tuning.get("strategy") or "tuning")
                    lines.append(f"- Best parameters ({strategy}): {tuning.get('best_params')}")
                artifacts = res.get("artifacts") if isinstance(res.get("artifacts"), dict) else {}
                tuning_path = artifacts.get("tuning_cv_results_path") if isinstance(artifacts, dict) else None
                if tuning_path:
                    rel = self._relpath_from_reports(str(tuning_path))
                    lines.append(f"- Tuning details: {rel}")
                if res.get('mean_cv_score') is not None:
                    lines.append(f"- Cross-validation ROC-AUC (mean): {float(res.get('mean_cv_score')):.4f}")
                train_metrics = res.get('train_metrics') if isinstance(res.get('train_metrics'), dict) else {}
                train_roc = train_metrics.get('roc_auc')
                if train_roc is not None:
                    lines.append(f"- Training-set ROC-AUC: {float(train_roc):.4f}")
                train_pr = train_metrics.get('pr_auc')
                if train_pr is not None:
                    lines.append(f"- Training-set PR-AUC: {float(train_pr):.4f}")
                train_acc = train_metrics.get('accuracy')
                if train_acc is not None:
                    lines.append(f"- Training-set Accuracy: {float(train_acc):.4f}")
                train_bal = train_metrics.get('balanced_accuracy')
                if train_bal is not None:
                    lines.append(f"- Training-set Balanced-Accuracy: {float(train_bal):.4f}")
                train_prec = train_metrics.get('precision_pos')
                if train_prec is not None:
                    lines.append(f"- Training-set Precision(positive): {float(train_prec):.4f}")
                train_rec = train_metrics.get('recall_pos')
                if train_rec is not None:
                    lines.append(f"- Training-set Recall(positive): {float(train_rec):.4f}")
                train_f1p = train_metrics.get('f1_pos')
                if train_f1p is not None:
                    lines.append(f"- Training-set F1(positive): {float(train_f1p):.4f}")
                train_kap = train_metrics.get('kappa')
                if train_kap is not None:
                    lines.append(f"- Training-set Kappa: {float(train_kap):.4f}")
                train_cm = res.get('train_confusion_matrix')
                if isinstance(train_cm, list) and len(train_cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in train_cm):
                    lines.append(f"- Training-set confusion matrix ([[TN, FP], [FN, TP]]): {train_cm}")
                metrics = res.get('test_metrics') if isinstance(res.get('test_metrics'), dict) else {}
                roc = metrics.get('roc_auc')
                if roc is not None:
                    lines.append(f"- {holdout_label}ROC-AUC: {float(roc):.4f}")
                pr = metrics.get('pr_auc')
                if pr is not None:
                    lines.append(f"- {holdout_label}PR-AUC: {float(pr):.4f}")
                acc = metrics.get('accuracy')
                if acc is not None:
                    lines.append(f"- {holdout_label}Accuracy: {float(acc):.4f}")
                bal = metrics.get('balanced_accuracy')
                if bal is not None:
                    lines.append(f"- {holdout_label}Balanced-Accuracy: {float(bal):.4f}")
                prec = metrics.get('precision_pos')
                if prec is not None:
                    lines.append(f"- {holdout_label} Precision(positive): {float(prec):.4f}")
                rec = metrics.get('recall_pos')
                if rec is not None:
                    lines.append(f"- {holdout_label} Recall(positive): {float(rec):.4f}")
                f1p = metrics.get('f1_pos')
                if f1p is not None:
                    lines.append(f"- {holdout_label} F1(positive): {float(f1p):.4f}")
                kap = metrics.get('kappa')
                if kap is not None:
                    lines.append(f"- {holdout_label}Kappa: {float(kap):.4f}")
                cm = res.get('confusion_matrix')
                if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
                    lines.append(f"- Confusion matrix ([[TN, FP], [FN, TP]]): {cm}")
                sec = res.get('secondary_tuning')
                if isinstance(sec, dict) and sec.get('activated'):
                    selected = bool(sec.get('selected'))
                    init = sec.get('initial') if isinstance(sec.get('initial'), dict) else {}
                    final = sec.get('final') if isinstance(sec.get('final'), dict) else {}
                    init_status = init.get('status')
                    final_status = final.get('status')
                    status_text = ''
                    if init_status or final_status:
                        status_text = f" ({init_status} -> {final_status})"
                    lines.append(f"- Iterative tuning: {'adopted' if selected else 'attempted but not adopted'}{status_text}")
                    if sec.get('best_cv_score') is not None:
                        try:
                            lines.append(f"- Iterative-tuning Ref-AUC: {float(sec.get('best_cv_score')):.4f}")
                        except Exception:
                            pass
                    tuning_path = sec.get("tuning_cv_results_path")
                    if tuning_path:
                        try:
                            rel = self._relpath_from_reports(str(tuning_path))
                            lines.append(f"- Iterative-tuning details: {rel}")
                        except Exception:
                            pass
                lines.append('')
        return '\n'.join(lines).strip() + '\n'
    def _generate_predictions(self, X: pd.DataFrame, best_model: Any, selected_features: List[str], previous_predictions: Optional[Dict]=None, y: Optional[pd.Series]=None) -> Dict[str, Any]:
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X)[:, 1]
        else:
            probabilities = best_model.predict(X)
        predictions = best_model.predict(X)
        confidence = np.zeros_like(probabilities)
        if hasattr(best_model, 'predict_proba'):
            confidence = np.max(best_model.predict_proba(X), axis=1)
        else:
            confidence = np.ones_like(predictions) * 0.5
        self.logger.info(f'Running prediction for all {len(X)} samples')
        high_potential_threshold = np.percentile(probabilities, 90)
        high_potential_indices = np.where(probabilities >= high_potential_threshold)[0].tolist()
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importances = np.asarray(best_model.feature_importances_, dtype=float)
            indices = np.argsort(importances)[::-1]
            feature_importance = [{'feature': str(selected_features[i]), 'importance': float(importances[i])} for i in indices]
        return {'probabilities': probabilities.tolist(), 'predictions': predictions.tolist(), 'confidence': confidence.tolist(), 'high_potential_threshold': float(high_potential_threshold), 'high_potential_indices': high_potential_indices, 'high_potential_count': len(high_potential_indices), 'feature_importance': feature_importance}
    def _generate_summary(self, model_results: Dict, predictions: Dict) -> str:
        summary = 'Predictive-model analysis results:\n'
        if 'best_model_name' in model_results:
            holdout_label = "Test set"
            split = model_results.get("train_test_split")
            if isinstance(split, dict):
                role = str(split.get("holdout_role") or "").strip().lower()
                if role == "validation":
                    holdout_label = "Validation set"
            best_model_name = model_results['best_model_name']
            best_model_display = self._display_model_name(best_model_name) if best_model_name else None
            metric = model_results.get('best_model_metric', 'roc_auc')
            metric_name = 'Cross-validation ROC-AUC' if 'roc' in str(metric).lower() else 'Cross-validation score'
            best_details = model_results.get(best_model_name)
            best_score = None
            if isinstance(best_details, dict):
                metrics = best_details.get('test_metrics') if isinstance(best_details.get('test_metrics'), dict) else {}
                best_score = metrics.get('roc_auc')
            if best_score is None:
                best_score = model_results.get('best_model_score')
            if best_score is None:
                summary += f'- Best model: {best_model_display or best_model_name}, {metric_name}: NA\n'
            else:
                summary += f'- Best model: {best_model_display or best_model_name}, {metric_name}: {float(best_score):.4f}\n'
            if isinstance(best_details, dict):
                metrics = best_details.get('test_metrics') if isinstance(best_details.get('test_metrics'), dict) else {}
                roc = metrics.get('roc_auc')
                pr = metrics.get('pr_auc')
                bal = metrics.get('balanced_accuracy')
                f1p = metrics.get('f1_pos')
                acc = metrics.get('accuracy')
                prec = metrics.get('precision_pos')
                rec = metrics.get('recall_pos')
                kap = metrics.get('kappa')
                parts = []
                if roc is not None:
                    parts.append(f'{holdout_label}ROC-AUC={float(roc):.4f}')
                if pr is not None:
                    parts.append(f'PR-AUC={float(pr):.4f}')
                if acc is not None:
                    parts.append(f'Accuracy={float(acc):.4f}')
                if bal is not None:
                    parts.append(f'BalancedAcc={float(bal):.4f}')
                if prec is not None:
                    parts.append(f'Precision(positive)={float(prec):.4f}')
                if rec is not None:
                    parts.append(f'Recall(positive)={float(rec):.4f}')
                if f1p is not None:
                    parts.append(f'F1(positive)={float(f1p):.4f}')
                if kap is not None:
                    parts.append(f'Kappa={float(kap):.4f}')
                if parts:
                    summary += f"- {holdout_label} metrics: {'; '.join(parts)}\n"
                train_cm = best_details.get('train_confusion_matrix')
                if isinstance(train_cm, list) and len(train_cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in train_cm):
                    summary += f'- Training-set confusion matrix ([[TN, FP], [FN, TP]]): {train_cm}\n'
                cm = best_details.get('confusion_matrix')
                if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
                    summary += f'- {holdout_label} confusion matrix ([[TN, FP], [FN, TP]]): {cm}\n'
                predict_cm = best_details.get('predict_confusion_matrix')
                if isinstance(predict_cm, list) and len(predict_cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in predict_cm):
                    summary += f'- Prediction-set confusion matrix ([[TN, FP], [FN, TP]]): {predict_cm}\n'
            fit_diag = model_results.get('best_model_fit_diagnosis')
            if isinstance(fit_diag, dict) and fit_diag.get('status'):
                status = fit_diag.get('status')
                summary += f'- Fit diagnosis: {status}\n'
                recs = fit_diag.get('recommendations') or []
                if recs:
                    rec_text = '; '.join([str(r) for r in recs[:3]])
                    summary += f'- Tuning recommendations: {rec_text}\n'
            tuning = None
            if best_model_name in model_results and isinstance(model_results.get(best_model_name), dict):
                tuning = model_results[best_model_name].get('tuning')
            sec = None
            if best_model_name in model_results and isinstance(model_results.get(best_model_name), dict):
                sec = model_results[best_model_name].get('secondary_tuning')
            if isinstance(sec, dict) and sec.get('activated') and sec.get('selected') and isinstance(sec.get('best_params'), dict) and sec.get('best_params'):
                summary += f"- Best parameters (iterative tuning): {sec.get('best_params')}\n"
            elif isinstance(tuning, dict) and tuning.get('best_params'):
                summary += f"- Best parameters: {tuning.get('best_params')}\n"
        else:
            summary += '- No suitable best model was found\n'
        high_potential_count = predictions.get('high_potential_count', 0)
        total_samples = len(predictions.get('predictions', []))
        if total_samples > 0:
            summary += f'- Predicted {high_potential_count} high-potential areas ({high_potential_count / total_samples:.2%} of all samples)\n'
        else:
            summary += '- No valid prediction samples are available\n'
        feature_importance = predictions.get('feature_importance', {})
        if feature_importance:
            if isinstance(feature_importance, dict):
                top_features = []
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features:
                    top_features.append(f'{feature}({importance:.3f})')
                top_features_str = ', '.join(top_features)
            else:
                top_features_str = ', '.join([f"{item['feature']}({item['importance']:.3f})" for item in feature_importance])
            summary += f'- Most important predictive features: {top_features_str}\n'
        return summary


MINERAL_ELEMENTS = ['Al2O3', 'Ba', 'Be', 'La', 'Na', 'Nb', 'Th', 'U', 'Y', 'Zr']
OUTPUT_DIR = os.path.join(".", "output", "qe")
PLOT_TITLE = "Geochemical Anomaly Distribution Map"


def ensure_2d_array(arr, var_name="array"):
    if arr is None:
        raise ValueError(f"{var_name} cannot be empty")
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
        logger.debug(f"{var_name} converted to a NumPy array with shape: {arr.shape}")
    if arr.ndim == 1:
        new_shape = (len(arr), 1)
        arr_2d = arr.reshape(new_shape)
        logger.warning(f"{var_name} is one-dimensional and has been reshaped to 2D: {new_shape}")
        return arr_2d
    if arr.ndim == 2:
        logger.debug(f"{var_name} is already 2D with shape: {arr.shape}")
        return arr
    raise ValueError(f"{var_name} has an invalid number of dimensions: expected 1 or 2, got {arr.ndim}")


def get_custom_cmap():
    colors = [
        (0.0, "#3B07B0"),
        (0.2, "#1F8EF0"),
        (0.4, "#32CD32"),
        (0.6, "#F9F908"),
        (0.8, "#FF7C00"),
        (1.0, "#FF0000"),
    ]
    return LinearSegmentedColormap.from_list("custom_bgyocr", colors)


def get_element_component_cmap():
    x_positions = np.linspace(0.0, 1.0, 6)
    colors = ["#334D7F", "#00FFFF", "#8FBF8F", "#FFFFE0", "#D97070", "#992E2E"]
    return LinearSegmentedColormap.from_list("element_component_cmap", list(zip(x_positions, colors)))


def get_enhanced_cmap():
    x_positions = np.linspace(0.0, 1.0, 8)
    colors = ["#0A3D91", "#1E88E5", "#64B5F6", "#A5D6A7", "#FFEE58", "#FFB74D", "#EF5350", "#B71C1C"]
    return LinearSegmentedColormap.from_list("enhanced_cmap", list(zip(x_positions, colors)))


def apply_clr_to_all_elements(X_processed, elements, epsilon=1e-6):
    X_processed = ensure_2d_array(X_processed, "X_processed (all-element CLR)")
    safe_values = np.where(X_processed > 0, X_processed, epsilon).astype(np.float64)
    log_values = np.log(safe_values)
    geom_log_mean = np.mean(log_values, axis=1, keepdims=True)
    transformed = log_values - geom_log_mean
    return transformed


def apply_log10_to_all_elements(X_processed, elements, epsilon=1e-6):
    X_processed = ensure_2d_array(X_processed, "X_processed (all-element log10)")
    safe_values = np.where(X_processed > 0, X_processed, epsilon).astype(np.float64)
    return np.log10(safe_values)


def load_geochem_data_from_df(
    df,
    elements,
    ore_elements=None,
    log_transform=False,
    log_mode="log10",
    normalize=False,
    save_preprocessing=True,
    output_dir="preprocessing",
):
    os.makedirs(output_dir, exist_ok=True)
    data = df.copy()
    print(f"Data loaded successfully. Sample count: {len(data)}")
    missing = [e for e in elements if e not in data.columns]
    if missing:
        raise ValueError(f"Missing element columns: {missing}")
    X_raw = data[elements].values.copy()
    X_raw = ensure_2d_array(X_raw, "raw data X_raw")
    X_processed = X_raw.copy()
    if log_transform:
        mode = str(log_mode or "log10").strip().lower()
        if mode == "clr":
            X_processed = apply_clr_to_all_elements(X_processed, elements, epsilon=1e-6)
            print("Applied unconditional CLR transformation to all selected elements")
        else:
            X_processed = apply_log10_to_all_elements(X_processed, elements, epsilon=1e-6)
            print("Applied unconditional log10 transformation to all selected elements")
        if save_preprocessing:
            np.save(f"{output_dir}/log_epsilon.npy", np.array([1e-6]))
    X_processed = np.nan_to_num(X_processed, nan=np.nanmedian(X_processed))
    scaler = None
    if normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = ensure_2d_array(X_scaled, "normalized X_scaled")
    else:
        X_scaled = ensure_2d_array(X_processed, "SOM input X_scaled (reusing preprocessing results)")
    if save_preprocessing:
        try:
            if scaler is not None:
                joblib.dump(scaler, f"{output_dir}/som_scaler.pkl")
        except Exception:
            pass
        try:
            elem_df = pd.DataFrame({"Element Name": elements})
            _localize_dataframe_headers(elem_df).to_csv(f"{output_dir}/main_elements_list.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pass
    print(f"Output X_scaled shape from load_geochem_data_from_df: {X_scaled.shape}")
    return X_scaled, X_raw, data, scaler, elements


def train_som(X, map_size=(22, 22), sigma=5.0, lr=0.5, iterations=1000):
    if MiniSom is None:
        raise ModuleNotFoundError("Missing dependency `minisom`. Please install it first: python -m pip install minisom")
    X = ensure_2d_array(X, "SOM training data X")
    n_features = X.shape[1]
    print(f"Training SOM {map_size[0]}x{map_size[1]} with input feature dimension {n_features} for {iterations} iterations...")
    som = MiniSom(
        map_size[0],
        map_size[1],
        n_features,
        sigma=sigma,
        learning_rate=lr,
        random_seed=42,
        topology='hexagonal',
        neighborhood_function='gaussian'
    )
    if X.ndim != 2:
        raise ValueError(f"SOM training data must be a 2D array, but got {X.ndim} dimensions")
    som.train_random(X, iterations, verbose=True)
    return som


def calculate_topographic_error(som, X):
    X = ensure_2d_array(X, "topographic-error input X")
    topo_error = 0.0
    n_samples = X.shape[0]
    map_shape = som.get_weights().shape[:2]
    map_rows, map_cols = map_shape
    for x in X:
        try:
            if x.ndim == 2:
                x = x.flatten()
            distances = []
            for i in range(map_rows):
                for j in range(map_cols):
                    w = som.get_weights()[i, j]
                    distances.append((np.linalg.norm(x - w), i, j))
            distances.sort()
            bmu1 = (distances[0][1], distances[0][2])
            bmu2 = (distances[1][1], distances[1][2])
            i, j = bmu1
            if i % 2 == 0:
                neighbors = [
                    (i - 1, j - 1), (i - 1, j),
                    (i, j - 1), (i, j + 1),
                    (i + 1, j - 1), (i + 1, j)
                ]
            else:
                neighbors = [
                    (i - 1, j), (i - 1, j + 1),
                    (i, j - 1), (i, j + 1),
                    (i + 1, j), (i + 1, j + 1)
                ]
            valid_neighbors = []
            for ni, nj in neighbors:
                if 0 <= ni < map_rows and 0 <= nj < map_cols:
                    valid_neighbors.append((ni, nj))
            if bmu2 not in valid_neighbors:
                topo_error += 1.0
        except Exception as e:
            print(f"Sample-processing error: {str(e)}")
            continue
    return topo_error / n_samples


def calculate_u_matrix(som):
    map_rows, map_cols = som.get_weights().shape[:2]
    weights = som.get_weights()
    u_matrix = np.zeros((map_rows, map_cols))
    hex_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
    for i in range(map_rows):
        for j in range(map_cols):
            neighbor_distances = []
            for di, dj in hex_neighbors:
                ni, nj = i + di, j + dj
                if i % 2 == 0:
                    if dj > 0:
                        nj -= 1
                else:
                    if dj < 0:
                        nj += 1
                if 0 <= ni < map_rows and 0 <= nj < map_cols:
                    dist = np.linalg.norm(weights[i, j] - weights[ni, nj])
                    neighbor_distances.append(dist)
            if neighbor_distances:
                u_matrix[i, j] = np.median(neighbor_distances)
            else:
                u_matrix[i, j] = 0.0
    u_matrix = np.log1p(u_matrix)
    min_val = np.min(u_matrix)
    max_val = np.max(u_matrix)
    if max_val > min_val:
        u_matrix = (u_matrix - min_val) / (max_val - min_val)
    else:
        u_matrix = np.zeros_like(u_matrix)
    return u_matrix


def _get_som_canvas_size(row_count, col_count):
    return max(12, col_count * 0.5), max(10, row_count * 0.5)


def _save_fixed_canvas_figure(fig, output_path, dpi=300):
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none')
    try:
        from PIL import Image
        expected_size = (
            int(round(float(fig.get_figwidth()) * float(dpi))),
            int(round(float(fig.get_figheight()) * float(dpi))),
        )
        with Image.open(output_path).convert("RGBA") as saved_img:
            if saved_img.size != expected_size:
                bg_color = saved_img.getpixel((0, 0))
                canvas = Image.new("RGBA", expected_size, bg_color)
                offset_x = max((expected_size[0] - saved_img.size[0]) // 2, 0)
                offset_y = max((expected_size[1] - saved_img.size[1]) // 2, 0)
                canvas.paste(saved_img, (offset_x, offset_y), saved_img)
                canvas.save(output_path)
    except Exception:
        pass


def plot_u_matrix(som, u_matrix, output_dir="Sample_Cluster_Results"):
    os.makedirs(output_dir, exist_ok=True)
    plt.close('all')
    map_rows, map_cols = som.get_weights().shape[:2]
    radius = 0.5
    hex_width = np.sqrt(3) * radius * 1.05
    hex_height = 1.5 * radius * 1.05
    fig_width, fig_height = _get_som_canvas_size(map_rows, map_cols)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')
    cmap = get_element_component_cmap()
    cmap.set_bad(color='lightgray', alpha=0.5)
    umin = float(np.nanmin(u_matrix))
    umax = float(np.nanmax(u_matrix))
    if np.isclose(umin, umax):
        umin -= 0.1
        umax += 0.1
    patches = []
    values_list = []
    for i in range(map_rows):
        for j in range(map_cols):
            x = j * hex_width + (i % 2) * hex_width / 2
            y = i * hex_height
            value = u_matrix[i, j]
            values_list.append(value)
            hexagon = RegularPolygon((x, y), 6, radius=radius, orientation=0, edgecolor='white', linewidth=0.5)
            if np.isnan(value):
                hexagon.set_facecolor('lightgray')
            else:
                norm_val = (float(value) - umin) / (umax - umin)
                norm_val = np.clip(norm_val, 0, 1)
                hexagon.set_facecolor(cmap(norm_val))
            patches.append(hexagon)
    pc = mpl.collections.PatchCollection(patches, alpha=0.9, zorder=2, cmap=cmap)
    pc.set_array(np.array(values_list))
    pc.set_clim(umin, umax)
    ax.add_collection(pc)
    x_min = -hex_width / 2
    x_max = (map_cols - 1) * hex_width + (1 if map_rows % 2 else 0) * hex_width / 2 + hex_width * 1.2
    y_min = -radius
    y_max = (map_rows - 1) * hex_height + radius * 1.35
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label("U-Matrix Distance", fontsize=27, fontname="Times New Roman")
    cbar.ax.tick_params(labelsize=27)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")
    ax.set_title("SOM U-Matrix", fontsize=27, pad=8, fontname="Times New Roman")
    output_path = os.path.join(output_dir, "U_Matrix_Enhanced.png")
    _save_fixed_canvas_figure(fig, output_path)
    plt.close(fig)
    print(f"Saved SOM U-matrix figure: {output_path}")


def apply_kmeans_to_samples(som, X, sample_ids, n_clusters=5, verbose=True):
    X = ensure_2d_array(X, "K-means input X")
    bmus = np.array([som.winner(x) for x in X])
    weights = som.get_weights().reshape(-1, som.get_weights().shape[2])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(weights)
    grid_labels = kmeans.labels_.reshape(som.get_weights().shape[:2])
    sample_labels = np.array([grid_labels[bmu[0], bmu[1]] for bmu in bmus])
    sample_clusters = {}
    for i in range(n_clusters):
        sample_clusters[i] = [sample_ids[j] for j, label in enumerate(sample_labels) if label == i]
    sample_weights = np.array([som.get_weights()[bmu[0], bmu[1]] for bmu in bmus])
    try:
        sil_score = silhouette_score(sample_weights, sample_labels)
    except Exception:
        sil_score = float('nan')
        print("Warning: failed to compute the silhouette score")
    try:
        ch_score = calinski_harabasz_score(sample_weights, sample_labels)
    except Exception:
        ch_score = float('nan')
        print("Warning: failed to compute the Calinski-Harabasz index")
    try:
        db_score = davies_bouldin_score(sample_weights, sample_labels)
    except Exception:
        db_score = float('nan')
        print("Warning: failed to compute the Davies-Bouldin index")
    try:
        inertia = kmeans.inertia_
    except Exception:
        inertia = float('nan')
    if bool(verbose):
        print("\nClustering evaluation metrics:")
        print(f"  - Silhouette Score: {sil_score:.4f} (closer to 1 is better)")
        print(f"  - Calinski-Harabasz Index: {ch_score:.2f} (higher is better)")
        print(f"  - Davies-Bouldin Index: {db_score:.4f} (closer to 0 is better)")
        print("\nSample clustering results:")
        for cluster, samples in sorted(sample_clusters.items()):
            print(f"Cluster {cluster}: {len(samples)} samples")
    return grid_labels, sample_labels, sample_clusters, bmus, {
        'silhouette': sil_score,
        'calinski_harabasz': ch_score,
        'davies_bouldin': db_score,
        'inertia': inertia
    }


def optimize_som_parameters(X, sample_ids, map_size, param_grid=None, output_dir=None):
    if MiniSom is None:
        raise ModuleNotFoundError("Missing dependency `minisom`. Please install it first: python -m pip install minisom")
    X = ensure_2d_array(X, "parameter-tuning input X")
    print(f"Parameter tuning - input data shape: {X.shape}")
    grid = param_grid if isinstance(param_grid, dict) else {}
    sigma_list = grid.get("sigma")
    lr_list = grid.get("lr")
    iters_list = grid.get("iterations")
    min_side = float(max(1, min(int(map_size[0]), int(map_size[1]))))
    sigma_candidates = [
        max(0.5, min_side * 0.15),
        max(0.7, min_side * 0.25),
        max(0.9, min_side * 0.35),
        max(1.2, min_side * 0.5),
    ]
    sigma_list_default = sorted({float(round(s, 3)) for s in sigma_candidates})
    lr_list_default = [0.03, 0.08, 0.2, 0.5]
    def _default_iters(n_tune: int) -> list[int]:
        base = int(min(50000, max(3000, 6 * int(n_tune))))
        cands = [int(round(base * 0.7)), int(base), int(round(base * 1.4))]
        cands = [int(min(80000, max(2000, x))) for x in cands]
        return sorted({int(x) for x in cands if int(x) > 0})
    if not isinstance(sigma_list, (list, tuple)) or not sigma_list:
        sigma_list = sigma_list_default
    if not isinstance(lr_list, (list, tuple)) or not lr_list:
        lr_list = lr_list_default
    qe_weight = float(grid.get("qe_weight", 0.5))
    te_weight = float(grid.get("te_weight", 0.5))
    if qe_weight < 0:
        qe_weight = 0.0
    if te_weight < 0:
        te_weight = 0.0
    if qe_weight + te_weight <= 0:
        qe_weight, te_weight = 0.5, 0.5
    seed = 42
    X_tune = X
    print(f"Parameter tuning - using all samples for tuning: {X_tune.shape}")
    if not isinstance(iters_list, (list, tuple)) or not iters_list:
        n_tune = int(X_tune.shape[0])
        iters_list = _default_iters(n_tune)
    try:
        sigma_list = [float(x) for x in sigma_list]
    except Exception:
        sigma_list = sigma_list_default
    try:
        lr_list = [float(x) for x in lr_list]
    except Exception:
        lr_list = lr_list_default
    try:
        iters_list = [int(float(x)) for x in iters_list]
    except Exception:
        n_tune = int(X_tune.shape[0])
        iters_list = _default_iters(n_tune)
    iters_list = [int(x) for x in iters_list if int(x) > 0]
    if not iters_list:
        n_tune = int(X_tune.shape[0])
        iters_list = _default_iters(n_tune)

    def _eval_combo(*, sigma: float, lr: float, iters: int):
        som = MiniSom(
            map_size[0],
            map_size[1],
            X_tune.shape[1],
            sigma=float(sigma),
            learning_rate=float(lr),
            topology="hexagonal",
            neighborhood_function="gaussian",
            random_seed=int(seed),
        )
        som.train_random(X_tune, int(iters), verbose=False)
        qe = float(np.mean([np.linalg.norm(x - som.get_weights()[som.winner(x)]) for x in X_tune]))
        te = float(calculate_topographic_error(som, X_tune))
        return qe, te

    def _normalize_scores(qe_arr: np.ndarray, te_arr: np.ndarray) -> np.ndarray:
        qe_min, qe_max = float(np.min(qe_arr)), float(np.max(qe_arr))
        te_min, te_max = float(np.min(te_arr)), float(np.max(te_arr))
        qe_norm = (qe_arr - qe_min) / (qe_max - qe_min) if qe_max > qe_min else np.zeros_like(qe_arr, dtype=float)
        te_norm = (te_arr - te_min) / (te_max - te_min) if te_max > te_min else np.zeros_like(te_arr, dtype=float)
        return qe_norm * float(qe_weight) + te_norm * float(te_weight)

    stage1 = list(product(sigma_list, lr_list, iters_list))
    all_results = []
    print(f"Parameter search: computing QE/TE for parameter combinations ({len(stage1)} groups; seed={int(seed)}; weights: QE={qe_weight}, TE={te_weight})...")
    for i, (sigma, lr, iters) in enumerate(stage1):
        try:
            qe, te = _eval_combo(sigma=float(sigma), lr=float(lr), iters=int(iters))
            all_results.append({"sigma": float(sigma), "lr": float(lr), "iters": int(iters), "qe": qe, "te": te, "stage": 1})
            print(f"Parameter set {i + 1}/{len(stage1)}: sigma={sigma}, lr={lr}, iters={iters}, QE={qe:.6f}, TE={te:.4f}")
        except Exception as e:
            all_results.append({"sigma": float(sigma), "lr": float(lr), "iters": int(iters), "qe": float("nan"), "te": float("nan"), "stage": 1, "error": str(e)})
            print(f"Parameter set {i + 1} failed: {str(e)}. Skipping this combination")

    def _valid_mask_from_results(results: list[dict]) -> np.ndarray:
        qe = np.asarray([r.get("qe") for r in results], dtype=float)
        te = np.asarray([r.get("te") for r in results], dtype=float)
        return np.isfinite(qe) & np.isfinite(te)

    valid_mask = _valid_mask_from_results(all_results)
    if not bool(np.any(valid_mask)):
        raise ValueError("All parameter combinations are invalid; parameter optimization cannot continue")
    qe_valid = np.asarray([r.get("qe") for r in all_results], dtype=float)[valid_mask]
    te_valid = np.asarray([r.get("te") for r in all_results], dtype=float)[valid_mask]
    score_valid = _normalize_scores(qe_valid, te_valid)
    valid_indices = np.flatnonzero(valid_mask)
    for j, idx in enumerate(valid_indices.tolist()):
        all_results[idx]["score"] = float(score_valid[j])
    for r in all_results:
        if "score" not in r:
            r["score"] = float("nan")
    best_pos = int(np.argmin(score_valid))
    best_idx = int(valid_indices[best_pos])
    best = all_results[best_idx]
    best_score = float(score_valid[best_pos])
    tuning_artifacts = {}
    if output_dir:
        try:
            out_dir = os.path.abspath(str(output_dir))
            os.makedirs(out_dir, exist_ok=True)
            tune_csv_path = os.path.join(out_dir, "som_qe_te_tuning_results.csv")
            tune_json_path = os.path.join(out_dir, "som_qe_te_tuning_results.json")
            tune_scatter_path = os.path.join(out_dir, "som_qe_te_tuning_scatter.png")
            tune_rank_path = os.path.join(out_dir, "som_qe_te_tuning_rank.png")
            tune_df = pd.DataFrame(all_results)
            base_cols = ["sigma", "lr", "iters", "qe", "te", "score", "stage", "error"]
            keep_cols = [c for c in base_cols if c in tune_df.columns]
            extra_cols = [c for c in tune_df.columns if c not in keep_cols]
            tune_df = tune_df[keep_cols + extra_cols]
            tune_df_raw = tune_df.copy()
            tune_df = tune_df.sort_values(by=["score", "qe", "te"], ascending=[True, True, True], na_position="last").reset_index(drop=True)
            tune_df.to_csv(tune_csv_path, index=False, encoding="utf-8-sig")
            with open(tune_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "weights": {"qe": float(qe_weight), "te": float(te_weight)},
                        "map_size": [int(map_size[0]), int(map_size[1])],
                        "param_grid": {
                            "sigma": [float(x) for x in sigma_list],
                            "lr": [float(x) for x in lr_list],
                            "iterations": [int(x) for x in iters_list],
                        },
                        "best": {
                            "sigma": float(best.get("sigma")),
                            "lr": float(best.get("lr")),
                            "iters_tune": int(best.get("iters")),
                            "qe": float(best.get("qe")),
                            "te": float(best.get("te")),
                            "score": float(best.get("score")),
                        },
                        "results": tune_df.to_dict(orient="records"),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            valid_df = tune_df[np.isfinite(pd.to_numeric(tune_df.get("score"), errors="coerce"))].copy()
            if not valid_df.empty:
                qe_vals = pd.to_numeric(valid_df["qe"], errors="coerce").to_numpy(dtype=float)
                te_vals = pd.to_numeric(valid_df["te"], errors="coerce").to_numpy(dtype=float)
                score_vals = pd.to_numeric(valid_df["score"], errors="coerce").to_numpy(dtype=float)
                _setup_matplotlib_output_style(plt)
                fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
                title_fontsize = 16
                other_fontsize = 16
                sc = ax.scatter(qe_vals, te_vals, c=score_vals, cmap="viridis_r", s=42, alpha=0.9)
                ax.set_xlabel("QE", fontsize=other_fontsize)
                ax.set_ylabel("TE", fontsize=other_fontsize)
                ax.set_title("SOM QE/TE Tuning Search", fontsize=title_fontsize)
                ax.tick_params(axis="both", labelsize=other_fontsize)
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("Comprehensive Score", fontsize=other_fontsize)
                cbar.ax.tick_params(labelsize=other_fontsize)
                bq = float(best.get("qe"))
                bt = float(best.get("te"))
                ax.scatter([bq], [bt], c=["red"], s=96, marker="*", edgecolors="black", linewidths=0.8, label="Best Parameters")
                ax.legend(loc="upper right", fontsize=other_fontsize, handletextpad=0.35, handlelength=1.0, markerscale=0.85)
                fig.tight_layout()
                fig.savefig(tune_scatter_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                rank_df = tune_df_raw[np.isfinite(pd.to_numeric(tune_df_raw.get("score"), errors="coerce"))].copy()
                rank_scores = pd.to_numeric(rank_df["score"], errors="coerce").to_numpy(dtype=float)
                rank_x = rank_df.index.to_numpy(dtype=int) + 1
                fig2, ax2 = plt.subplots(figsize=(11, 6), dpi=300)
                bars = ax2.bar(rank_x, rank_scores, color="#4C78A8", alpha=0.9, label="Parameter Combination Score")
                best_rank_idx = int(best_idx) + 1
                best_bar_pos = np.where(rank_x == best_rank_idx)[0]
                if best_bar_pos.size > 0:
                    bars[int(best_bar_pos[0])].set_color("#E45756")
                step = int(max(1, round(len(rank_df) / 12)))
                ax2.set_xticks(rank_x[::step])
                ax2.set_xlim(float(rank_x.min()) - 0.5, float(rank_x.max()) + 0.5)
                ax2.set_xlabel("Combination Index", fontsize=16)
                ax2.set_ylabel("Score", fontsize=16)
                ax2.set_title("QE/TE Tuning Scores", fontsize=16)
                ax2.tick_params(axis="both", labelsize=16)
                from matplotlib.patches import Patch
                legend_handles = [
                    Patch(facecolor="#4C78A8", alpha=0.9, label="Parameter Combination Score"),
                    Patch(facecolor="#E45756", alpha=0.9, label="Best Parameters"),
                ]
                ax2.legend(handles=legend_handles, loc="upper left", frameon=True, edgecolor="black", fontsize=16)
                fig2.tight_layout()
                fig2.savefig(tune_rank_path, dpi=300, bbox_inches="tight")
                plt.close(fig2)
            tuning_artifacts = {
                "tuning_results_csv": os.path.abspath(tune_csv_path),
                "tuning_results_json": os.path.abspath(tune_json_path),
                "tuning_scatter_plot": os.path.abspath(tune_scatter_path) if os.path.exists(tune_scatter_path) else "",
                "tuning_rank_plot": os.path.abspath(tune_rank_path) if os.path.exists(tune_rank_path) else "",
            }
            print(f"Saved parameter-search table: {tune_csv_path}")
            print(f"Saved parameter-search JSON: {tune_json_path}")
            if tuning_artifacts.get("tuning_scatter_plot"):
                print(f"Saved parameter-search scatter plot: {tuning_artifacts['tuning_scatter_plot']}")
            if tuning_artifacts.get("tuning_rank_plot"):
                print(f"Saved top-combination ranking plot: {tuning_artifacts['tuning_rank_plot']}")
        except Exception as e:
            print(f"Failed to save parameter-search visualizations: {e}")
    final_iters = grid.get("final_iterations")
    if final_iters is None:
        n_full = int(X.shape[0])
        final_iters = int(min(200000, max(20000, 2 * n_full)))
    try:
        final_iters = int(float(final_iters))
    except Exception:
        n_full = int(X.shape[0])
        final_iters = int(min(200000, max(20000, 2 * n_full)))
    if int(final_iters) <= 0:
        n_full = int(X.shape[0])
        final_iters = int(min(200000, max(20000, 2 * n_full)))
    best_sigma = float(best["sigma"])
    best_lr = float(best["lr"])
    print("\nParameter tuning completed (tuning stage).")
    print(f"Best tuning result: sigma={best_sigma}, lr={best_lr}, iters(tune)={int(best['iters'])}, QE={float(best['qe']):.6f}, TE={float(best['te']):.4f}")
    print(f"The SOM will now be retrained once on the full dataset: iters(full)={int(final_iters)}")
    best_som = MiniSom(
        map_size[0],
        map_size[1],
        X.shape[1],
        sigma=float(best_sigma),
        learning_rate=float(best_lr),
        topology="hexagonal",
        neighborhood_function="gaussian",
        random_seed=42,
    )
    best_som.train_random(X, int(final_iters), verbose=False)
    final_qe = float(np.mean([np.linalg.norm(x - best_som.get_weights()[best_som.winner(x)]) for x in X]))
    final_te = float(calculate_topographic_error(best_som, X))
    best_params = {
        "sigma": float(best_sigma),
        "lr": float(best_lr),
        "iterations": int(final_iters),
        "map_size": map_size,
        "best_qe": final_qe,
        "best_te": final_te,
        "best_comprehensive_score": best_score,
        "tuning_meta": {
            "tune_shape": tuple(X_tune.shape),
            "full_shape": tuple(X.shape),
            "seed": int(seed),
            "weights": {"qe": float(qe_weight), "te": float(te_weight)},
            "stages": {"stage1": int(len(stage1)), "total": int(len(all_results))},
        },
        "tuning_artifacts": tuning_artifacts,
    }
    if best_som is None:
        print("\nWarning: all parameter combinations failed. Falling back to default parameters")
        default_sigma = 9.0
        default_lr = 0.4
        default_iters = int(min(200000, max(20000, 2 * len(X))))
        best_som = MiniSom(
            map_size[0], map_size[1], X.shape[1],
            sigma=default_sigma,
            learning_rate=default_lr,
            topology='hexagonal',
            neighborhood_function='gaussian',
            random_seed=42
        )
        best_som.train_random(X, default_iters, verbose=True)
        default_qe = np.mean([np.linalg.norm(x - best_som.get_weights()[best_som.winner(x)]) for x in X])
        default_te = calculate_topographic_error(best_som, X)
        best_params = {
            'sigma': default_sigma,
            'lr': default_lr,
            'iterations': default_iters,
            'map_size': map_size,
            'best_qe': default_qe,
            'best_te': default_te,
            'best_comprehensive_score': np.nan,
            'note': 'Default parameters were used because all candidate parameter groups failed'
        }
        print(f"Default-parameter QE: {default_qe:.6f}, TE: {default_te:.4f}")
    print("\nParameter tuning completed.")
    print(f"Best parameters: {best_params}")
    return best_som, best_params


def suggest_best_k(som, X, sample_ids, k_range=range(2, 11), output_dir="Sample_Cluster_Results"):
    rows, cols = som.get_weights().shape[:2]
    max_k = int(rows * cols)
    k_values = []
    for k in list(k_range):
        try:
            k_int = int(k)
        except Exception:
            continue
        if 2 <= k_int <= max_k:
            k_values.append(k_int)
    if not k_values:
        raise ValueError(f"No valid clustering-number range is available: map_size={rows}x{cols}, max_k={max_k}")
    k_range = k_values
    inertias = []
    sil_scores = []
    db_scores = []
    ch_scores = []
    for k in k_range:
        _, _, _, _, metrics = apply_kmeans_to_samples(som, X, sample_ids, n_clusters=k, verbose=False)
        print(
            f"k={k} | Sil={float(metrics['silhouette']):.4f} | "
            f"CH={float(metrics['calinski_harabasz']):.2f} | DB={float(metrics['davies_bouldin']):.4f}"
        )
        inertias.append(metrics['inertia'])
        sil_scores.append(metrics['silhouette'])
        db_scores.append(metrics['davies_bouldin'])
        ch_scores.append(metrics['calinski_harabasz'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    ax1.plot(k_range, inertias, 'o-', color='blue')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)
    drop_rates = [np.nan]
    for i in range(1, len(inertias)):
        prev = float(inertias[i - 1])
        curr = float(inertias[i])
        if np.isclose(prev, 0.0):
            drop_rates.append(np.nan)
        else:
            drop_rates.append((prev - curr) / prev)
    elbow_k = None
    for i in range(1, len(inertias)):
        if np.isfinite(drop_rates[i]) and float(drop_rates[i]) < 0.1:
            elbow_k = k_range[i]
            break
    valid_k_for_drop = [k_range[i] for i in range(1, len(k_range)) if np.isfinite(drop_rates[i])]
    valid_drop_rates = [float(drop_rates[i]) for i in range(1, len(k_range)) if np.isfinite(drop_rates[i])]
    ax2.plot(valid_k_for_drop, valid_drop_rates, 'o-', color='purple')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Drop Ratio')
    ax2.set_title('Inertia Drop Ratio')
    ax2.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "cluster_count_selection_reference.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig_sil, ax_sil = plt.subplots(figsize=(6, 4), dpi=300)
    ax_sil.plot(k_range, sil_scores, 'o-', color='green')
    ax_sil.set_xlabel('Number of Clusters')
    ax_sil.set_ylabel('Silhouette Score')
    ax_sil.set_title('Silhouette Score')
    ax_sil.grid(True)
    fig_sil.tight_layout()
    fig_sil.savefig(os.path.join(output_dir, "silhouette_reference.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_sil)
    fig_db, ax_db = plt.subplots(figsize=(6, 4), dpi=300)
    ax_db.plot(k_range, db_scores, 'o-', color='orange')
    ax_db.set_xlabel('Number of Clusters')
    ax_db.set_ylabel('Davies-Bouldin Index')
    ax_db.set_title('Davies-Bouldin Index')
    ax_db.grid(True)
    fig_db.tight_layout()
    fig_db.savefig(os.path.join(output_dir, "davies_bouldin_reference.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_db)
    fig_ch, ax_ch = plt.subplots(figsize=(6, 4), dpi=300)
    ax_ch.plot(k_range, ch_scores, 'o-', color='teal')
    ax_ch.set_xlabel('Number of Clusters')
    ax_ch.set_ylabel('Calinski-Harabasz Index')
    ax_ch.set_title('Calinski-Harabasz Index')
    ax_ch.grid(True)
    fig_ch.tight_layout()
    fig_ch.savefig(os.path.join(output_dir, "calinski_harabasz_reference.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_ch)
    print("\n" + "=" * 60)
    print("Summary of cluster-count evaluation metrics (for manual reference):")
    print("-" * 60)
    print(f"{'k':<5} | {'Inertia (lower better)':<22} | {'Inertia drop':<15} | {'Silhouette (closer to 1)':<27} | {'DB index (lower better)':<24}")
    print("-" * 60)
    for i, k in enumerate(k_range):
        kd_text = f"{float(drop_rates[i]):.4f}" if i > 0 and np.isfinite(drop_rates[i]) else "-"
        print(f"{k:<5} | {inertias[i]:<20.4f} | {kd_text:<15} | {sil_scores[i]:<25.4f} | {db_scores[i]:<20.4f}")
    print("=" * 60)
    inertia_k = elbow_k
    if inertia_k is None:
        finite_drop_pairs = [
            (k_range[i], float(drop_rates[i]))
            for i in range(1, len(k_range))
            if np.isfinite(drop_rates[i])
        ]
        if finite_drop_pairs:
            max_drop = max(v for _, v in finite_drop_pairs)
            inertia_k = min(k for k, v in finite_drop_pairs if np.isclose(v, max_drop))
        else:
            inertia_k = int(k_range[0]) if k_range else None
    finite_sil_pairs = [(int(k), float(v)) for k, v in zip(k_range, sil_scores) if np.isfinite(v)]
    finite_db_pairs = [(int(k), float(v)) for k, v in zip(k_range, db_scores) if np.isfinite(v)]
    finite_ch_pairs = [(int(k), float(v)) for k, v in zip(k_range, ch_scores) if np.isfinite(v)]
    silhouette_k = min((k for k, v in finite_sil_pairs if np.isclose(v, max(vv for _, vv in finite_sil_pairs))), default=None) if finite_sil_pairs else None
    davies_bouldin_k = min((k for k, v in finite_db_pairs if np.isclose(v, min(vv for _, vv in finite_db_pairs))), default=None) if finite_db_pairs else None
    calinski_harabasz_k = min((k for k, v in finite_ch_pairs if np.isclose(v, max(vv for _, vv in finite_ch_pairs))), default=None) if finite_ch_pairs else None
    metric_k = {
        "inertia": int(inertia_k) if inertia_k is not None else None,
        "silhouette": int(silhouette_k) if silhouette_k is not None else None,
        "davies_bouldin": int(davies_bouldin_k) if davies_bouldin_k is not None else None,
        "calinski_harabasz": int(calinski_harabasz_k) if calinski_harabasz_k is not None else None,
    }
    metric_candidates = [
        int(v)
        for v in [metric_k["inertia"], metric_k["silhouette"], metric_k["davies_bouldin"], metric_k["calinski_harabasz"]]
        if v is not None
    ]
    auto_final_k = int(min(metric_candidates)) if metric_candidates else None
    print("Recommended k values from each metric:")
    print(f"  - Inertia/kd: {metric_k['inertia']}")
    print(f"  - Silhouette: {metric_k['silhouette']}")
    print(f"  - Davies-Bouldin: {metric_k['davies_bouldin']}")
    print(f"  - Calinski-Harabasz: {metric_k['calinski_harabasz']}")
    print(f"Automatic cluster count using the minimum recommended-k rule: {auto_final_k}")
    metric_rows = [
        {"metric": "inertia_kd", "recommended_k": metric_k["inertia"]},
        {"metric": "silhouette", "recommended_k": metric_k["silhouette"]},
        {"metric": "davies_bouldin", "recommended_k": metric_k["davies_bouldin"]},
        {"metric": "calinski_harabasz", "recommended_k": metric_k["calinski_harabasz"]},
        {"metric": "auto_final_k_min_recommended", "recommended_k": auto_final_k},
    ]
    metric_k_csv_path = os.path.join(output_dir, "cluster_count_metric_recommendations.csv")
    pd.DataFrame(metric_rows).to_csv(metric_k_csv_path, index=False, encoding="utf-8-sig")
    metric_k_json_path = os.path.join(output_dir, "cluster_count_metric_recommendations.json")
    with open(metric_k_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_k": metric_k,
                "auto_final_k": auto_final_k,
                "k_range": [int(x) for x in k_range],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved metric-wise recommended k values: {metric_k_csv_path}")
    print(f"Saved metric-wise recommended k values (JSON): {metric_k_json_path}")
    return k_range, inertias, sil_scores, db_scores, ch_scores, elbow_k, metric_k, auto_final_k


def plot_sample_hex_cluster(som, grid_labels, sample_clusters, sample_ids, bmus, output_dir="Sample_Cluster_Results"):
    os.makedirs(output_dir, exist_ok=True)
    plt.close('all')
    _setup_matplotlib_output_style(plt)
    rows, cols = som.get_weights().shape[:2]
    radius = 0.5
    hex_width = np.sqrt(3) * radius * 1.05
    hex_height = 1.5 * radius * 1.05
    fig_width, fig_height = _get_som_canvas_size(rows, cols)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_aspect('equal')
    n_clusters = len(np.unique(grid_labels))
    base_cmap = get_element_component_cmap()
    cluster_colors = [base_cmap(i / max(1, n_clusters - 1)) for i in range(n_clusters)]
    cmap = ListedColormap(cluster_colors)
    norm = mpl.colors.BoundaryNorm(np.arange(n_clusters + 1) - 0.5, n_clusters)
    for i in range(rows):
        for j in range(cols):
            x = j * hex_width + (i % 2) * hex_width / 2
            y = i * hex_height
            hexagon = RegularPolygon(
                (x, y), 6, radius=radius, orientation=0,
                facecolor=cmap(norm(grid_labels[i, j])),
                edgecolor='white', linewidth=0.8
            )
            ax.add_patch(hexagon)
    x_min = -hex_width / 2
    x_max = (cols - 1) * hex_width + (1 if rows % 2 else 0) * hex_width / 2 + hex_width * 1.2
    y_min = -radius
    y_max = (rows - 1) * hex_height + radius * 1.35
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_ticks(np.arange(n_clusters))
    cb.set_ticklabels([f"Cluster {i}" for i in range(n_clusters)])
    cb.ax.tick_params(labelsize=27)
    for tick_label in cb.ax.get_yticklabels():
        tick_label.set_fontname("Times New Roman")
    cb.set_label("Cluster ID", fontsize=27, fontname="Times New Roman")
    ax.set_title("SOM Sample Cluster Map", fontsize=27, pad=8, fontname="Times New Roman")
    output_path = f"{output_dir}/som_hex_sample_clusters.png"
    _save_fixed_canvas_figure(fig, output_path)
    plt.close(fig)
    print(f"Saved SOM hexagonal sample-cluster figure: {output_path}")
    return output_path


def plot_sample_cluster_spatial_map(
    data,
    sample_labels,
    x_col="XXX",
    y_col="YYY",
    output_dir="Sample_Cluster_Results",
    output_filename="sample_cluster_spatial_map.png",
    title="Sample Cluster Spatial Map",
):
    lang = _resolve_output_language()
    _setup_matplotlib_output_style(plt)
    if not isinstance(data, pd.DataFrame) or data.empty:
        return ""
    if sample_labels is None:
        return ""
    labels_arr = np.asarray(sample_labels).reshape(-1)
    if labels_arr.shape[0] != len(data):
        return ""
    x_candidates = [str(x_col), "Longitude", "longitude", "LONGITUDE", "Lon", "lon", "X", "x"]
    y_candidates = [str(y_col), "Latitude", "latitude", "LATITUDE", "Lat", "lat", "Y", "y"]
    x_use = None
    y_use = None
    for c in x_candidates:
        if c in data.columns:
            x_use = c
            break
    for c in y_candidates:
        if c in data.columns:
            y_use = c
            break
    if x_use is None or y_use is None:
        return ""
    df_map = data.copy()
    try:
        try:
            from utils.data_utils import normalize_coordinates as _normalize_coordinates
        except Exception:
            from .utils.data_utils import normalize_coordinates as _normalize_coordinates
        df_map, _ = _normalize_coordinates(df_map, x_col=str(x_use), y_col=str(y_use), lon_col="Longitude", lat_col="Latitude")
        x_use = "Longitude"
        y_use = "Latitude"
    except Exception:
        pass
    draw_df = pd.DataFrame(
        {
            "x": pd.to_numeric(df_map[x_use], errors="coerce"),
            "y": pd.to_numeric(df_map[y_use], errors="coerce"),
            "cluster": pd.to_numeric(pd.Series(labels_arr, index=df_map.index), errors="coerce"),
        }
    ).dropna()
    if draw_df.empty:
        return ""
    x = draw_df["x"].to_numpy(dtype=float)
    y = draw_df["y"].to_numpy(dtype=float)
    c = draw_df["cluster"].to_numpy(dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x = x[finite_mask]
    y = y[finite_mask]
    c = c[finite_mask]
    if x.size == 0:
        return ""
    unique_clusters = np.unique(c.astype(int))
    if unique_clusters.size == 0:
        return ""
    n_clusters = int(unique_clusters.size)
    cluster_min = int(unique_clusters.min())
    cluster_max = int(unique_clusters.max())
    grid_size = 300
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(xi, yi)
    try:
        from scipy.spatial import cKDTree
        pts = np.column_stack([x, y])
        tree = cKDTree(pts)
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        _, nn_idx = tree.query(grid_pts, k=1)
        grid_cluster = c[nn_idx].reshape(grid_x.shape)
    except Exception:
        return ""
    base_colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
    if n_clusters <= len(base_colors):
        cmap = ListedColormap(base_colors[:n_clusters])
    else:
        cmap = plt.cm.get_cmap("tab20", n_clusters)
    norm = mpl.colors.BoundaryNorm(np.arange(cluster_min - 0.5, cluster_max + 1.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.grid(False)
    im = ax.imshow(
        grid_cluster,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        aspect="equal",
    )
    ax.scatter(x, y, c=c, s=8, cmap=cmap, norm=norm, edgecolors="none", alpha=0.45)
    possible_label_cols = ["label", "target", "deposit", "label_encoded", "target_encoded", "labeled", "has_deposit", "is_deposit", "Ore"]
    label_col = None
    for col in possible_label_cols:
        if col in df_map.columns:
            label_col = col
            break
    if label_col is None:
        for col in df_map.columns:
            col_lower = str(col).lower()
            if any((lc.lower() in col_lower for lc in possible_label_cols)):
                label_col = col
                break
    if label_col is not None and label_col in df_map.columns:
        try:
            known_mask = pd.to_numeric(df_map[label_col], errors="coerce") == 1
            known_df = df_map.loc[known_mask, [x_use, y_use]].copy()
            known_df["x"] = pd.to_numeric(known_df[x_use], errors="coerce")
            known_df["y"] = pd.to_numeric(known_df[y_use], errors="coerce")
            known_df = known_df.dropna(subset=["x", "y"])
            if not known_df.empty:
                ax.scatter(
                    known_df["x"].to_numpy(dtype=float),
                    known_df["y"].to_numpy(dtype=float),
                    color="cyan",
                    s=30,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.85,
                    label=_localize_text("Known deposit", lang=lang),
                )
        except Exception:
            pass
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=14,
            loc="lower left",
            scatterpoints=1,
            handlelength=0.9,
            handletextpad=0.35,
            borderpad=0.35,
            labelspacing=0.3,
            markerscale=0.9,
        )
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, fraction=0.06, aspect=10)
    cbar.set_ticks(unique_clusters)
    cbar.set_ticklabels([f"Cluster {int(v)}" for v in unique_clusters])
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(_localize_text(title, lang=lang), fontsize=20)
    ax.set_xlabel(_localize_text("Longitude", lang=lang), fontsize=16)
    ax.set_ylabel(_localize_text("Latitude", lang=lang), fontsize=16)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_filename)
    fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight", pil_kwargs={"optimize": True, "compress_level": 9})
    plt.close(fig)
    return out_path


def plot_known_sample_count_by_cluster(
    data,
    sample_labels,
    output_dir="Sample_Cluster_Results",
    output_filename="known_sample_count_by_cluster.png",
    output_csv_filename="known_sample_count_by_cluster.csv",
):
    if not isinstance(data, pd.DataFrame) or data.empty:
        return {"plot_path": "", "csv_path": "", "label_col": "", "counts": []}
    if sample_labels is None:
        return {"plot_path": "", "csv_path": "", "label_col": "", "counts": []}
    labels_arr = np.asarray(sample_labels).reshape(-1)
    if labels_arr.shape[0] != len(data):
        return {"plot_path": "", "csv_path": "", "label_col": "", "counts": []}
    possible_label_cols = ["Ore", "label", "target", "deposit", "label_encoded", "target_encoded", "labeled", "has_deposit", "is_deposit"]
    label_col = None
    for col in possible_label_cols:
        if col in data.columns:
            label_col = col
            break
    if label_col is None:
        for col in data.columns:
            col_lower = str(col).lower()
            if any((lc.lower() in col_lower for lc in possible_label_cols)):
                label_col = str(col)
                break
    if label_col is None:
        return {"plot_path": "", "csv_path": "", "label_col": "", "counts": []}
    label_num = pd.to_numeric(data[label_col], errors="coerce")
    known_mask = label_num == 1
    cluster_series = pd.to_numeric(pd.Series(labels_arr, index=data.index), errors="coerce")
    stat_df = pd.DataFrame({"cluster": cluster_series, "known": known_mask.astype(int)}).dropna(subset=["cluster"])
    if stat_df.empty:
        return {"plot_path": "", "csv_path": "", "label_col": str(label_col), "counts": []}
    stat_df["cluster"] = stat_df["cluster"].astype(int)
    total_counts = stat_df.groupby("cluster", as_index=False).size().rename(columns={"size": "sample_count"})
    known_counts = stat_df.groupby("cluster", as_index=False)["known"].sum().rename(columns={"known": "known_sample_count"})
    merged = total_counts.merge(known_counts, on="cluster", how="left")
    merged["known_ratio"] = np.where(merged["sample_count"] > 0, merged["known_sample_count"] / merged["sample_count"], 0.0)
    total_known = float(pd.to_numeric(merged["known_sample_count"], errors="coerce").fillna(0).sum())
    merged["known_share"] = np.where(total_known > 0.0, merged["known_sample_count"] / total_known, 0.0)
    merged = merged.sort_values("cluster").reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, output_csv_filename)
    _localize_dataframe_headers(merged).to_csv(csv_path, index=False, encoding="utf-8-sig")
    lang = _resolve_output_language()
    _setup_matplotlib_output_style(plt)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    known_pct = merged["known_share"].to_numpy(dtype=float) * 100.0
    bars = ax.bar(
        [f"Cluster {int(x)}" for x in merged["cluster"].tolist()],
        known_pct,
        color="#4E79A7",
        alpha=0.9,
    )
    for rect, value in zip(bars, known_pct.tolist()):
        ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(), f"{float(value):.2f}%", ha="center", va="bottom", fontsize=16)
    title_prefix = _get_bilingual_text("Known Sample Share Distribution", "Known Sample Share Distribution", lang=lang)
    ax.set_title(f"{title_prefix}", fontsize=20)
    ax.set_xlabel(_get_bilingual_text("Cluster", "Cluster", lang=lang), fontsize=16)
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, output_filename)
    fig.savefig(plot_path, dpi=300, format="png", bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_path": os.path.abspath(plot_path),
        "csv_path": os.path.abspath(csv_path),
        "label_col": str(label_col),
        "counts": merged.to_dict(orient="records"),
    }


def calculate_element_component_value(som, X_element, bmus):
    X_element = ensure_2d_array(X_element, "element data X_element")
    if X_element.shape[1] != 1:
        raise ValueError(f"X_element must be a 2D array with exactly one column, but got shape {X_element.shape}")
    som_rows, som_cols = som.get_weights().shape[:2]
    count_matrix = np.zeros((som_rows, som_cols), dtype=int)
    sum_matrix = np.zeros((som_rows, som_cols), dtype=float)
    if len(bmus.shape) != 2 or bmus.shape[1] != 2:
        raise ValueError(f"bmus must have shape (n_samples, 2), but got {bmus.shape}")
    if X_element.shape[0] != bmus.shape[0]:
        raise ValueError(f"Sample-count mismatch: X_element={X_element.shape[0]}, bmus={bmus.shape[0]}")
    bmus_i = bmus[:, 0].astype(int)
    bmus_j = bmus[:, 1].astype(int)
    valid_mask = (bmus_i >= 0) & (bmus_i < som_rows) & (bmus_j >= 0) & (bmus_j < som_cols)
    valid_count = np.sum(valid_mask)
    logger.debug(f"Element-value range: [{X_element.min():.4f}, {X_element.max():.4f}]")
    logger.debug(f"Valid BMU ratio: {valid_count / len(bmus):.2%} ({valid_count}/{len(bmus)})")
    np.add.at(count_matrix, (bmus_i[valid_mask], bmus_j[valid_mask]), 1)
    np.add.at(sum_matrix, (bmus_i[valid_mask], bmus_j[valid_mask]), X_element[valid_mask, 0])
    component_values = np.divide(
        sum_matrix,
        count_matrix,
        out=np.full_like(sum_matrix, np.nan),
        where=count_matrix > 0
    )
    zero_mask = count_matrix == 0
    num_zero_neurons = np.sum(zero_mask)
    if num_zero_neurons > 0:
        logger.debug(f"Found {num_zero_neurons} gray neurons; filling them with nearest valid values...")
        valid_coords = np.argwhere(~zero_mask)
        valid_values = component_values[~zero_mask]
        for (i, j) in np.argwhere(zero_mask):
            distances = np.sqrt((valid_coords[:, 0] - i) ** 2 + (valid_coords[:, 1] - j) ** 2)
            nearest_idx = np.argmin(distances)
            component_values[i, j] = valid_values[nearest_idx]
    valid_neurons = np.sum(~np.isnan(component_values))
    logger.debug(f"Valid neurons: {valid_neurons}/{som_rows * som_cols}")
    return component_values


def plot_single_element_component_plane(som, component_values, element_name, output_dir, global_min=None, global_max=None, use_element_range=True, grid_labels=None, with_boundary=False):
    os.makedirs(output_dir, exist_ok=True)
    plt.close('all')
    som_rows, som_cols = som.get_weights().shape[:2]
    radius = 0.5
    hex_width = np.sqrt(3) * radius * 1.05
    hex_height = 1.5 * radius * 1.05
    element_min = np.nanmin(component_values)
    element_max = np.nanmax(component_values)
    vmin, vmax = element_min, element_max
    if np.isclose(vmin, vmax):
        vmin -= 0.1
        vmax += 0.1
    cmap = get_element_component_cmap()
    cmap.set_bad(color='lightgray', alpha=0.5)
    fig_width = max(12, som_cols * 0.5)
    fig_height = max(10, som_rows * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')
    patches = []
    values_list = []
    for i in range(som_rows):
        for j in range(som_cols):
            x = j * hex_width + (i % 2) * hex_width / 2
            y = i * hex_height
            val = component_values[i, j]
            values_list.append(val)
            hexagon = RegularPolygon((x, y), 6, radius=radius, orientation=0, edgecolor='white', linewidth=0.5)
            if np.isnan(val):
                hexagon.set_facecolor('lightgray')
            else:
                norm_val = (val - vmin) / (vmax - vmin)
                norm_val = np.clip(norm_val, 0, 1)
                hexagon.set_facecolor(cmap(norm_val))
            patches.append(hexagon)
    pc = mpl.collections.PatchCollection(patches, alpha=0.9, zorder=2, cmap=cmap)
    pc.set_array(np.array(values_list))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    if with_boundary and grid_labels is not None:
        hex_neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1)]
        for i in range(som_rows):
            for j in range(som_cols):
                current_label = grid_labels[i, j]
                for di, dj in hex_neighbors:
                    ni, nj = i + di, j + dj
                    if i % 2 == 0:
                        if dj > 0:
                            nj -= 1
                    else:
                        if dj < 0:
                            nj += 1
                    if 0 <= ni < som_rows and 0 <= nj < som_cols:
                        if grid_labels[ni, nj] != current_label:
                            x1 = j * hex_width + (i % 2) * hex_width / 2
                            y1 = i * hex_height
                            x2 = nj * hex_width + (ni % 2) * hex_width / 2
                            y2 = ni * hex_height
                            ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, zorder=3)
    x_min = -hex_width / 2
    x_max = (som_cols - 1) * hex_width + (1 if som_rows % 2 else 0) * hex_width / 2 + hex_width * 1.2
    y_min = -radius
    y_max = (som_rows - 1) * hex_height + radius * 1.35
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    cb = plt.colorbar(pc, cax=cax)
    cb.set_label(f'{element_name} Concentration', fontsize=20)
    cb.ax.tick_params(labelsize=16)
    title_suffix = " (with cluster boundaries)" if with_boundary else ""
    ax.set_title(f'{element_name} Component Plane{title_suffix}', fontsize=20, pad=8)
    boundary_suffix = "_with_boundary" if with_boundary else ""
    output_path = f"{output_dir}/{element_name}_component_plane{boundary_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.debug(f"Saved {element_name} component plane{title_suffix}: {output_path}")


def generate_all_elements_component_planes(som, X_scaled, elements, bmus, grid_labels, output_dir, force_element_range=False):
    X_scaled = ensure_2d_array(X_scaled, "component-plane input X_scaled")
    component_plane_dir = output_dir
    os.makedirs(component_plane_dir, exist_ok=True)
    print(f"\nStarting batch generation of component planes for {len(elements)} elements (boundary-free mode only)...")
    all_component_values = []
    for elem_idx in range(X_scaled.shape[1]):
        X_element = X_scaled[:, [elem_idx]]
        component_values = calculate_element_component_value(som, X_element, bmus)
        all_component_values.append(component_values)
    flattened = np.concatenate([cv[~np.isnan(cv)].flatten() for cv in all_component_values])
    global_min, global_max = np.min(flattened), np.max(flattened)
    logger.debug(f"Global element-value range: {global_min:.4f} to {global_max:.4f}")
    value_ranges = []
    for elem_idx, elem_name in enumerate(elements):
        logger.debug(f"Processing element: {elem_name} ({elem_idx + 1}/{len(elements)})")
        X_element = X_scaled[:, [elem_idx]]
        component_values = calculate_element_component_value(som, X_element, bmus)
        elem_min = float(np.nanmin(component_values))
        elem_max = float(np.nanmax(component_values))
        value_ranges.append(
            {
                "element": str(elem_name),
                "component_value_min": elem_min,
                "component_value_max": elem_max,
            }
        )
        plot_single_element_component_plane(
            som,
            component_values,
            elem_name,
            component_plane_dir,
            global_min=global_min,
            global_max=global_max,
            use_element_range=force_element_range,
            grid_labels=grid_labels,
            with_boundary=False
        )
    value_range_csv_path = os.path.join(component_plane_dir, "component_plane_value_ranges.csv")
    _localize_dataframe_headers(pd.DataFrame(value_ranges)).to_csv(value_range_csv_path, index=False, encoding="utf-8-sig")
    print(f"All element component planes have been saved to: {component_plane_dir}")
    print(f"Component-plane value ranges have been saved to: {value_range_csv_path}")
    print("Each element generates a single file only: xxx_component_plane.png (no boundaries)")
    return all_component_values


def analyze_element_component_correlation(som, X_scaled, elements, bmus, all_component_values, output_dir):
    correlation_dir = output_dir
    os.makedirs(correlation_dir, exist_ok=True)
    component_values = {}
    for idx, elem_name in enumerate(elements):
        component_values[elem_name] = all_component_values[idx]
    n_elements = len(elements)
    pearson_corr = np.zeros((n_elements, n_elements))
    pattern_similarity = np.zeros((n_elements, n_elements))
    for i, elem1 in enumerate(elements):
        cv1 = component_values[elem1].flatten()
        threshold1 = np.percentile(cv1, 75)
        high_mask1 = cv1 >= threshold1
        for j, elem2 in enumerate(elements):
            cv2 = component_values[elem2].flatten()
            pearson_corr[i, j] = np.corrcoef(cv1, cv2)[0, 1]
            threshold2 = np.percentile(cv2, 75)
            high_mask2 = cv2 >= threshold2
            overlap = np.sum(high_mask1 & high_mask2) / np.sum(high_mask1 | high_mask2) if np.sum(high_mask1 | high_mask2) > 0 else 0
            pattern_similarity[i, j] = overlap
    corr_df = pd.DataFrame(pearson_corr, index=elements, columns=elements)
    pattern_df = pd.DataFrame(pattern_similarity, index=elements, columns=elements)
    _localize_dataframe_headers(corr_df).to_csv(f"{correlation_dir}/element_component_plane_pearson_correlation.csv", encoding='utf-8-sig')
    _localize_dataframe_headers(pattern_df).to_csv(f"{correlation_dir}/element_component_plane_high_value_overlap.csv", encoding='utf-8-sig')
    plt.figure(figsize=(14, 12), dpi=300)
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=False, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, cbar_kws={"label": "Pearson Correlation Coefficient"})
    plt.title("Correlation Heatmap of Element Component Planes", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{correlation_dir}/element_component_plane_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(12, 8), dpi=300)
    sns.clustermap(corr_df, cmap="coolwarm", vmin=-1, vmax=1, figsize=(12, 10), cbar_pos=(0.02, 0.8, 0.05, 0.18))
    plt.savefig(f"{correlation_dir}/element_component_plane_clustermap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Element component-plane correlation analysis completed. Results saved in: {correlation_dir}")
    return corr_df, pattern_df


def analyze_cluster_elements(data, elements, cluster_col='cluster', output_dir="cluster_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    cluster_stats = {}
    overall_mean = data[elements].mean()
    for cluster_id in sorted(data[cluster_col].unique()):
        if pd.isna(cluster_id):
            continue
        cluster_samples = data[data[cluster_col] == cluster_id]
        cluster_mean = cluster_samples[elements].mean()
        cluster_std = cluster_samples[elements].std()
        enrichment = cluster_mean / overall_mean.replace(0, 1e-10)
        cluster_stats[cluster_id] = {
            'mean': cluster_mean,
            'std': cluster_std,
            'enrichment': enrichment,
            'sample_count': len(cluster_samples)
        }
        print(f"Cluster {cluster_id} contains {len(cluster_samples)} samples")
    stats_df = pd.DataFrame({f"cluster_{cid}_mean": stats['mean'] for cid, stats in cluster_stats.items()})
    _localize_dataframe_headers(stats_df).to_csv(f"{output_dir}/cluster_element_means.csv", encoding='utf-8-sig')
    print(f"Saved cluster element statistics: {output_dir}/cluster_element_means.csv")
    enrichment_df = pd.DataFrame({f"Cluster {cid}": stats['enrichment'] for cid, stats in cluster_stats.items()})
    plt.close('all')
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(enrichment_df, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Enrichment Factor (Cluster Mean / Global Mean)'})
    plt.title("Element Enrichment Heatmap by Cluster", fontsize=15, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_element_enrichment_heatmap.png", bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    print(f"Saved enrichment heatmap: {output_dir}/cluster_element_enrichment_heatmap.png")
    with open(f"{output_dir}/cluster_signature_elements.txt", 'w', encoding='utf-8') as f:
        for cid, stats in cluster_stats.items():
            top_elements = stats['enrichment'].sort_values(ascending=False).head(3).index.tolist()
            f.write(f"Cluster {cid} signature elements (highest enrichment factors): {', '.join(top_elements)}\n")
            print(f"Cluster {cid} signature elements: {', '.join(top_elements)}")
    return cluster_stats


def train_base_model_for_shap(X, y, random_state=42):
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    base_model.fit(X, y)
    print(f"Base model training completed. Accuracy: {base_model.score(X, y):.4f}")
    return base_model


def calculate_shap_values(base_model, X, elements, sample_ids, core_elements_config):
    valid_core_elements = [elem for elem in core_elements_config if elem in elements]
    if not valid_core_elements:
        raise ValueError(f"None of the configured core elements {core_elements_config} exist in the data element list. Please check the input.")
    if len(valid_core_elements) != len(core_elements_config):
        missing = [elem for elem in core_elements_config if elem not in elements]
        print(f"Warning: some core elements do not exist in the dataset and were filtered automatically: {missing}")
    core_indices = [elements.index(elem) for elem in valid_core_elements]
    num_core = len(valid_core_elements)
    print(f"Locked core-element combination ({num_core} elements): {valid_core_elements}")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) >= 2 else shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if shap_values.ndim != 2:
        raise ValueError(f"Invalid SHAP dimension: a 2D matrix is required, but got {shap_values.ndim} dimensions")
    shap_values_core = shap_values[:, core_indices]
    global_shap = np.mean(np.abs(shap_values_core), axis=0).flatten()
    sorted_indices = np.lexsort((valid_core_elements, -global_shap))
    rank = np.zeros(num_core, dtype=int)
    for r, idx in enumerate(sorted_indices, 1):
        rank[idx] = r
    global_shap_df = pd.DataFrame({
        "Element Name": valid_core_elements,
        "Global SHAP Value (Mean Absolute)": global_shap,
        "Contribution Rank Within Combination": rank
    }).sort_values("Global SHAP Value (Mean Absolute)", ascending=False)
    sample_shap_df = pd.DataFrame(
        shap_values_core,
        columns=[f"{elem}_SHAP" for elem in valid_core_elements],
        index=sample_ids
    )
    sample_shap_df["Total SHAP Value Of Core Elements"] = sample_shap_df.sum(axis=1)
    print("\nSHAP calculation check:")
    print(f"  - Full SHAP matrix shape: {shap_values.shape} (expected: sample_count x 37)")
    print(f"  - Number of core elements: {num_core} (consistent with the input combination)")
    print(f"  - Sample-level SHAP shape: {sample_shap_df.shape} (sample_count x core_element_count + 1)")
    print(f"  - Rank range within the combination: 1-{num_core} (as expected)")
    return shap_values, explainer, global_shap_df, sample_shap_df, valid_core_elements


def plot_shap_importance_ranking(global_shap_df, valid_core_elements, output_dir):
    core_shap_data = global_shap_df[global_shap_df["Element Name"].isin(valid_core_elements)].sort_values("Global SHAP Value (Mean Absolute)", ascending=True)
    plt.figure(figsize=(10, 8), dpi=300)
    bars = plt.barh(
        y=core_shap_data["Element Name"],
        width=core_shap_data["Global SHAP Value (Mean Absolute)"],
        color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(core_shap_data))),
        edgecolor="black",
        linewidth=0.8
    )
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(
            width + max(core_shap_data["Global SHAP Value (Mean Absolute)"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            fontsize=9,
            fontweight="bold"
        )
    if chinese_font is not None:
        plt.xlabel("SHAP value", fontsize=12, fontproperties=chinese_font, fontweight="bold")
        plt.ylabel("Core Ore-Forming Elements", fontsize=12, fontproperties=chinese_font, fontweight="bold")
        plt.title(
            f"SHAP Feature Importance Ranking of Core Ore-Forming Elements\n(Core Cluster Classification: {', '.join(valid_core_elements)})",
            fontsize=14,
            pad=20,
            fontproperties=chinese_font,
            fontweight="bold"
        )
    else:
        plt.xlabel("SHAP value", fontsize=12, fontweight="bold")
        plt.ylabel("Core Ore-Forming Elements", fontsize=12, fontweight="bold")
        plt.title(
            f"SHAP Feature Importance Ranking of Core Ore-Forming Elements\n(Core Cluster Classification: {', '.join(valid_core_elements)})",
            fontsize=14,
            pad=20,
            fontweight="bold"
        )
    plt.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    ranking_plot_path = f"{output_dir}/shap_core_element_importance_ranking.png"
    plt.savefig(ranking_plot_path, bbox_inches="tight", dpi=300, facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved SHAP feature-importance ranking plot: {ranking_plot_path}")


def plot_shap_visualizations(explainer, X, shap_values, elements, sample_shap_df, global_shap_df, output_dir, valid_core_elements):
    shap_output_dir = output_dir
    os.makedirs(shap_output_dir, exist_ok=True)
    core_indices = [elements.index(elem) for elem in valid_core_elements]
    if max(core_indices) >= shap_values.shape[1]:
        raise ValueError(
            f"Core-element indices exceed the SHAP matrix width.\n"
            f"Number of SHAP columns: {shap_values.shape[1]}, maximum core-element index: {max(core_indices)}\n"
            f"Please confirm that X_scaled is complete (expected 37 columns) and that the core elements exist in the elements list."
        )
    shap_values_core = shap_values[:, core_indices]
    X_core = X[:, core_indices]
    original_font = plt.rcParams['font.family']
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(4, 30), dpi=300)
    shap.summary_plot(
        shap_values_core,
        X_core,
        feature_names=valid_core_elements,
        plot_type="dot",
        show=False,
        cmap=get_custom_cmap()
    )
    plt.xlabel("SHAP value", fontsize=16, fontweight="bold")
    plt.yticks(fontsize=14)
    if chinese_font is not None:
        plt.title(
            f"Element SHAP Contribution (Core Elements: {', '.join(valid_core_elements)})",
            fontsize=18,
            pad=20,
            fontproperties=chinese_font
        )
    else:
        plt.title(
            f"Element SHAP Contribution (Core Elements: {', '.join(valid_core_elements)})",
            fontsize=18,
            pad=20
        )
    plt.rcParams['font.family'] = original_font
    plt.tight_layout()
    summary_plot_path = f"{shap_output_dir}/shap_core_element_contribution_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved core-element SHAP contribution plot: {summary_plot_path}")
    plt.figure(figsize=(10, 6), dpi=300)
    sns.histplot(sample_shap_df["Total SHAP Value Of Core Elements"], bins=50, kde=True, color="#1f77b4")
    shap_95 = sample_shap_df["Total SHAP Value Of Core Elements"].quantile(0.95)
    plt.axvline(x=shap_95, color='red', linestyle='--', linewidth=2, label=f"95th Percentile: {shap_95:.4f}")
    if chinese_font is not None:
        plt.xlabel(f"Total SHAP Value of Core Elements ({', '.join(valid_core_elements)})", fontsize=12, fontproperties=chinese_font)
        plt.ylabel("Sample Count", fontsize=12, fontproperties=chinese_font)
        plt.title(
            f"Distribution of Total SHAP Values for Core Elements ({', '.join(valid_core_elements)})",
            fontsize=14,
            pad=20,
            fontproperties=chinese_font
        )
        plt.legend(prop=chinese_font)
    else:
        plt.xlabel(f"Total SHAP Value of Core Elements ({', '.join(valid_core_elements)})", fontsize=12)
        plt.ylabel("Sample Count", fontsize=12)
        plt.title(
            f"Distribution of Total SHAP Values for Core Elements ({', '.join(valid_core_elements)})",
            fontsize=14,
            pad=20
        )
        plt.legend()
    plt.tight_layout()
    dist_plot_path = f"{shap_output_dir}/core_element_total_shap_distribution.png"
    plt.savefig(dist_plot_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    plot_shap_importance_ranking(global_shap_df=global_shap_df, valid_core_elements=valid_core_elements, output_dir=shap_output_dir)
    global_shap_core = global_shap_df[global_shap_df["Element Name"].isin(valid_core_elements)].sort_values("Global SHAP Value (Mean Absolute)", ascending=False)
    global_shap_path = f"{shap_output_dir}/core_element_global_shap_contribution.csv"
    _localize_dataframe_headers(global_shap_core).to_csv(global_shap_path, index=False, encoding='utf-8-sig')
    sample_shap_path = f"{shap_output_dir}/sample_level_core_element_shap_values.csv"
    _localize_dataframe_headers(sample_shap_df).to_csv(sample_shap_path, encoding='utf-8-sig')
    print(f"Saved core-element SHAP data to: {shap_output_dir}")
    return shap_output_dir


def run_shap_analysis(X_scaled, sample_labels, elements, sample_ids, output_dir, core_clusters=None, core_elements=None):
    print("\n" + "=" * 60)
    print("===== Added Module: SHAP Analysis For Mineralization-Indicator Element Combinations =====")
    print("=" * 60)
    if shap is None:
        print("SHAP dependency is unavailable. Skipping SHAP analysis")
        return None, None, ""
    unique_clusters = sorted(np.unique(sample_labels).tolist())
    print("\nStep 1: Select the core mineralization clusters")
    print(f"Clusters available in the current dataset: {unique_clusters}")
    if core_clusters is None or core_clusters == "":
        print("No core clusters were provided. Skipping SHAP analysis")
        return None, None, ""
    try:
        if isinstance(core_clusters, str):
            core_cluster = [int(c.strip()) for c in str(core_clusters).split(",") if c.strip()]
        else:
            core_cluster = [int(x) for x in list(core_clusters)]
        invalid_clusters = [c for c in core_cluster if c not in unique_clusters]
        if invalid_clusters:
            raise ValueError(f"The input clusters {invalid_clusters} do not exist. Available clusters are {unique_clusters}")
        if not core_cluster:
            raise ValueError("No core clusters were provided")
    except ValueError as e:
        print(f"Input error: {e}")
        raise
    print(f"Selected core clusters: {core_cluster}")
    y = np.where(np.isin(sample_labels, core_cluster), 1, 0)
    print(f"Core-cluster labeling completed: core-cluster samples={np.sum(y)}, non-core samples={len(y) - np.sum(y)}")
    print("\nStep 2: Configure the core mineralization-related elements")
    print(f"All elements in the current dataset: {elements}")
    if core_elements is None or core_elements == "":
        print("No core mineralization-related elements were provided. Skipping SHAP analysis")
        return None, None, ""
    try:
        if isinstance(core_elements, str):
            core_elements_config = [elem.strip() for elem in str(core_elements).split(",") if elem.strip()]
        else:
            core_elements_config = [str(elem).strip() for elem in list(core_elements) if str(elem).strip()]
        invalid_elems = [elem for elem in core_elements_config if elem not in elements]
        if invalid_elems:
            raise ValueError(f"The input elements {invalid_elems} do not exist. Available elements are {elements}")
        if not core_elements_config:
            raise ValueError("No core elements were provided")
    except ValueError as e:
        print(f"Input error: {e}")
        raise
    print(f"Configured core elements: {core_elements_config}")
    base_model = train_base_model_for_shap(X_scaled, y)
    shap_values, explainer, global_shap_df, sample_shap_df, valid_core_elements = calculate_shap_values(
        base_model=base_model,
        X=X_scaled,
        elements=elements,
        sample_ids=sample_ids,
        core_elements_config=core_elements_config
    )
    shap_output_dir = plot_shap_visualizations(
        explainer=explainer,
        X=X_scaled,
        shap_values=shap_values,
        elements=elements,
        sample_shap_df=sample_shap_df,
        global_shap_df=global_shap_df,
        output_dir=output_dir,
        valid_core_elements=valid_core_elements
    )
    print("\nSHAP contribution ranking within the core-element combination:")
    core_shap_rank = global_shap_df[global_shap_df["Element Name"].isin(valid_core_elements)].sort_values("Global SHAP Value (Mean Absolute)", ascending=False)
    print(core_shap_rank[["Element Name", "Global SHAP Value (Mean Absolute)", "Contribution Rank Within Combination"]].to_string(index=False))
    return shap_values, sample_shap_df, shap_output_dir


def run_som_cluster_analysis_from_df(
    *,
    df,
    output_dir: str,
    x_col: str = "XXX",
    y_col: str = "YYY",
    elements=None,
    ore_elements=None,
    k: int | None = None,
    som_grid_m: int = 11,
    som_grid_n: int = 11,
    apply_log_all: bool = False,
    enable_shap: bool = False,
    core_clusters=None,
    core_elements=None,
):
    if elements is None:
        raise ValueError("elements cannot be empty")
    if ore_elements is None:
        ore_elements = []
    output_dir = os.path.abspath(str(output_dir))
    artifacts_dir = output_dir
    preprocessing_dir = output_dir
    os.makedirs(preprocessing_dir, exist_ok=True)
    X_scaled, X_raw, data, scaler, elements = load_geochem_data_from_df(
        df=df,
        elements=elements,
        ore_elements=ore_elements,
        log_transform=bool(apply_log_all),
        log_mode="clr",
        normalize=False,
        save_preprocessing=True,
        output_dir=preprocessing_dir,
    )
    som_input_matrix_path = os.path.join(preprocessing_dir, "som_input_X_scaled.npy")
    try:
        np.save(som_input_matrix_path, np.asarray(X_scaled, dtype=float))
    except Exception:
        som_input_matrix_path = ""
    sample_ids = data.index.tolist()
    try:
        joblib.dump(elements, os.path.join(preprocessing_dir, "som_elements.pkl"))
    except Exception:
        pass
    final_som_grid = (int(som_grid_m), int(som_grid_n))
    best_som, best_params = optimize_som_parameters(X=X_scaled, sample_ids=sample_ids, map_size=final_som_grid, output_dir=artifacts_dir)
    param_output_dir = artifacts_dir
    os.makedirs(param_output_dir, exist_ok=True)
    param_file_path = os.path.join(param_output_dir, "best_som_params.txt")
    with open(param_file_path, "w") as f:
        f.write(f"map_size={best_params['map_size'][0]},{best_params['map_size'][1]}\n")
        f.write(f"iterations={best_params['iterations']}\n")
        f.write(f"sigma={best_params['sigma']}\n")
        f.write(f"lr={best_params['lr']}\n")
    try:
        joblib.dump(best_som, os.path.join(param_output_dir, "som_model.pkl"))
    except Exception:
        pass
    qe = np.mean([np.linalg.norm(x - best_som.get_weights()[best_som.winner(x)]) for x in X_scaled])
    te = calculate_topographic_error(best_som, X_scaled)
    u_matrix = calculate_u_matrix(best_som)
    plot_u_matrix(best_som, u_matrix, output_dir=output_dir)
    k_range, inertias, sil_scores, db_scores, ch_scores, elbow_k, metric_k, auto_final_k = suggest_best_k(
        som=best_som,
        X=X_scaled,
        sample_ids=sample_ids,
        k_range=range(2, 11),
        output_dir=artifacts_dir,
    )
    max_k = int(best_som.get_weights().shape[0] * best_som.get_weights().shape[1])
    final_k = int(k) if k is not None else (int(auto_final_k) if auto_final_k is not None else (int(elbow_k) if elbow_k is not None else 5))
    if final_k > max_k:
        final_k = max_k
    if final_k < 3:
        final_k = 3
    if final_k > max_k:
        final_k = max_k
    if final_k < 3:
        raise ValueError(f"Invalid final_k for clustering: final_k={final_k}, max_k={max_k}")
    grid_labels, sample_labels, sample_clusters, bmus, metrics = apply_kmeans_to_samples(
        som=best_som,
        X=X_scaled,
        sample_ids=sample_ids,
        n_clusters=final_k,
    )
    hex_cluster_map_path = ""
    try:
        hex_cluster_map_path = plot_sample_hex_cluster(
            som=best_som,
            grid_labels=grid_labels,
            sample_clusters=sample_clusters,
            sample_ids=sample_ids,
            bmus=bmus,
            output_dir=artifacts_dir,
        )
    except Exception:
        hex_cluster_map_path = ""
    cluster_spatial_map_path = ""
    try:
        cluster_spatial_map_path = plot_sample_cluster_spatial_map(
            data=data,
            sample_labels=sample_labels,
            x_col=str(x_col),
            y_col=str(y_col),
            output_dir=artifacts_dir,
            output_filename="sample_cluster_spatial_map.png",
            title="Sample Cluster Spatial Map",
        )
    except Exception:
        cluster_spatial_map_path = ""
    arcgis_cluster_path = ""
    try:
        x_candidates = [str(x_col), "Longitude", "longitude", "LONGITUDE", "Lon", "lon", "X", "x"]
        y_candidates = [str(y_col), "Latitude", "latitude", "LATITUDE", "Lat", "lat", "Y", "y"]
        x_use = None
        y_use = None
        for c in x_candidates:
            if c in data.columns:
                x_use = c
                break
        for c in y_candidates:
            if c in data.columns:
                y_use = c
                break
        if x_use is not None and y_use is not None:
            arcgis_cluster_data = pd.DataFrame(
                {
                    "Sample ID": sample_ids,
                    str(x_use): pd.to_numeric(data[x_use], errors="coerce"),
                    str(y_use): pd.to_numeric(data[y_use], errors="coerce"),
                    "Cluster ID": pd.to_numeric(pd.Series(sample_labels, index=data.index), errors="coerce"),
                }
            ).dropna(subset=[str(x_use), str(y_use), "Cluster ID"])
            arcgis_cluster_data["Cluster ID"] = arcgis_cluster_data["Cluster ID"].astype(int)
            arcgis_cluster_path = os.path.join(artifacts_dir, "Cluster_for_ArcGIS.csv")
            _localize_dataframe_headers(arcgis_cluster_data).to_csv(arcgis_cluster_path, index=False, encoding="utf-8-sig")
    except Exception:
        arcgis_cluster_path = ""
    known_sample_count_plot = {"plot_path": "", "csv_path": "", "label_col": "", "counts": []}
    try:
        known_sample_count_plot = plot_known_sample_count_by_cluster(
            data=data,
            sample_labels=sample_labels,
            output_dir=artifacts_dir,
            output_filename="known_sample_count_by_cluster.png",
            output_csv_filename="known_sample_count_by_cluster.csv",
        )
    except Exception:
        known_sample_count_plot = {"plot_path": "", "csv_path": "", "label_col": "", "counts": []}
    component_plane_dir = os.path.join(artifacts_dir, "component_planes")
    component_plane_error = ""
    try:
        generate_all_elements_component_planes(
            som=best_som,
            X_scaled=X_scaled,
            elements=elements,
            bmus=bmus,
            grid_labels=grid_labels,
            output_dir=component_plane_dir,
            force_element_range=False,
        )
    except Exception as e:
        component_plane_error = str(e)
    shap_result = {"enabled": False, "output_dir": "", "error": ""}
    if bool(enable_shap):
        try:
            shap_values, sample_shap_df, shap_output_dir = run_shap_analysis(
                X_scaled=X_scaled,
                sample_labels=sample_labels,
                elements=elements,
                sample_ids=sample_ids,
                output_dir=artifacts_dir,
                core_clusters=core_clusters,
                core_elements=core_elements,
            )
            shap_result = {"enabled": True, "output_dir": str(shap_output_dir or ""), "error": ""}
            _ = shap_values
            _ = sample_shap_df
        except Exception as e:
            shap_result = {"enabled": False, "output_dir": "", "error": str(e)}
    return {
        "output_dir": output_dir,
        "artifacts_dir": artifacts_dir,
        "preprocessing_dir": preprocessing_dir,
        "som_input_matrix_path": os.path.abspath(som_input_matrix_path) if som_input_matrix_path else "",
        "param_file_path": os.path.abspath(param_file_path),
        "som_model_path": os.path.abspath(os.path.join(param_output_dir, "som_model.pkl")),
        "best_params": best_params,
        "qe": float(qe),
        "te": float(te),
        "u_matrix_shape": tuple(np.asarray(u_matrix).shape),
        "k_suggestion": {
            "elbow_k": elbow_k,
            "metric_k": metric_k,
            "auto_final_k": auto_final_k,
            "k_range": list(k_range) if k_range is not None else None,
            "inertias": [float(x) for x in inertias] if inertias is not None else None,
            "sil_scores": [float(x) for x in sil_scores] if sil_scores is not None else None,
            "db_scores": [float(x) for x in db_scores] if db_scores is not None else None,
            "ch_scores": [float(x) for x in ch_scores] if ch_scores is not None else None,
        },
        "final_k": int(final_k),
        "metrics": metrics,
        "sample_labels": np.asarray(sample_labels).tolist(),
        "hex_cluster_map_path": os.path.abspath(hex_cluster_map_path) if hex_cluster_map_path else "",
        "cluster_spatial_map_path": os.path.abspath(cluster_spatial_map_path) if cluster_spatial_map_path else "",
        "cluster_for_arcgis_csv": os.path.abspath(arcgis_cluster_path) if arcgis_cluster_path else "",
        "known_sample_count_by_cluster": known_sample_count_plot,
        "component_planes": {
            "output_dir": os.path.abspath(component_plane_dir),
            "error": component_plane_error,
        },
        "shap": shap_result,
    }


def load_best_som_params_from_main(params_path):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"The best-parameter file from the main code does not exist: {params_path}")
    best_params = {}
    with open(params_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if key == 'map_size':
                s = str(value).strip().strip("()")
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) >= 2:
                    best_params[key] = (int(float(parts[0])), int(float(parts[1])))
                else:
                    raise ValueError(f"Failed to parse map_size: {value}")
            elif key in ['sigma', 'lr']:
                best_params[key] = float(value)
            elif key == 'iterations':
                best_params[key] = int(value)
    required_keys = ['sigma', 'lr', 'iterations', 'map_size']
    missing_keys = [k for k in required_keys if k not in best_params]
    if missing_keys:
        raise ValueError(f"The main-code parameter file is missing required keys: {missing_keys}")
    logger.info("Best SOM parameters loaded from the main code:")
    logger.info(f"  - Neighborhood radius (sigma): {best_params['sigma']}")
    logger.info(f"  - Learning rate (lr): {best_params['lr']}")
    logger.info(f"  - Iterations: {best_params['iterations']}")
    logger.info(f"  - Grid size (map_size): {best_params['map_size']}")
    return best_params


def calculate_adaptive_spatial_radius(coordinates, k_neighbors=5):
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(coordinates)
    distances, _ = nn.kneighbors(coordinates)
    adaptive_radius = np.mean(distances[:, -1])
    logger.info(f"Adaptive spatial-filter radius: {adaptive_radius:.2f} (estimated from 3-nearest-neighbor distances)")
    return adaptive_radius


def load_mineral_elements_data_from_df(df, x_col, y_col, mineral_elements, sample_id_col=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty DataFrame")
    missing_cols = [col for col in mineral_elements if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df is missing mineralization-related element columns: {missing_cols}")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"df is missing coordinate columns: x_col={x_col}, y_col={y_col}")
    data = df[mineral_elements].to_numpy(copy=True)
    coordinates = df[[x_col, y_col]].to_numpy(copy=True)
    sample_ids = df[sample_id_col].to_numpy(copy=True) if sample_id_col else None
    logger.info(f"Loaded mineralization-related element data successfully: {len(data)} samples, {len(mineral_elements)} elements")
    data = np.where(data <= 0, np.min(data[data > 0]) / 10, data)
    return data, coordinates, mineral_elements, sample_ids, df[[x_col, y_col]].copy()


def train_mineral_som(data, mineral_elements, grid_size, sigma, learning_rate, iterations, save_path=None, full_som_path=None, full_elements_path=None):
    if MiniSom is None:
        raise ModuleNotFoundError("Missing dependency `minisom`. Please install it first: python -m pip install minisom")
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "mineral_som_model.pkl")
    n_features = data.shape[1]
    logger.info("\n=== Training The Mineral-Element-Specific SOM Model ===")
    logger.info(f"  - Input dimension: {n_features} (number of mineralization-related elements)")
    logger.info(f"  - Grid size: {grid_size[0]}x{grid_size[1]}")
    logger.info(f"  - Iterations: {iterations}")
    init_weights = None
    if full_som_path and os.path.exists(full_som_path):
        full_som = joblib.load(full_som_path)
        if full_elements_path is None:
            base_dir = os.path.dirname(full_som_path)
            candidate_paths = [
                os.path.join(base_dir, "som_elements.pkl"),
                os.path.join(base_dir, "preprocessing", "som_elements.pkl"),
            ]
        else:
            candidate_paths = [str(full_elements_path)]
        resolved_elements_path = ""
        for p in candidate_paths:
            if os.path.exists(p):
                resolved_elements_path = str(p)
                break
        if not resolved_elements_path:
            logger.warning(f"The element-name list file does not exist: {candidate_paths}. Cannot initialize with weights from the full-element model.")
        else:
            full_elements = joblib.load(resolved_elements_path)
            try:
                mineral_indices = [list(full_elements).index(elem) for elem in mineral_elements if elem in full_elements]
                if len(mineral_indices) == len(mineral_elements):
                    init_weights = full_som.get_weights()[:, :, mineral_indices]
                    logger.info("Initialized with weights from the full-element SOM model in the main code (restricted to mineral-element dimensions)")
                else:
                    missing_in_full = [elem for elem in mineral_elements if elem not in full_elements]
                    logger.warning(f"The full-element model from the main code is missing some mineral elements: {missing_in_full}. Using random initialization.")
            except Exception as e:
                logger.warning(f"Failed to retrieve weights from the full-element model: {e}. Using random initialization.")
    else:
        logger.warning("The full-element SOM model from the main code does not exist. Using random initialization.")
    som = MiniSom(
        x=grid_size[0],
        y=grid_size[1],
        input_len=n_features,
        sigma=sigma,
        learning_rate=learning_rate,
        topology='hexagonal',
        random_seed=42
    )
    if init_weights is not None and init_weights.shape == (grid_size[0], grid_size[1], n_features):
        som._weights = init_weights
    som.train_random(data, iterations)
    joblib.dump(som, save_path)
    logger.info(f"Mineral-element-specific SOM model saved to {save_path}")
    return som


def calculate_mineral_qe(data, som):
    bmus = [som.winner(x) for x in data]
    winner_weights = np.array([som.get_weights()[i, j] for i, j in bmus])
    conc_mean = np.mean(data, axis=1)
    conc_percentile = np.array([np.sum(conc_mean <= c) / len(conc_mean) for c in conc_mean])
    positive_diff = np.maximum(data - winner_weights, 0)
    elem_correlations = []
    for i in range(data.shape[1]):
        corr = np.corrcoef(data[:, i], conc_mean)[0, 1]
        elem_correlations.append(abs(corr))
    elem_weights = np.array(elem_correlations) / np.sum(elem_correlations)
    weighted_diff = positive_diff * elem_weights
    base_qe = np.linalg.norm(weighted_diff, axis=1)
    conc_weight = 2 / (1 + np.exp(-5 * (conc_percentile - 0.5))) - 1
    enhanced_qe = base_qe * (1 + conc_weight)
    enhanced_qe = (enhanced_qe - np.min(enhanced_qe)) / (np.max(enhanced_qe) - np.min(enhanced_qe))
    return enhanced_qe


def calculate_element_qe_correlation(original_data, qe_scores, element_names):
    logger.info("\n=== Computing Correlations Between Element Concentrations And QE Anomaly Scores ===")
    original_data = ensure_2d_array(original_data, "original element data")
    qe_scores = ensure_2d_array(qe_scores, "QE scores").flatten()
    correlation_dict = {}
    for i, element in enumerate(element_names):
        element_data = original_data[:, i]
        correlation, p_value = pearsonr(element_data, qe_scores)
        correlation_dict[element] = {'correlation': correlation, 'p_value': p_value}
        significance = "significant" if p_value < 0.05 else "not significant"
        direction = "positive correlation" if correlation > 0 else "negative correlation"
        magnitude = "strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.3 else "weak"
        logger.info(f"Element {element}: correlation={correlation:.4f} ({direction}, {magnitude}, {significance})")
    return correlation_dict


def visualize_element_qe_correlation(correlation_dict, output_path=None):
    logger.info("\n=== Visualizing Correlations Between Element Concentrations And QE Anomaly Scores ===")
    _setup_matplotlib_output_style(plt)
    elements = list(correlation_dict.keys())
    correlations = [correlation_dict[elem]['correlation'] for elem in elements]
    p_values = [correlation_dict[elem]['p_value'] for elem in elements]
    sorted_indices = np.argsort(correlations)[::-1]
    elements_sorted = [elements[i] for i in sorted_indices]
    correlations_sorted = [correlations[i] for i in sorted_indices]
    p_values_sorted = [p_values[i] for i in sorted_indices]
    n_elements = len(elements_sorted)
    fig_width = min(30, max(14, n_elements * 0.45))
    if n_elements >= 30:
        x_tick_fontsize = 22
        x_tick_rotation = 60
        x_tick_ha = "right"
    elif n_elements >= 20:
        x_tick_fontsize = 22
        x_tick_rotation = 45
        x_tick_ha = "right"
    else:
        x_tick_fontsize = 22
        x_tick_rotation = 0
        x_tick_ha = "center"
    plt.figure(figsize=(fig_width, 6), dpi=300)
    bars = plt.bar(elements_sorted, correlations_sorted, color="#4C78A8", width=0.68, alpha=0.9)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    for bar, p_val in zip(bars, p_values_sorted):
        if p_val < 0.05:
            bar.set_edgecolor('#2C3E50')
            bar.set_linewidth(1.5)
    plt.title('Element-QE Correlation', fontsize=24, pad=12)
    plt.xlabel('Element', fontsize=22)
    plt.ylabel('Pearson Correlation', fontsize=22)
    plt.xticks(rotation=x_tick_rotation, ha=x_tick_ha, fontsize=x_tick_fontsize)
    plt.yticks(fontsize=22)
    plt.grid(False)
    plt.margins(x=0.01)
    max_corr = max(np.array(correlations_sorted)) if correlations_sorted else 0
    plt.ylim(0, max_corr * 1.2 if max_corr > 0 else 1)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Element-QE correlation figure saved to {output_path}")
    else:
        plt.show()
    plt.close()


def apply_spatial_filter(qe_scores, coordinates, radius=None):
    if radius is None:
        radius = calculate_adaptive_spatial_radius(coordinates)
    filtered_scores = np.zeros_like(qe_scores, dtype=float)
    n_samples = len(qe_scores)
    for i in range(n_samples):
        distances = np.sqrt(np.sum((coordinates - coordinates[i]) ** 2, axis=1))
        neighbors = np.where((distances <= radius) & (distances > 0))[0]
        if len(neighbors) > 0:
            weights = np.exp(-(distances[neighbors] ** 2) / (2 * (radius / 2) ** 2))
            weighted_sum = np.sum(qe_scores[neighbors] * weights)
            filtered_scores[i] = weighted_sum / np.sum(weights)
        else:
            filtered_scores[i] = qe_scores[i]
    logger.info(f"QE score range after improved spatial filtering: {np.min(filtered_scores):.6f} - {np.max(filtered_scores):.6f}")
    return filtered_scores


def generate_arcgis_output(coordinates_df, qe_scores, output_path=None):
    if coordinates_df is None or len(coordinates_df) == 0:
        raise ValueError("Coordinate data are empty; ArcGIS output cannot be generated")
    if output_path:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
    arcgis_df = coordinates_df.copy()
    arcgis_df["QE_score"] = qe_scores
    retry_count = 0
    max_retries = 3
    while retry_count < max_retries:
        try:
            _localize_dataframe_headers(arcgis_df).to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Mineral-element ArcGIS file saved to {output_path}")
            break
        except PermissionError as e:
            retry_count += 1
            logger.warning(f"Failed to export the ArcGIS file due to a permission error. Retrying ({retry_count}/{max_retries}): {e}")
            if retry_count == max_retries:
                base_name, ext = os.path.splitext(output_path)
                backup_path = f"{base_name}_{int(time.time())}{ext}"
                _localize_dataframe_headers(arcgis_df).to_csv(backup_path, index=False, encoding='utf-8-sig')
                logger.warning(f"Saved with a fallback filename: {backup_path}")
                break
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to export the ArcGIS file: {e}")
            raise
    return arcgis_df


def plot_anomaly_map(raw_data, qe_scores, title, output_path=None):
    if output_path:
        output_dir = os.path.dirname(os.path.abspath(str(output_path)))
        output_filename = os.path.basename(str(output_path))
    else:
        output_dir = os.path.abspath(OUTPUT_DIR)
        output_filename = "mineral_anomaly_map.png"
    if generate_prediction_map_idw is None:
        raise ImportError("Failed to import the result-output plotting function `generate_prediction_map_idw`")
    qe_arr = np.asarray(qe_scores, dtype=float)
    if qe_arr.ndim != 1:
        qe_arr = qe_arr.reshape(-1)
    n = min(len(raw_data), int(qe_arr.shape[0]))
    if n <= 0:
        raise ValueError("Plotting failed because the input data are empty")
    probs = qe_arr[:n].tolist()
    high_potential_indices = []
    map_path = generate_prediction_map_idw(
        raw_data=raw_data.iloc[:n].copy(),
        all_results={"prediction_model": {"predictions": {"probabilities": probs, "high_potential_indices": high_potential_indices}}},
        output_dir=output_dir,
        logger=logger,
        processed_data=None,
        output_filename=output_filename,
        title=title,
    )
    if not map_path:
        raise RuntimeError("The result-output plotting routine failed and no figure was generated")
    logger.info(f"Mineral-element anomaly map saved to {map_path}")


def load_data(mineral_shp_path, sample_csv_path):
    if gpd is None or Point is None:
        raise ImportError("`geopandas`/`shapely` are not installed, so ROC analysis cannot be performed")
    if not os.path.exists(mineral_shp_path):
        raise FileNotFoundError(f"Mineral-occurrence shapefile does not exist: {mineral_shp_path}")
    if not os.path.exists(sample_csv_path):
        raise FileNotFoundError(f"Sample CSV file does not exist: {sample_csv_path}")
    try:
        mineral_gdf = gpd.read_file(mineral_shp_path)
        logger.info(f"Mineral-occurrence data loaded successfully: {len(mineral_gdf)} known occurrences")
        logger.info(f"Mineral-occurrence CRS: {mineral_gdf.crs}")
    except Exception as e:
        raise RuntimeError(f"Failed to read the mineral-occurrence shapefile: {str(e)}") from e
    try:
        sample_df = pd.read_csv(sample_csv_path)
        logger.info(f"Sample data loaded successfully: {len(sample_df)} sample points")
        logger.info(f"Sample-data columns: {sample_df.columns.tolist()}")
    except Exception as e:
        raise RuntimeError(f"Failed to read the sample CSV file: {str(e)}") from e
    x_col = "X"
    y_col = "Y"
    qe_score_col = "QE_score"
    required_cols = [x_col, y_col, qe_score_col]
    missing_cols = [col for col in required_cols if col not in sample_df.columns]
    if missing_cols:
        raise ValueError(f"Sample data are missing required columns: {missing_cols}. Please check the CSV headers.")
    sample_gdf = gpd.GeoDataFrame(
        sample_df,
        geometry=[Point(xy) for xy in zip(sample_df[x_col], sample_df[y_col])],
        crs=mineral_gdf.crs
    )
    sample_gdf = sample_gdf.rename(columns={qe_score_col: "qe_score"})
    logger.info("QE anomaly-score statistics:")
    logger.info(f"   - Minimum: {sample_gdf['qe_score'].min():.4f}")
    logger.info(f"   - Maximum: {sample_gdf['qe_score'].max():.4f}")
    logger.info(f"   - Mean: {sample_gdf['qe_score'].mean():.4f}")
    return mineral_gdf, sample_gdf


def generate_sample_labels(mineral_gdf, sample_gdf, buffer_dist=1000):
    mineral_buffers = mineral_gdf.geometry.buffer(buffer_dist)
    buffer_gdf = gpd.GeoDataFrame(geometry=mineral_buffers, crs=mineral_gdf.crs)
    union_buffer = buffer_gdf.geometry.union_all()
    sample_gdf["is_mineral"] = sample_gdf.geometry.within(union_buffer).astype(int)
    pos_count = sample_gdf["is_mineral"].sum()
    neg_count = len(sample_gdf) - pos_count
    logger.info("\nSample-label statistics:")
    logger.info(f"   - Positive samples (within mineral-buffer zones): {pos_count} ({pos_count / len(sample_gdf) * 100:.2f}%)")
    logger.info(f"   - Negative samples (outside mineral-buffer zones): {neg_count} ({neg_count / len(sample_gdf) * 100:.2f}%)")
    if pos_count == 0:
        logger.warning("\nWarning: no sample points were detected inside the mineral-buffer zones")
        logger.warning(f"   Recommendation: 1. verify CRS consistency; 2. adjust the buffer_dist parameter (current value: {buffer_dist})")
    labeled_data_path = os.path.join(OUTPUT_DIR, "labeled_sample_data.csv")
    _localize_dataframe_headers(sample_gdf).to_csv(labeled_data_path, index=False)
    logger.info(f"Labeled sample data saved to: {labeled_data_path}")
    sample_gdf["is_original"] = True
    return sample_gdf


def apply_smote_oversampling(sample_gdf):
    if SMOTE is None or gpd is None or Point is None:
        raise ImportError("`imblearn`/`geopandas`/`shapely` are not installed, so SMOTE oversampling cannot be performed")
    X = sample_gdf[["X", "Y", "qe_score"]].values
    y = sample_gdf["is_mineral"].values
    n_original_samples = len(sample_gdf)
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=["X", "Y", "qe_score"])
    resampled_df["is_mineral"] = y_resampled
    resampled_df["is_original"] = False
    resampled_df.loc[:n_original_samples - 1, "is_original"] = True
    resampled_gdf = gpd.GeoDataFrame(
        resampled_df,
        geometry=[Point(xy) for xy in zip(resampled_df["X"], resampled_df["Y"])],
        crs=sample_gdf.crs
    )
    logger.info("Sample statistics after SMOTE oversampling:")
    logger.info(f"   - Total samples: {len(resampled_gdf)} (original: {n_original_samples}, synthetic: {len(resampled_gdf)-n_original_samples})")
    logger.info(f"   - Positive samples: {resampled_gdf['is_mineral'].sum()}")
    logger.info(f"   - Negative samples: {len(resampled_gdf) - resampled_gdf['is_mineral'].sum()}")
    return resampled_gdf


def calculate_roc_auc_metrics(sample_gdf):
    if roc_curve is None or auc is None or precision_recall_curve is None or average_precision_score is None:
        raise ImportError("`sklearn.metrics` is not fully available, so ROC analysis cannot be performed")
    y_true = sample_gdf["is_mineral"].values
    y_score = sample_gdf["qe_score"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision,
        "y_true": y_true,
        "y_score": y_score,
    }


def plot_and_save_roc_curve(metrics, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    _setup_matplotlib_output_style(plt)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 8))
    plt.plot(
        metrics["fpr"],
        metrics["tpr"],
        color="#d62728",
        linewidth=2,
        label=f"AUC={metrics['roc_auc']:.3f}",
    )
    plt.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--", label="_nolegend_")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, labelpad=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, labelpad=10)
    plt.title("ROC Curve", fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=14, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.7, color="gray")
    roc_save_path = os.path.join(save_dir, "mineral_roc_curve.png")
    plt.savefig(roc_save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10, 8))
    plt.plot(
        metrics["recall"],
        metrics["precision"],
        color="green",
        linewidth=2,
        label=f"Precision-Recall Curve\n(AP = {metrics['average_precision']:.3f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12, labelpad=10)
    plt.ylabel("Precision", fontsize=12, labelpad=10)
    plt.title("Precision-Recall Curve Evaluation", fontsize=14, pad=20)
    plt.legend(loc="lower left", fontsize=14, frameon=True, shadow=True)
    plt.grid(True, linestyle="--", alpha=0.7, color="gray")
    pr_save_path = os.path.join(save_dir, "mineral_pr_curve.png")
    plt.savefig(pr_save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"ROC curve saved to: {roc_save_path}")
    logger.info(f"PR curve saved to: {pr_save_path}")


def find_optimal_threshold(metrics, method="youden"):
    if method == "youden":
        youden_index = metrics["tpr"] - metrics["fpr"]
        best_idx = np.argmax(youden_index)
        best_threshold = metrics["thresholds"][best_idx]
        best_score = youden_index[best_idx]
        logger.info(f"Best threshold (Youden Index): {best_threshold:.4f}, Youden score={best_score:.4f}")
        return best_threshold
    if method == "f1":
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = metrics["thresholds"][best_idx]
        best_score = f1_scores[best_idx]
        logger.info(f"Best threshold (F1 Score): {best_threshold:.4f}, F1={best_score:.4f}")
        return best_threshold
    raise ValueError("Unknown threshold-optimization method")


def run_mineral_qe_analysis_from_df(
    *,
    df,
    output_dir: str,
    cluster_results_dir: str,
    qe_dirname: str = "qe",
    x_col: str,
    y_col: str,
    sample_id_col: str | None = None,
    mineral_elements=None,
    threshold_percentile: float | None = None,
    enable_spatial_filter: bool = True,
    spatial_radius=None,
    mineral_shp_path: str | None = None,
    buffer_dist: float = 3000.0,
    use_smote: bool = True,
    llm: Any = None,
):
    base_output_dir = os.path.abspath(str(output_dir))
    out_dir = os.path.join(base_output_dir, str(qe_dirname))
    os.makedirs(out_dir, exist_ok=True)
    cluster_results_dir = os.path.abspath(str(cluster_results_dir))
    som_best_params_path = os.path.join(cluster_results_dir, "best_som_params.txt")
    som_full_model_path = os.path.join(cluster_results_dir, "som_model.pkl")
    if mineral_elements is None:
        mineral_elements = MINERAL_ELEMENTS
    mineral_elements = [str(x) for x in list(mineral_elements) if str(x).strip()]
    best_params = load_best_som_params_from_main(som_best_params_path)
    mineral_som_grid = best_params["map_size"]
    mineral_som_sigma = best_params["sigma"]
    mineral_som_lr = best_params["lr"]
    mineral_som_iter = int(best_params["iterations"] * 1.0)
    data = None
    elements = [str(x) for x in mineral_elements]
    coordinates = df[[x_col, y_col]].to_numpy(copy=True)
    sample_ids = df[sample_id_col].to_numpy(copy=True) if sample_id_col else None
    coords_df = df[[x_col, y_col]].copy()
    som_input_candidates = [
        os.path.join(cluster_results_dir, "som_input_X_scaled.npy"),
        os.path.join(cluster_results_dir, "preprocessing", "som_input_X_scaled.npy"),
    ]
    som_elements_candidates = [
        os.path.join(cluster_results_dir, "som_elements.pkl"),
        os.path.join(cluster_results_dir, "preprocessing", "som_elements.pkl"),
    ]
    for input_path, elem_path in zip(som_input_candidates, som_elements_candidates):
        if not (os.path.exists(input_path) and os.path.exists(elem_path)):
            continue
        try:
            som_input_full = np.asarray(np.load(input_path), dtype=float)
            som_elements_full = [str(x) for x in joblib.load(elem_path)]
            if som_input_full.ndim != 2 or som_input_full.shape[1] != len(som_elements_full):
                continue
            idx = [som_elements_full.index(str(e)) for e in elements]
            data = som_input_full[:, idx].copy()
            logger.info("QE analysis reused the SOM input matrix to remain strictly consistent with the preprocessing applied before SOM training")
            break
        except Exception:
            continue
    if data is None:
        data, coordinates, elements, sample_ids, coords_df = load_mineral_elements_data_from_df(df, x_col, y_col, mineral_elements, sample_id_col)
    som = train_mineral_som(
        data,
        mineral_elements,
        grid_size=mineral_som_grid,
        sigma=mineral_som_sigma,
        learning_rate=mineral_som_lr,
        iterations=mineral_som_iter,
        save_path=os.path.join(out_dir, "mineral_som_model.pkl"),
        full_som_path=som_full_model_path,
    )
    qe_scores = calculate_mineral_qe(data, som)
    if bool(enable_spatial_filter):
        qe_scores = apply_spatial_filter(qe_scores, coordinates, spatial_radius)
    output_plot = os.path.join(out_dir, "mineral_anomaly_map.png")
    output_stats = os.path.join(out_dir, "anomaly.csv")
    output_arcgis = os.path.join(out_dir, "arcgis_anomaly.csv")
    arcgis_df = generate_arcgis_output(coords_df, qe_scores, output_path=output_arcgis)
    correlation_dict = calculate_element_qe_correlation(data, qe_scores, elements)
    correlation_plot_path = os.path.join(out_dir, "element_qe_correlation.png")
    visualize_element_qe_correlation(correlation_dict, correlation_plot_path)
    correlation_df = pd.DataFrame.from_dict(correlation_dict, orient="index")
    correlation_df.index.name = "Element"
    correlation_csv_path = os.path.join(out_dir, "element_qe_correlation.csv")
    _localize_dataframe_headers(correlation_df).to_csv(correlation_csv_path, encoding="utf-8-sig")
    plot_df = coords_df.copy()
    possible_label_cols = ["Ore", "label", "target", "deposit", "label_encoded", "target_encoded", "labeled", "has_deposit", "is_deposit"]
    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        for col in df.columns:
            col_lower = str(col).lower()
            if any((lc.lower() in col_lower for lc in possible_label_cols)):
                label_col = col
                break
    if label_col is not None and label_col in df.columns and label_col not in plot_df.columns:
        try:
            plot_df[label_col] = df[label_col].to_numpy(copy=True)
        except Exception:
            plot_df[label_col] = df[label_col].values
    plot_anomaly_map(plot_df, qe_scores, PLOT_TITLE, output_plot)
    stats_df = pd.DataFrame({"Sample_ID": sample_ids if sample_ids is not None else range(len(qe_scores)), "Mineral_QE": qe_scores})
    _localize_dataframe_headers(stats_df).to_csv(output_stats, index=False, encoding="utf-8-sig")
    roc_result = {"enabled": False, "error": "", "source": "", "roc_curve": "", "pr_curve": ""}
    try:
        if label_col is not None and label_col in df.columns:
            y_true = pd.to_numeric(df[label_col], errors="coerce")
            valid_mask = np.isfinite(y_true.to_numpy(dtype=float))
            if bool(np.any(valid_mask)):
                y_true_valid = y_true.to_numpy(dtype=float)[valid_mask]
                qe_valid = np.asarray(qe_scores, dtype=float)[valid_mask]
                y_true_bin = (y_true_valid > 0).astype(int)
                class_count = np.unique(y_true_bin).size
                if class_count >= 2:
                    label_eval_df = pd.DataFrame({"qe_score": qe_valid, "is_mineral": y_true_bin})
                    metrics = calculate_roc_auc_metrics(label_eval_df)
                    plot_and_save_roc_curve(metrics, save_dir=out_dir)
                    roc_result = {
                        "enabled": True,
                        "error": "",
                        "source": "label_col",
                        "label_col": str(label_col),
                        "roc_auc": float(metrics.get("roc_auc", float("nan"))),
                        "average_precision": float(metrics.get("average_precision", float("nan"))),
                        "roc_curve": os.path.abspath(os.path.join(out_dir, "mineral_roc_curve.png")),
                        "pr_curve": os.path.abspath(os.path.join(out_dir, "mineral_pr_curve.png")),
                    }
                    logger.info(f"Generated ROC/PR curves using label column {label_col} and qe_scores")
                else:
                    roc_result = {
                        "enabled": False,
                        "error": f"Label column {label_col} contains only one class, so ROC cannot be computed",
                        "source": "label_col",
                        "label_col": str(label_col),
                        "roc_curve": "",
                        "pr_curve": "",
                    }
            else:
                roc_result = {
                    "enabled": False,
                        "error": f"Label column {label_col} contains no valid numeric values, so ROC cannot be computed",
                    "source": "label_col",
                    "label_col": str(label_col),
                    "roc_curve": "",
                    "pr_curve": "",
                }
    except Exception as e:
        roc_result = {"enabled": False, "error": str(e), "source": "label_col", "roc_curve": "", "pr_curve": ""}
    try:
        if mineral_shp_path and os.path.exists(str(mineral_shp_path)) and os.path.exists(output_arcgis):
            mineral_gdf, sample_gdf = load_data(str(mineral_shp_path), output_arcgis)
            sample_gdf = generate_sample_labels(mineral_gdf, sample_gdf, buffer_dist=float(buffer_dist))
            if int(sample_gdf["is_mineral"].sum()) > 0:
                if bool(use_smote):
                    sample_gdf = apply_smote_oversampling(sample_gdf)
                metrics = calculate_roc_auc_metrics(sample_gdf)
                plot_and_save_roc_curve(metrics, save_dir=out_dir)
                _ = find_optimal_threshold(metrics)
                if not bool(roc_result.get("enabled")):
                    roc_result = {
                        "enabled": True,
                        "error": "",
                        "source": "mineral_shp_buffer",
                        "roc_auc": float(metrics.get("roc_auc", float("nan"))),
                        "average_precision": float(metrics.get("average_precision", float("nan"))),
                        "roc_curve": os.path.abspath(os.path.join(out_dir, "mineral_roc_curve.png")),
                        "pr_curve": os.path.abspath(os.path.join(out_dir, "mineral_pr_curve.png")),
                    }
    except Exception as e:
        if not bool(roc_result.get("enabled")):
            roc_result = {"enabled": False, "error": str(e), "source": "mineral_shp_buffer", "roc_curve": "", "pr_curve": ""}
    return {
        "output_dir": out_dir,
        "cluster_results_dir": cluster_results_dir,
        "mineral_elements": mineral_elements,
        "best_params": best_params,
        "threshold": None,
        "output_plot": os.path.abspath(output_plot),
        "output_stats": os.path.abspath(output_stats),
        "output_arcgis": os.path.abspath(output_arcgis),
        "correlation_plot": os.path.abspath(correlation_plot_path),
        "correlation_csv": os.path.abspath(correlation_csv_path),
        "roc": roc_result,
        "arcgis_preview": arcgis_df.head(5).to_dict(orient="records") if isinstance(arcgis_df, pd.DataFrame) else None,
    }
