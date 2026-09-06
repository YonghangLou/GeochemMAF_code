import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shutil
import zipfile
import datetime
import warnings
import base64
from typing import Dict, List, Any, Optional
try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent
try:
    from .geology_expert_agent import GeologyExpertAgent
except Exception:
    try:
        from geology_expert_agent import GeologyExpertAgent
    except Exception:
        GeologyExpertAgent = None
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
try:
    from .utils.token_counter import TokenMonitor
except ImportError:
    from utils.token_counter import TokenMonitor
try:
    from .utils.token_reporter import TokenReporter
except ImportError:
    try:
        from utils.token_reporter import TokenReporter
    except ImportError:
        TokenReporter = None
try:
    from .utils.data_utils import atomic_write_csv as _atomic_write_csv
    from .utils.data_utils import atomic_write_json as _atomic_write_json
    from .utils.data_utils import atomic_write_markdown as _atomic_write_markdown
    from .utils.data_utils import atomic_write_text as _atomic_write_text
    from .utils.data_utils import localize_text as _localize_text
    from .utils.data_utils import resolve_output_language as _resolve_output_language
    from .utils.data_utils import setup_matplotlib_output_style as _setup_matplotlib_output_style
except ImportError:
    from utils.data_utils import atomic_write_csv as _atomic_write_csv
    from utils.data_utils import atomic_write_json as _atomic_write_json
    from utils.data_utils import atomic_write_markdown as _atomic_write_markdown
    from utils.data_utils import atomic_write_text as _atomic_write_text
    from utils.data_utils import localize_text as _localize_text
    from utils.data_utils import resolve_output_language as _resolve_output_language
    from utils.data_utils import setup_matplotlib_output_style as _setup_matplotlib_output_style
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')


def generate_prediction_map_idw(
    *,
    raw_data: pd.DataFrame,
    all_results: Dict[str, Any],
    output_dir: str,
    logger: Any,
    processed_data: pd.DataFrame=None,
    output_filename: str = "mineralization_potential_idw_map.png",
    title: str = "Mineralization Potential Prediction Map",
) -> str:
    lang = _resolve_output_language()
    _setup_matplotlib_output_style(plt)
    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        try:
            logger.error('Unable to generate the prediction map: raw_data is empty or is not a valid DataFrame')
        except Exception:
            pass
        return ''
    if 'prediction_model' not in all_results or all_results['prediction_model'] is None:
        try:
            logger.error('Unable to generate the prediction map: prediction results are missing')
        except Exception:
            pass
        return ''
    prediction_results = all_results['prediction_model']
    predictions = prediction_results.get('predictions', {})
    possible_x_cols = ['\u7ecf\u5ea6', 'longitude', 'LONGITUDE', 'Lon', 'lon']
    possible_y_cols = ['\u7eac\u5ea6', 'latitude', 'LATITUDE', 'Lat', 'lat']
    x_col = None
    y_col = None
    for col in raw_data.columns:
        if col in possible_x_cols:
            x_col = col
        elif col in possible_y_cols:
            y_col = col
    if x_col is None or y_col is None:
        numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            try:
                logger.warning(f'Standard coordinate columns were not found; using numeric columns {x_col} and {y_col} as coordinates')
            except Exception:
                pass
        else:
            try:
                logger.error('Unable to generate the prediction map: not enough coordinate or numeric columns were found')
            except Exception:
                pass
            return ''
    result_df = raw_data.copy()
    if 'probabilities' in predictions:
        if len(predictions['probabilities']) == len(result_df):
            result_df['prediction_probability'] = predictions['probabilities']
        elif processed_data is not None and len(predictions['probabilities']) == len(processed_data):
            result_df['prediction_probability'] = predictions['probabilities']
        else:
            probabilities = [None] * len(result_df)
            valid_len = min(len(predictions['probabilities']), len(result_df))
            probabilities[:valid_len] = predictions['probabilities'][:valid_len]
            result_df['prediction_probability'] = probabilities
    high_potential_indices = predictions.get('high_potential_indices', [])
    result_df['is_high_potential'] = 0
    valid_indices = [idx for idx in high_potential_indices if idx < len(result_df)]
    result_df.loc[valid_indices, 'is_high_potential'] = 1
    try:
        try:
            from utils.data_utils import normalize_coordinates as _normalize_coordinates
        except Exception:
            from .utils.data_utils import normalize_coordinates as _normalize_coordinates
        result_df, coord_meta = _normalize_coordinates(result_df, x_col=str(x_col), y_col=str(y_col), lon_col='Longitude', lat_col='Latitude')
        x_col = 'Longitude'
        y_col = 'Latitude'
        try:
            if isinstance(coord_meta, dict) and coord_meta.get("is_projected_input"):
                logger.info(f"Coordinates were normalized to longitude/latitude: {coord_meta}")
        except Exception:
            pass
    except Exception as e:
        try:
            logger.error(f"Unable to normalize coordinates to longitude/latitude, so the prediction map cannot be generated: {e}")
        except Exception:
            pass
        return ""

    valid_data = result_df.dropna(subset=['prediction_probability', x_col, y_col])
    if valid_data.empty:
        try:
            logger.warning('Unable to generate the prediction map: prediction-probability data is empty')
        except Exception:
            pass
        return ''
    _setup_matplotlib_output_style(plt)
    plt.figure(figsize=(12, 10))
    x_values = pd.to_numeric(valid_data[x_col], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(valid_data[y_col], errors="coerce").to_numpy(dtype=float)
    z_values = pd.to_numeric(valid_data['prediction_probability'], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(z_values)
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    z_values = z_values[finite_mask]
    if x_values.size == 0:
        try:
            logger.warning('Unable to generate the prediction map: the longitude/latitude columns do not contain valid numeric values')
        except Exception:
            pass
        return ""
    x_label = _localize_text('Longitude', lang=lang)
    y_label = _localize_text('Latitude', lang=lang)
    z_values = np.clip(z_values, 0.0, 1.0)
    vmin = 0.0
    vmax = 1.0
    cmap = plt.cm.get_cmap('jet')
    grid_size = 300
    x_min, x_max = (x_values.min(), x_values.max())
    y_min, y_max = (y_values.min(), y_values.max())
    x_range = x_max - x_min
    y_range = y_max - y_min
    pad_ratio = 0.0
    if x_range > 0:
        x_min -= x_range * pad_ratio
        x_max += x_range * pad_ratio
    if y_range > 0:
        y_min -= y_range * pad_ratio
        y_max += y_range * pad_ratio
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    xi, yi = np.meshgrid(xi, yi)
    try:
        logger.info(f'Running IDW interpolation with {len(x_values)} sample points')
        logger.info(f'Interpolation grid size: {grid_size}x{grid_size}')
    except Exception:
        pass
    try:
        from scipy.spatial import cKDTree
        points = np.column_stack([x_values.astype(float), y_values.astype(float)])
        values = z_values.astype(float)
        tree = cKDTree(points)
        grid_points = np.column_stack([xi.ravel().astype(float), yi.ravel().astype(float)])
        k = int(min(12, len(values)))
        power = 2.0
        eps = 1e-12
        distances, indices = tree.query(grid_points, k=k)
        if k == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        zi_flat = np.empty(shape=(grid_points.shape[0],), dtype=float)
        zero_mask = distances <= eps
        has_zero = zero_mask.any(axis=1)
        if np.any(has_zero):
            first_zero = zero_mask.argmax(axis=1)
            zi_flat[has_zero] = values[indices[has_zero, first_zero[has_zero]]]
        if np.any(~has_zero):
            d = distances[~has_zero]
            idx = indices[~has_zero]
            w = 1.0 / np.maximum(d, eps) ** power
            v = values[idx]
            zi_flat[~has_zero] = np.sum(w * v, axis=1) / np.sum(w, axis=1)
        zi = np.clip(zi_flat.reshape(xi.shape), 0.0, 1.0)
    except Exception as e:
        try:
            logger.error(f'IDW interpolation failed. Could not generate the prediction map: {str(e)}')
        except Exception:
            pass
        plt.close()
        return ''
    ax = plt.gca()
    ax.grid(False)
    im = ax.imshow(zi, extent=[x_min, x_max, y_min, y_max], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', alpha=0.9, aspect='equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    possible_label_cols = ['label', 'target', 'deposit', '\u77ff\u5e8a', '\u6807\u7b7e', 'label_encoded', 'target_encoded', 'labeled', 'has_deposit', 'is_deposit', 'Ore']
    label_col = None
    for col in possible_label_cols:
        if col in valid_data.columns:
            label_col = col
            break
    if label_col is None:
        for col in valid_data.columns:
            col_lower = col.lower()
            if any((lc.lower() in col_lower for lc in possible_label_cols)):
                label_col = col
                break
    if label_col is not None and label_col in valid_data.columns:
        try:
            deposit_samples = valid_data[valid_data[label_col] == 1]
            if len(deposit_samples) == 0:
                try:
                    deposit_samples = valid_data[pd.to_numeric(valid_data[label_col], errors='coerce') == 1]
                except Exception:
                    pass
            if len(deposit_samples) > 0:
                sample_x = deposit_samples[x_col].values
                sample_y = deposit_samples[y_col].values
                ax.scatter(
                    sample_x,
                    sample_y,
                    color='cyan',
                    s=50,
                    edgecolor='black',
                    linewidth=2,
                    alpha=0.8,
                    label=_localize_text('Known Deposit', lang=lang),
                )
        except Exception:
            pass
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=10,
            loc='lower left',
            scatterpoints=1,
            handlelength=0.9,
            handletextpad=0.35,
            borderpad=0.35,
            labelspacing=0.3,
            markerscale=0.9,
        )
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, fraction=0.06, aspect=10)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_label(_localize_text('Mineral Potential Probability', lang=lang), fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    map_path = os.path.join(str(output_dir), str(output_filename))
    try:
        plt.savefig(map_path, dpi=300, format='png', bbox_inches='tight', pil_kwargs={'optimize': True, 'compress_level': 9})
    except Exception as e:
        try:
            logger.exception(f'Failed to save the mineralization-potential prediction map: {str(e)}')
        except Exception:
            pass
        return ''
    finally:
        plt.close()
    try:
        logger.info(f'Mineralization-potential prediction map saved to: {map_path}')
    except Exception:
        pass
    return map_path
class ResultOutputAgent(BaseAgent):
    CAPABILITIES = [
        'Comprehensive reporting: aggregate end-to-end workflow results into Markdown reports',
        'Visualization: generate mineral-potential prediction maps with IDW interpolation',
        'Result export: write prediction tables to CSV with probabilities/labels/high-potential flags',
        'Full-run archiving: export serializable end-to-end results as JSON',
        'Delivery packaging: package files and output folders into a zip archive',
    ]
    def __init__(self, output_dir: str='./output', llm=None):
        role_description = 'Responsible for comprehensive geological analysis reports and visualizations. Integrate the results from all agents into professional, clear, understandable reports, with useful geological interpretations and exploration recommendations.'
        super().__init__('ResultOutputAgent', role_description, llm)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if hasattr(self.llm, 'logger') and self.llm.logger is not None:
            self.logger = self.llm.logger
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(self.agent_name)
        if self.logger.level and self.logger.level > logging.INFO:
            self.logger.setLevel(logging.INFO)
        self.logger.propagate = True
        self._register_skills()

    def _display_model_name(self, model_key: Any) -> str:
        if model_key is None:
            return ''
        raw = str(model_key).strip()
        low = raw.lower()
        if low in {"som", "self organizing map", "self-organizing map", "selforganizingmap", '\u81ea\u7ec4\u7ec7\u6620\u5c04', '\u81ea\u7ec4\u7ec7\u6620\u5c04\u795e\u7ecf\u7f51\u7edc'}:
            return "SOM"
        return raw

    def _register_skills(self) -> None:
        reg = getattr(self, "skills", None)
        if reg is None or SkillSpec is None:
            return
        return

    def export_prediction_table(
        self,
        *,
        raw_data: pd.DataFrame,
        all_results: Dict[str, Any],
        output_dir: str,
        processed_data: Optional[pd.DataFrame] = None,
        output_filename: str = "prediction_results.csv",
    ) -> str:
        if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
            return ""
        if not isinstance(all_results, dict):
            return ""
        prediction_results = all_results.get("prediction_model")
        if not isinstance(prediction_results, dict):
            return ""
        predictions = prediction_results.get("predictions")
        if not isinstance(predictions, dict):
            return ""

        result_df = raw_data.copy()
        probs = predictions.get("probabilities")
        if isinstance(probs, list):
            if len(probs) == len(result_df):
                result_df["prediction_probability"] = probs
            elif processed_data is not None and len(probs) == len(processed_data):
                result_df["prediction_probability"] = probs
            else:
                padded: List[Any] = [None] * len(result_df)
                valid_len = min(len(probs), len(result_df))
                padded[:valid_len] = probs[:valid_len]
                result_df["prediction_probability"] = padded

        preds = predictions.get("predictions")
        if isinstance(preds, list):
            if len(preds) == len(result_df):
                result_df["prediction_label"] = preds
            elif processed_data is not None and len(preds) == len(processed_data):
                result_df["prediction_label"] = preds
            else:
                padded: List[Any] = [None] * len(result_df)
                valid_len = min(len(preds), len(result_df))
                padded[:valid_len] = preds[:valid_len]
                result_df["prediction_label"] = padded

        high_potential_indices = predictions.get("high_potential_indices", [])
        result_df["is_high_potential"] = 0
        if isinstance(high_potential_indices, list):
            valid_indices = [int(idx) for idx in high_potential_indices if isinstance(idx, (int, float)) and int(idx) < len(result_df)]
            if valid_indices:
                result_df.loc[valid_indices, "is_high_potential"] = 1

        out_dir = os.path.abspath(str(output_dir or ""))
        if not out_dir:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, str(output_filename or "prediction_results.csv"))
        _atomic_write_csv(result_df, out_path, index=False, encoding="utf-8-sig")
        return out_path

    def export_all_results_json(self, *, all_results: Dict[str, Any], output_dir: str, output_filename: str = "all_results.json") -> str:
        def _sanitize(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, tuple):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, pd.DataFrame):
                try:
                    return obj.to_dict(orient="list")
                except Exception:
                    return {"_type": "DataFrame", "shape": list(obj.shape)}
            if isinstance(obj, np.ndarray):
                try:
                    return obj.tolist()
                except Exception:
                    return {"_type": "ndarray", "shape": list(obj.shape)}
            return str(obj)

        out_dir = os.path.abspath(str(output_dir or ""))
        if not out_dir:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, str(output_filename or "all_results.json"))
        _atomic_write_json(_sanitize(all_results), out_path)
        return out_path

    def package_delivery(self, *, paths: Any = None, output_dir: str, zip_name: str = "delivery.zip") -> str:
        out_dir = os.path.abspath(str(output_dir or ""))
        if not out_dir:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        zip_path = os.path.join(out_dir, str(zip_name or "delivery.zip"))
        candidates: List[str] = []
        if isinstance(paths, list):
            for p in paths:
                if not p:
                    continue
                candidates.append(os.path.abspath(str(p)))
        elif isinstance(paths, str) and paths.strip():
            candidates.append(os.path.abspath(paths.strip()))
        else:
            candidates.append(out_dir)

        def _add_path(zf: zipfile.ZipFile, p: str, base: str) -> None:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        rel = os.path.relpath(fp, start=base)
                        zf.write(fp, arcname=rel)
            elif os.path.isfile(p):
                rel = os.path.relpath(p, start=base)
                zf.write(p, arcname=rel)

        base = out_dir
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in candidates:
                if not os.path.exists(p):
                    continue
                base = os.path.dirname(p) if os.path.isfile(p) else p
                _add_path(zf, p, base=base)
        return zip_path

    def _image_to_data_uri(self, image_path: str) -> Optional[str]:
        if not image_path or not isinstance(image_path, str):
            return None
        abs_path = os.path.abspath(image_path)
        if not os.path.exists(abs_path):
            return None
        try:
            with open(abs_path, "rb") as f:
                data = f.read()
        except Exception:
            return None
        if not data:
            return None
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _analyze_prediction_map_with_llm(self, map_path: str, all_results: Dict[str, Any]) -> str:
        if not getattr(self, "llm", None):
            return ""
        data_uri = self._image_to_data_uri(map_path)
        if not data_uri:
            return ""
        prediction_results = all_results.get("prediction_model", {}) if isinstance(all_results, dict) else {}
        predictions = prediction_results.get("predictions", {}) if isinstance(prediction_results, dict) else {}
        high_potential_count = 0
        if isinstance(predictions, dict):
            high_potential_count = int(predictions.get("high_potential_count", 0) or 0)
            if high_potential_count == 0 and isinstance(predictions.get("high_potential_indices"), list):
                high_potential_count = len(predictions.get("high_potential_indices") or [])
        total_samples = 0
        try:
            total_samples = int(all_results.get("data_preprocessing", {}).get("basic_stats", {}).get("total_samples", 0) or 0)
        except Exception:
            total_samples = 0
        ctx = f"Total sample count={total_samples}; high-potential sample count={high_potential_count}."
        prompt_text = (
            'You are a senior geochemical prospectivity-mapping expert. Interpret only the provided mineral-potential prediction map (probability heatmap) and return Markdown that can be inserted directly into a report.\n'
            f"Context: {ctx}\n\n"
            'Writing requirements:\n'
            '1) The output must include the following section titles in order:\n'
            '   - Overall map characteristics\n'
            '   - Distribution of high-value hotspots and low-value areas\n'
            '   - Relationship to known deposits/sample points, if such marks are visible\n'
            '   - Uncertainty and possible sources of misinterpretation\n'
            '   - Priority exploration zones and next-step validation suggestions\n'
            '2) Be as specific as possible: describe hotspot count, relative orientation/trend, and whether the pattern is belt-like, clustered, or isolated. Avoid vague descriptions.\n'
            "3) Do not invent map legends or coordinate information. If something cannot be determined from the map, explicitly state 'uncertain'.\n"
        )
        try:
            from langchain_core.messages import HumanMessage

            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
            )
            llm = self.llm
            resp = llm.invoke([msg])
            text = getattr(resp, "content", resp)
            return str(text or "").strip()
        except Exception as e:
            self.logger.warning(f"Prediction-map interpretation failed: {e}")
            return ""

    def _interpret_youden_targets(self, item: Dict[str, Any], all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate geological interpretation; report assembly does not author it."""
        config = all_results.get("config") or {}
        if not bool(config.get("geology_expert_enabled", True)):
            return {"status": "skipped", "reason": "geology_expert_disabled"}
        result = item["interpretation"]
        if result.get("status") in {"success", "failed", "skipped"}:
            return result
        try:
            if GeologyExpertAgent is None:
                raise RuntimeError('The Geology Expert Agent is unavailable')
            expert = getattr(self, "_target_geology_expert", None)
            if expert is None:
                expert = GeologyExpertAgent(
                    output_dir=os.path.dirname(str(item["map_path"])),
                    llm=getattr(self, "llm", None),
                )
                self._target_geology_expert = expert
            return expert.interpret_youden_targets(item=item, all_results=all_results)
        except Exception as e:
            result.update(status="failed", reason=str(e))
            self.logger.warning(f"Geology expert target interpretation failed: {e}")
            return result

    def _collect_target_delineations(self, all_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        prediction_results = all_results.get("prediction_model", {}) if isinstance(all_results, dict) else {}
        if not isinstance(prediction_results, dict):
            return entries
        seen: set[str] = set()

        def _append_entry(tag: str, obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            delineation = obj.get("target_delineation")
            if not isinstance(delineation, dict):
                return
            key = str(delineation.get("map_path") or f"{tag}:{len(entries)}")
            if key in seen:
                return
            seen.add(key)
            item = dict(delineation)
            item["interpretation"] = delineation.setdefault("interpretation", {})
            item["mineral_elements"] = obj.get("mineral_elements", [])
            item["run_tag"] = tag
            item["output_plot"] = obj.get("output_plot")
            entries.append(item)

        _append_entry("default", prediction_results.get("mineral_qe_analysis"))
        som_cluster_analysis = prediction_results.get("som_cluster_analysis")

        def _append_from_container(container: Any, run_prefix: str = "") -> None:
            if not isinstance(container, dict):
                return
            for tag in ["all_elements", "filtered_elements"]:
                run_obj = container.get(tag)
                if not isinstance(run_obj, dict):
                    continue
                run_tag = f"{run_prefix}{tag}" if run_prefix else tag
                _append_entry(run_tag, run_obj.get("qe_analysis"))

        if isinstance(som_cluster_analysis, dict):
            _append_from_container(som_cluster_analysis)
            dual_source_runs = som_cluster_analysis.get("dual_source_runs")
            if isinstance(dual_source_runs, dict):
                for source_name, run_container in dual_source_runs.items():
                    source_prefix = str(source_name).strip()
                    if source_prefix:
                        source_prefix = f"{source_prefix}_"
                    _append_from_container(run_container, run_prefix=source_prefix)
        return entries

    def _build_results_geological_interpretation_block(self, all_results: Dict[str, Any], reports_dir: str) -> str:
        lines: List[str] = []
        geo_interp = all_results.get("geological_interpretation")
        if geo_interp is None:
            geo_interp = all_results.get("geology_expert_results", {}).get("geological_interpretation", {})
        prediction_results = all_results.get("prediction_model", {}) if isinstance(all_results, dict) else {}
        visualizations = all_results.get("visualizations") if isinstance(all_results, dict) else None
        map_analysis = visualizations.get("prediction_map_analysis") if isinstance(visualizations, dict) else None

        lines.append('### Geological Interpretation Of Results')
        lines.append("")
        summary_text = ""
        confidence_text = ""
        mineralization_text = ""
        if isinstance(geo_interp, dict):
            summary_text = str(geo_interp.get("summary") or "").strip()
            confidence_text = str(geo_interp.get("confidence_level") or "").strip()
            m_types = geo_interp.get("mineralization_types")
            if isinstance(m_types, list) and m_types:
                mineralization_text = ", ".join([str(x) for x in m_types if str(x).strip()][:6])
            elif isinstance(m_types, dict):
                primary_type = str(m_types.get("primary_type") or "").strip()
                if primary_type:
                    mineralization_text = primary_type
        if summary_text:
            lines.append(f"The integrated geochemical response of the study area indicates: {summary_text}")
            lines.append("")
        if confidence_text:
            lines.append(f"The current geological-interpretation confidence level is assessed as '{confidence_text}'.")
            lines.append("")
        if mineralization_text:
            lines.append(f"The recognized mineralization types are dominated by {mineralization_text}, suggesting a coupling relationship ")
            lines.append("")
        if map_analysis:
            lines.append('Spatial interpretation based on the mineral-potential map indicates:')
            lines.append("")
            lines.append(str(map_analysis).strip())
            lines.append("")
        geo_results = all_results.get("geology_expert_results", {}) if isinstance(all_results, dict) else {}
        key_element_analysis = geo_results.get("key_element_analysis", {}) if isinstance(geo_results, dict) else {}
        if isinstance(key_element_analysis, dict):
            relation_lines = key_element_analysis.get("relationship_summary") or []
            if isinstance(relation_lines, list) and relation_lines:
                lines.append('Key observations on the relationship between selected elements and the target deposit are as follows:')
                lines.append("")
                for item in relation_lines[:8]:
                    text = str(item).strip()
                    if text:
                        lines.append(f"- {text}")
                lines.append("")

        som_interp = prediction_results.get("som_geology_interpretation") if isinstance(prediction_results, dict) else None
        som_lines: List[str] = []
        if isinstance(som_interp, dict):
            for tag in ["all_elements", "filtered_elements"]:
                run_obj = som_interp.get(tag)
                if isinstance(run_obj, dict):
                    txt = str(run_obj.get("text") or "").strip()
                    if txt:
                        som_lines.append(txt)
        if som_lines:
            lines.append('Key geological observations from the SOM clustering results are as follows:')
            lines.append("")
            for txt in som_lines[:2]:
                lines.append(txt)
                lines.append("")

        delineations = self._collect_target_delineations(all_results)
        success_items = [x for x in delineations if x.get("enabled") and x.get("map_path")]
        if success_items:
            lines.append('#### Target Delineation Using the Maximum-Youden Threshold')
            lines.append("")
            lines.append('The ROC curve is calculated from valid original full-region labels and QE scores. The threshold maximizing TPR minus FPR delineates targets; samples with QE at or above the threshold are classified as targets.')
            lines.append("")
            lines.append('This delineation is retrospective because full-region labels participate in threshold selection; it is not independent held-out test performance. Binary classes are displayed using nearest neighbors.')
            lines.append("")
            for item in success_items:
                run_tag = str(item.get("run_tag") or "run")
                lines.append(f"- {run_tag}: QE threshold={float(item['threshold']):.6g}, Youden={float(item['youden_index']):.4f}, sensitivity={float(item['sensitivity']):.4f}, specificity={float(item['specificity']):.4f}.")
                for field, label in [("map_path", 'Prospecting Target Map'), ("source_csv", 'Per-Sample Classification Data'), ("metrics_json", 'Threshold and Evaluation Metrics')]:
                    path = item.get(field)
                    if path:
                        rel_path = os.path.relpath(str(path), reports_dir).replace("\\", "/")
                        lines.append(f"- {run_tag} {label}: [{label}](<{rel_path}>)")
                interpretation = self._interpret_youden_targets(item, all_results)
                if interpretation.get("status") == "success":
                    lines.extend(["", f"##### {run_tag} Geological Interpretation of Targets", "", interpretation["markdown"], ""])
                elif interpretation.get("status") == "failed":
                    lines.extend(["", f"{run_tag}: LLM interpretation was not completed; the target map and numerical metrics above remain valid outputs.", ""])
            lines.append("")

        text = "\n".join(lines).strip()
        return text + "\n" if text else ""

    def _inject_into_results_section(self, report_content: str, section_md: str) -> str:
        text = str(report_content or "").rstrip() + "\n"
        block = str(section_md or "").strip()
        if not block:
            return text
        if block in text:
            return text
        import re

        for hdr in ['\\n##\\s*3\\.\\s*Results\\b', '\\n##\\s*3\\.\\s*Results And Analysis\\b', '\\n##\\s*5\\.\\s*Mineral Potential Prediction\\b', '\\n##\\s*Results\\b']:
            m = re.search(hdr, text)
            if not m:
                continue
            next_hdr = re.search(r"\n##\s+", text[m.end() :])
            if next_hdr:
                insert_at = int(m.end() + next_hdr.start())
                return text[:insert_at].rstrip() + "\n\n" + block + "\n\n" + text[insert_at:].lstrip()
            return text.rstrip() + "\n\n" + block + "\n"
        return text.rstrip() + "\n\n" + block + "\n"

    def run(self, state: dict) -> dict:
        self.logger.info('Generating output artifacts...')
        try:
            if 'processing_history' not in state:
                state['processing_history'] = []
            if 'errors' not in state:
                state['errors'] = []
            processed_data = state.get('processed_data', state.get('preprocessed_data'))
            if processed_data is None:
                raise ValueError('Processed data was not found. Run the data-science expert agent first.')
            raw_data = state.get('data')
            if raw_data is None:
                raise ValueError('Raw input data was not found.')
            geology_bundle = state.get('geology_expert_results')
            if not isinstance(geology_bundle, dict):
                geology_bundle = {}
            geology_expert_enabled = bool((state.get('config') or {}).get('geology_expert_enabled', True))
            if geology_expert_enabled:
                geology_bundle.setdefault('ca_results', state.get('ca_results', {}))
                geology_bundle.setdefault('key_element_analysis', state.get('key_element_analysis', {}))
                geology_bundle.setdefault('feature_analysis', state.get('feature_analysis_results', {}))
                geology_bundle.setdefault('element_combos', state.get('element_combos', {}))
                geology_bundle.setdefault('geological_interpretation', state.get('geology_interpretation', {}))
            all_results = {
                'data_preprocessing': state.get('preprocessing_results', {}),
                'prediction_model': state.get('prediction_results', {}),
                'geology_expert_results': geology_bundle,
                'basic_stats': {'total_samples': len(processed_data)},
                'config': state.get('config'),
                'model_input_metadata': state.get('model_input_metadata', {}),
                'target_deposit_type': state.get('target_deposit_type'),
                'study_area_location': state.get('study_area_location'),
            }
            output_paths = self.output(all_results, processed_data, raw_data)
            state['output_results'] = output_paths
            state['processing_history'].append(f'{self.agent_name}: result output completed')
            state['next_agent'] = None
            self.logger.info('Result output completed')
            self.logger.info('Output file paths:')
            for key, path in output_paths.items():
                if path:
                    self.logger.info(f'  - {key}: {path}')
        except Exception as e:
            self.logger.error(f'Result output failed: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'{self.agent_name}: {str(e)}')
            state['next_agent'] = 'final_decision'
        return state
    def _ensure_output_dirs(self) -> Dict[str, str]:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        reports_dir = os.path.join(self.output_dir, 'reports')
        images_dir = os.path.join(reports_dir, 'images')
        data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        return {'reports_dir': reports_dir, 'images_dir': images_dir, 'data_dir': data_dir}
    def _resolve_existing_image_path(self, value: str) -> str:
        text = str(value or '').strip().strip('"').strip("'")
        if not text:
            return ''
        candidates = [text]
        if not os.path.isabs(text):
            candidates.append(os.path.join(self.output_dir, text))
            candidates.append(os.path.join(self.output_dir, 'reports', text))
        valid_ext = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.svg', '.webp'}
        for c in candidates:
            try:
                p = os.path.abspath(c)
            except Exception:
                p = c
            ext = os.path.splitext(str(p))[1].lower()
            if ext in valid_ext and os.path.isfile(p):
                return p
        return ''
    def _extract_image_paths(self, obj: Any, limit: int=2000) -> List[str]:
        if limit <= 0:
            return []
        found: List[str] = []
        stack: List[Any] = [obj]
        while stack and len(found) < limit:
            cur = stack.pop()
            if isinstance(cur, str):
                p = self._resolve_existing_image_path(cur)
                if p:
                    found.append(p)
            elif isinstance(cur, dict):
                stack.extend(list(cur.values()))
            elif isinstance(cur, (list, tuple, set)):
                stack.extend(list(cur))
        unique: List[str] = []
        seen = set()
        for p in found:
            if p not in seen:
                unique.append(p)
                seen.add(p)
        return unique
    def _sync_images_folder(self, all_results: Dict[str, Any]) -> List[str]:
        dirs = self._ensure_output_dirs()
        images_dir = dirs['images_dir']
        image_paths = self._extract_image_paths(all_results)
        copied: List[str] = []
        for src in image_paths:
            base = os.path.basename(src)
            if not base:
                continue
            dst = os.path.join(images_dir, base)
            if os.path.abspath(src) == os.path.abspath(dst):
                copied.append(dst)
                continue
            root, ext = os.path.splitext(base)
            idx = 1
            while os.path.exists(dst):
                try:
                    if os.path.samefile(src, dst):
                        break
                except Exception:
                    pass
                dst = os.path.join(images_dir, f"{root}_{idx}{ext}")
                idx += 1
            try:
                shutil.copy2(src, dst)
                copied.append(dst)
            except Exception:
                continue
        return copied
    def output(self, all_results: Dict[str, Any], data: pd.DataFrame, raw_data: pd.DataFrame=None) -> Dict[str, str]:
        self.logger.info('Result-output agent started.')
        output_paths = {}
        try:
            dirs = self._ensure_output_dirs()
            output_paths['data_dir'] = dirs['data_dir']
            output_paths['images_dir'] = dirs['images_dir']
            if not isinstance(all_results, dict):
                all_results = {}
            visualizations = all_results.get("visualizations")
            if not isinstance(visualizations, dict):
                visualizations = {}
                all_results["visualizations"] = visualizations
            output_paths['prediction_map'] = ''
            try:
                report_path = self._output_comprehensive_report(all_results)
                output_paths['comprehensive_report'] = report_path
            except Exception as e:
                self.logger.error(f'Error while generating the comprehensive report: {str(e)}')
                output_paths['comprehensive_report'] = ''
            try:
                feature_doc_path = self._output_feature_analysis_selection_doc(all_results)
                output_paths['feature_analysis_selection_doc'] = feature_doc_path
            except Exception as e:
                self.logger.error(f'Error while generating the feature analysis and selection document: {str(e)}')
                output_paths['feature_analysis_selection_doc'] = ''
            if bool((all_results.get('config') or {}).get('geology_expert_enabled', True)):
                try:
                    element_doc_path = self._output_target_element_selection_doc(all_results)
                    output_paths['target_element_selection_doc'] = element_doc_path
                except Exception as e:
                    self.logger.error(f'Error while generating the target-related element document: {str(e)}')
                    output_paths['target_element_selection_doc'] = ''
            else:
                output_paths['target_element_selection_doc'] = ''
            try:
                predictions_path = self._output_predictions_data(data, all_results)
                output_paths['predictions_data'] = predictions_path
            except Exception as e:
                self.logger.error(f'Error while exporting prediction results: {str(e)}')
                output_paths['predictions_data'] = ''
            try:
                anomalies_path = self._output_anomaly_analysis(all_results)
                output_paths['anomaly_analysis'] = anomalies_path
            except Exception as e:
                self.logger.error(f'Error while exporting anomaly-analysis results: {str(e)}')
                output_paths['anomaly_analysis'] = ''
            try:
                features_path = self._output_feature_importance(all_results)
                output_paths['feature_importance'] = features_path
            except Exception as e:
                self.logger.error(f'Error while exporting feature importance: {str(e)}')
                output_paths['feature_importance'] = ''
            try:
                json_path = self._output_json_results(all_results)
                output_paths['json_results'] = json_path
            except Exception as e:
                self.logger.error(f'Error while exporting JSON results: {str(e)}')
                output_paths['json_results'] = ''
            try:
                token_report = TokenMonitor().get_report()
                reports_dir = os.path.join(self.output_dir, 'reports')
                os.makedirs(reports_dir, exist_ok=True)
                output_paths['token_usage'] = ''
                try:
                    if TokenReporter:
                        try:
                            base_dir = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
                        except Exception:
                            base_dir = os.path.expanduser("~")
                        log_file = os.path.join(base_dir, "GAI-MAS", "logs", "agent_system.log")
                        reporter = TokenReporter(self.output_dir, log_file)
                        report_path = reporter.generate_report(token_report)
                        if report_path:
                            self.logger.info(f'Detailed token analysis report generated: {report_path}')
                    else:
                        self.logger.warning('TokenReporter is not imported. Skipping the detailed token analysis report.')
                except Exception as e:
                    self.logger.warning(f'Could not generate the detailed token analysis report: {e}')
            except Exception as e:
                self.logger.error(f'Error while exporting the token report: {str(e)}')
                output_paths['token_usage'] = ''
            self.logger.info(f'All results were exported to: {self.output_dir}')
        except Exception as e:
            self.logger.error(f'Error while exporting results: {str(e)}')
            output_paths = {'comprehensive_report': '', 'predictions_data': '', 'anomaly_analysis': '', 'feature_importance': '', 'json_results': '', 'token_usage': '', 'target_element_selection_doc': ''}
        return output_paths
    def _output_comprehensive_report(self, all_results: Dict[str, Any]) -> str:
        try:
            if self.llm:
                self.logger.info('Generating the final comprehensive report with the LLM...')
                report_content = self._generate_final_report_with_llm(all_results)
            else:
                report_content = self.generate_report(all_results)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            report_path = os.path.join(reports_dir, 'comprehensive_report.md')
            visualizations = all_results.get("visualizations") if isinstance(all_results, dict) else None
            if isinstance(visualizations, dict):
                map_path = visualizations.get("prediction_map_path")
                map_analysis = visualizations.get("prediction_map_analysis")
            else:
                map_path = None
                map_analysis = None
            if (map_path or map_analysis) and ('LLM-based geological interpretation' not in str(report_content or "")) and ('prediction-map interpretation' not in str(report_content or "")):
                block_lines: List[str] = []
                block_lines.append('## LLM-Based Geological Interpretation From Key-Element Spatial Patterns And The Mineralization-Potential Prediction Map')
                block_lines.append("")
                if map_path:
                    try:
                        rel = os.path.relpath(str(map_path), reports_dir).replace("\\", "/")
                    except Exception:
                        rel = str(map_path)
                    block_lines.append(f"![Mineralization-Potential Prediction Map]({rel})")
                    block_lines.append("")
                if map_analysis:
                    block_lines.append(str(map_analysis).strip())
                    block_lines.append("")
                import re

                section_md = "\n".join(block_lines).rstrip() + "\n"
                text = str(report_content or "").rstrip() + "\n"
                m = re.search('\\n##\\s+(Conclusions And Recommendations|Conclusions And Outlook|Conclusions)\\b', text)
                if m:
                    insert_at = int(m.start())
                    report_content = text[:insert_at].rstrip() + "\n\n" + section_md + "\n" + text[insert_at:].lstrip()
                else:
                    m2 = re.search('\\n##\\s+(Mineralization Prediction Analysis|Mineralization Potential Prediction)\\b', text)
                    if m2:
                        next_hdr = re.search(r"\n##\s+", text[m2.end() :])
                        if next_hdr:
                            insert_at = int(m2.end() + next_hdr.start())
                            report_content = text[:insert_at].rstrip() + "\n\n" + section_md + "\n" + text[insert_at:].lstrip()
                        else:
                            report_content = text.rstrip() + "\n\n" + section_md + "\n"
                    else:
                        report_content = text.rstrip() + "\n\n" + section_md + "\n"
            appendix_lines: List[str] = []
            spatial_coupling = None
            coupling_imgs: List[str] = []
            if isinstance(visualizations, dict):
                spatial_coupling = visualizations.get("spatial_potential_coupling_analysis")
                imgs_obj = visualizations.get("spatial_potential_coupling_images")
                if isinstance(imgs_obj, list):
                    coupling_imgs = [str(p) for p in imgs_obj if p]
            if spatial_coupling:
                appendix_lines.append('## Appendix: Coupled Interpretation Of Key-Element Spatial Patterns And Mineralization Potential (Geology Expert / LLM)')
                appendix_lines.append("")
                for p in coupling_imgs:
                    try:
                        relp = os.path.relpath(str(p), reports_dir).replace("\\", "/")
                    except Exception:
                        relp = str(p)
                    try:
                        base = os.path.basename(str(p))
                        if "_spatial_distribution" in base:
                            elem = base.split("_spatial_distribution")[0].strip()
                        else:
                            elem = base.split("_spatial_anomaly")[0].strip()
                        title = f"{elem} Spatial Distribution Map" if elem else 'Key-Element Spatial Distribution Map'
                    except Exception:
                        title = 'Key-Element Spatial Distribution Map'
                    appendix_lines.append(f"![{title}]({relp})")
                    appendix_lines.append("")
                appendix_lines.append(str(spatial_coupling).strip())
                appendix_lines.append("")
            if appendix_lines:
                report_content = str(report_content or "").rstrip() + "\n\n" + "\n".join(appendix_lines).rstrip() + "\n"
            geology_block = self._build_results_geological_interpretation_block(all_results, reports_dir)
            if geology_block.strip():
                report_content = self._inject_into_results_section(report_content, geology_block)
            _atomic_write_text(report_path, report_content, encoding='utf-8')
            self.logger.info(f'Comprehensive report saved to: {report_path}')
            return report_path
        except Exception as e:
            self.logger.error(f'Error while generating the comprehensive report: {str(e)}')
            error_report = '# Comprehensive Report For Geochemical Mineralization-Potential Prediction\n\n'
            error_report += '## Report Generation Error\n'
            error_report += f'- Error message: {str(e)}\n'
            error_report += '- Please check the data format and processing workflow.\n\n'
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            error_report_path = os.path.join(reports_dir, 'comprehensive_report_error.md')
            _atomic_write_text(error_report_path, error_report, encoding='utf-8')
            self.logger.info(f'Error report saved to: {error_report_path}')
            return error_report_path
    def _output_feature_analysis_selection_doc(self, all_results: Dict[str, Any]) -> str:
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        doc_path = os.path.join(reports_dir, 'feature_analysis_and_selection.md')
        geo = all_results.get('geology_expert_results') if isinstance(all_results, dict) else None
        if not isinstance(geo, dict):
            geo = {}
        feature_analysis = geo.get('feature_analysis', {})
        key_element_analysis = geo.get('key_element_analysis', {})
        if not isinstance(key_element_analysis, dict):
            key_element_analysis = {}
        ca_results = key_element_analysis.get('element_statistics', {})
        if not isinstance(ca_results, dict) or not ca_results:
            ca_results = geo.get('ca_results', {})
        stage = ''
        element_count = None
        sample_count = None
        if isinstance(feature_analysis, dict):
            stage = str(feature_analysis.get('stage') or '')
            element_count = feature_analysis.get('element_count')
            sample_count = feature_analysis.get('sample_count')
        selection = geo.get('target_element_selection', {})
        selected_elements = geo.get('target_related_elements') or selection.get('selected_elements') if isinstance(selection, dict) else geo.get('target_related_elements') or []
        if not isinstance(selected_elements, list):
            selected_elements = []
        sources = {}
        if isinstance(selection, dict):
            sources = selection.get('sources') or {}
        pred = all_results.get('prediction_model') if isinstance(all_results, dict) else None
        if not isinstance(pred, dict):
            pred = {}
        selected_features = pred.get('selected_features') or []
        if not isinstance(selected_features, list):
            selected_features = []
        model_results = pred.get('model_results') if isinstance(pred.get('model_results'), dict) else {}
        best_model_name = model_results.get('best_model_name') if isinstance(model_results, dict) else None
        best_model_score = model_results.get('best_model_score') if isinstance(model_results, dict) else None
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines: List[str] = []
        lines.append('# Feature Analysis And Feature Selection Notes')
        lines.append('')
        lines.append(f'- Generated at: {now}')
        lines.append(f'- Output directory: {os.path.abspath(self.output_dir)}')
        lines.append('')
        lines.append('## 1. Feature Analysis (Geology Expert)')
        lines.append('')
        if stage:
            lines.append(f'- Analysis stage: {stage}')
        if element_count is not None:
            lines.append(f'- Number of analyzed elements: {element_count}')
        if sample_count is not None:
            lines.append(f'- Number of valid samples: {sample_count}')
        if isinstance(feature_analysis, dict) and feature_analysis.get('summary'):
            summary_text = str(feature_analysis.get('summary')).strip()
            if summary_text:
                lines.append('')
                lines.append('**Summary**')
                lines.append('')
                lines.append('```text')
                lines.append(summary_text)
                lines.append('```')
        corr = feature_analysis.get('correlation_analysis', {}) if isinstance(feature_analysis, dict) else {}
        if isinstance(corr, dict):
            pearson = corr.get('pearson', {})
            heatmap_path = corr.get('heatmap_path')
            if isinstance(pearson, dict):
                high_corrs = pearson.get('high_correlations', [])
                if isinstance(high_corrs, list):
                    lines.append('')
                    lines.append('**Correlation Analysis**')
                    lines.append('')
                    lines.append(f'- Number of strong correlation pairs (>0.7, Pearson): {len(high_corrs)}')
            if heatmap_path:
                lines.append(f'- Correlation heatmap: {heatmap_path}')
        hc = feature_analysis.get('hierarchical_clustering', {}) if isinstance(feature_analysis, dict) else {}
        if isinstance(hc, dict) and hc:
            lines.append('')
            lines.append('**Hierarchical Clustering**')
            lines.append('')
            cc = hc.get('cluster_count')
            if cc is not None:
                try:
                    lines.append(f"- Number of clusters (maxclust): {int(cc)}")
                except Exception:
                    pass
            if hc.get('dendrogram_path'):
                lines.append(f"- Dendrogram: {hc.get('dendrogram_path')}")
            if hc.get('reordered_correlation_heatmap_path'):
                lines.append(f"- Reordered correlation heatmap: {hc.get('reordered_correlation_heatmap_path')}")
        factor = feature_analysis.get('factor_analysis', {}) if isinstance(feature_analysis, dict) else {}
        if isinstance(factor, dict) and factor:
            lines.append('')
            lines.append('**Factor Loadings**')
            lines.append('')
            if factor.get('factor_loading_plot_path'):
                lines.append(f"- Factor-loading heatmap: {factor.get('factor_loading_plot_path')}")
        if isinstance(ca_results, dict) and ca_results:
            lines.append('')
            lines.append('## 2. Key-Element Analysis (Geology Expert)')
            lines.append('')
            lines.append(f'- Number of key elements: {len(ca_results)}')
        lines.append('')
        lines.append('## 3. Target-Related Element Extraction (Geology Expert -> Data Expert)')
        lines.append('')
        if selected_elements:
            preview = ', '.join([str(x) for x in selected_elements[:50]])
            lines.append(f'- Target-related element list: {preview}')
            if len(selected_elements) > 50:
                lines.append(f'- Number of target-related elements: {len(selected_elements)} (truncated for display)')
        else:
            lines.append('- Target-related element list: none')
        if isinstance(sources, dict) and sources:
            lines.append('')
            lines.append('**Element Source Breakdown**')
            lines.append('')
            for key, vals in sources.items():
                if isinstance(vals, list) and vals:
                    vals_preview = ', '.join([str(v) for v in vals[:30]])
                    tail = f' (total {len(vals)})' if len(vals) > 30 else ''
                    lines.append(f'- {key}: {vals_preview}{tail}')
        lines.append('')
        lines.append('## 4. Feature Selection And Modeling (Data Expert)')
        lines.append('')
        if selected_features:
            lines.append(f'- Number of features actually used for training/prediction: {len(selected_features)}')
            lines.append('')
            lines.append('**Training Feature List**')
            lines.append('')
            lines.append('```text')
            lines.append(', '.join([str(x) for x in selected_features]))
            lines.append('```')
        else:
            lines.append('- Number of features actually used for training/prediction: not recorded')
        if best_model_name:
            lines.append('')
            lines.append('**Model-Selection Result**')
            lines.append('')
            best_model_display = self._display_model_name(best_model_name)
            best_low = str(best_model_name).strip().lower()
            model_type = 'SOM (QE scoring, with optional probability calibration)' if best_low in {"som", "self organizing map", "self-organizing map", "selforganizingmap", '\u81ea\u7ec4\u7ec7\u6620\u5c04', '\u81ea\u7ec4\u7ec7\u6620\u5c04\u795e\u7ecf\u7f51\u7edc'} else 'Supervised binary classification'
            lines.append(f'- Model type: {model_type}')
            if best_model_score is not None:
                lines.append(f'- Best model: {best_model_display} (score={best_model_score})')
            else:
                lines.append(f'- Best model: {best_model_display}')
        content = '\n'.join(lines) + '\n'
        _atomic_write_markdown(doc_path, content, encoding='utf-8')
        self.logger.info(f'Feature analysis and selection document saved to: {doc_path}')
        return doc_path
    def _output_target_element_selection_doc(self, all_results: Dict[str, Any]) -> str:
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        doc_path = os.path.join(reports_dir, 'target_element_selection.md')
        geo = all_results.get('geology_expert_results') if isinstance(all_results, dict) else None
        if not isinstance(geo, dict):
            geo = {}
        selection = geo.get('target_element_selection', {})
        if not isinstance(selection, dict):
            selection = {}
        selected_elements_obj = selection.get('selected_elements', [])
        selected_elements = [str(x) for x in selected_elements_obj] if isinstance(selected_elements_obj, list) else []
        sources_obj = selection.get('sources', {})
        sources = sources_obj if isinstance(sources_obj, dict) else {}
        source_name_map = {
            'target_key_elements': 'Key Elements for the Target Deposit Type',
            'correlation_related': 'Correlation Analysis',
            'factor_related': 'Factor Analysis',
            'cluster_related': 'Hierarchical Clustering',
            'anomaly_related': 'Key Element Analysis'
        }
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        counts_by_element: Dict[str, int] = {}
        source_hits: Dict[str, List[str]] = {}
        for key, vals in sources.items():
            if not isinstance(vals, list):
                continue
            for val in vals:
                elem = str(val).strip()
                if not elem:
                    continue
                counts_by_element[elem] = counts_by_element.get(elem, 0) + 1
                if elem not in source_hits:
                    source_hits[elem] = []
                src_label = source_name_map.get(str(key), str(key))
                if src_label not in source_hits[elem]:
                    source_hits[elem].append(src_label)
        lines: List[str] = []
        lines.append('# Target-Related Element Selection Notes')
        lines.append('')
        lines.append(f'- Generated at: {now}')
        lines.append(f'- Output directory: {os.path.abspath(self.output_dir)}')
        lines.append('')
        lines.append('## 1. Results From Three Feature-Analysis Sources')
        lines.append('')
        for source_key in ['correlation_related', 'factor_related', 'cluster_related']:
            vals_obj = sources.get(source_key, [])
            vals = [str(x) for x in vals_obj] if isinstance(vals_obj, list) else []
            lines.append(f"### {source_name_map.get(source_key, source_key)}")
            lines.append('')
            lines.append(f"- Number of elements: {len(vals)}")
            if vals:
                lines.append('')
                lines.append('```text')
                lines.append(', '.join(vals))
                lines.append('```')
            else:
                lines.append('- Element list: none')
            lines.append('')
        lines.append('## 2. Final Selected Elements')
        lines.append('')
        lines.append('- Current selection rule: an element must be supported by at least 2 of the 4 source groups: target-deposit key elements, correlation analysis, factor analysis, and hierarchical clustering.')
        lines.append(f'- Number of final selected elements: {len(selected_elements)}')
        if selected_elements:
            lines.append('')
            lines.append('```text')
            lines.append(', '.join(selected_elements))
            lines.append('```')
        else:
            lines.append('- Final element list: none')
        if selected_elements and source_hits:
            lines.append('')
            lines.append('## 3. Support Details For Final Selected Elements')
            lines.append('')
            lines.append('| Element | Number of supporting sources | Supporting sources |')
            lines.append('|---|---:|---|')
            for elem in selected_elements:
                hit_list = source_hits.get(elem, [])
                lines.append(f"| {elem} | {counts_by_element.get(elem, 0)} | {', '.join(hit_list)} |")
        content = '\n'.join(lines).rstrip() + '\n'
        _atomic_write_markdown(doc_path, content, encoding='utf-8')
        self.logger.info(f'Target-related element document saved to: {doc_path}')
        return doc_path
    def _resolve_report_basic_stats(self, all_results: Dict[str, Any]) -> Dict[str, int]:
        total_samples = 0
        total_columns = 0
        bundle_candidates = [
            all_results.get('data_preprocessing'),
            all_results.get('preprocessing_results'),
        ]
        for bundle in bundle_candidates:
            if not isinstance(bundle, dict):
                continue
            basic_stats = bundle.get('basic_stats')
            if not isinstance(basic_stats, dict):
                continue
            if total_samples <= 0:
                try:
                    total_samples = int(basic_stats.get('total_samples', 0) or 0)
                except Exception:
                    total_samples = 0
            if total_columns <= 0:
                columns = basic_stats.get('columns', [])
                if isinstance(columns, list):
                    total_columns = len(columns)
        top_basic_stats = all_results.get('basic_stats')
        if isinstance(top_basic_stats, dict):
            if total_samples <= 0:
                try:
                    total_samples = int(top_basic_stats.get('total_samples', 0) or 0)
                except Exception:
                    total_samples = 0
            if total_columns <= 0:
                columns = top_basic_stats.get('columns', [])
                if isinstance(columns, list):
                    total_columns = len(columns)
        prediction_model = all_results.get('prediction_model')
        if isinstance(prediction_model, dict):
            predictions = prediction_model.get('predictions')
            if isinstance(predictions, dict):
                if total_samples <= 0:
                    try:
                        total_samples = int(predictions.get('total_samples', 0) or 0)
                    except Exception:
                        total_samples = 0
                if total_samples <= 0:
                    probs = predictions.get('probabilities')
                    if isinstance(probs, list):
                        total_samples = len(probs)
                if total_samples <= 0:
                    pred_labels = predictions.get('predictions')
                    if isinstance(pred_labels, list):
                        total_samples = len(pred_labels)
            if total_columns <= 0:
                selected_features = prediction_model.get('selected_features')
                if isinstance(selected_features, list):
                    total_columns = len(selected_features)
        return {'total_samples': int(max(0, total_samples)), 'total_columns': int(max(0, total_columns))}
    def _generate_final_report_with_llm(self, all_results: Dict[str, Any]) -> str:
        key_findings = self._extract_key_findings(all_results)
        llm_interpretation = ''
        geo_interp = all_results.get('geological_interpretation')
        if geo_interp is None:
            geo_interp = all_results.get('geology_expert_results', {}).get('geological_interpretation', {})
        if isinstance(geo_interp, dict):
            llm_interpretation = geo_interp.get('llm_interpretation', '')
        stats = self._resolve_report_basic_stats(all_results)
        total_samples = stats.get('total_samples', 0)
        feature_count = stats.get('total_columns', 0)
        prediction_summary = ''
        best_model_name = ''
        best_model_score = None
        high_potential_count = 0
        high_potential_ratio = 0.0
        if 'prediction_model' in all_results:
            if isinstance(all_results['prediction_model'], dict):
                pred_obj = all_results['prediction_model']
                prediction_summary = pred_obj.get('summary', '')
                model_results = pred_obj.get('model_results', {})
                if isinstance(model_results, dict):
                    best_model_name = str(model_results.get('best_model_name', '') or '')
                    best_model_score = model_results.get('best_model_score')
                predictions = pred_obj.get('predictions', {})
                if isinstance(predictions, dict):
                    high_potential_count = int(predictions.get('high_potential_count', 0) or 0)
        if total_samples > 0:
            high_potential_ratio = high_potential_count / total_samples * 100
        quantitative_summary_lines: List[str] = [
            f'- Number of valid samples: {total_samples}',
            f'- Total number of variables: {feature_count}',
            f'- Number of high-potential samples: {high_potential_count}',
            f'- High-potential proportion: {high_potential_ratio:.2f}%'
        ]
        if best_model_name:
            if best_model_score is None:
                quantitative_summary_lines.append(f'- Best model: {best_model_name}')
            else:
                quantitative_summary_lines.append(f'- Best model: {best_model_name} (score={best_model_score})')
        geology_enabled = bool((all_results.get('config') or {}).get('geology_expert_enabled', True))
        if geology_enabled:
            prompt = 'You are an academic writer in geochemistry and economic geology. Based on the input results, generate a rigorous, auditable, academically styled comprehensive report.\n\n### Project Context\n- Analysis time: {current_time}\n\n### Quantitative Result Summary\n{quantitative_summary}\n\n### Key Findings\n{key_findings}\n\n### In-Depth Geological Interpretation (provided by experts)\n{llm_interpretation}\n\n### Prediction-Model Results\n{prediction_summary}\n\n### Writing And Structure Requirements (must be followed strictly)\n1. Output Markdown body text only, with no extra explanation.\n2. Use this exact title: # Comprehensive Multi-Agent Assessment Report For Geochemical Mineralization Potential\n3. Use this exact section structure:\n   - ## Abstract\n   - ## 1. Research Context And Data Basis\n   - ## 2. Methodological Framework And Criteria\n   - ## 3. Results\n   - ## 4. Discussion: Uncertainty And Limitations\n   - ## 5. Conclusions\n4. The Results section must include these three third-level headings:\n   - ### 3.1 Geochemical Anomaly Assemblages And Geological Significance\n   - ### 3.2 Spatial Pattern Of Mineralization Potential\n   - ### 3.3 Model Performance And Credibility\n5. Use an academic tone throughout: avoid slogans, exaggerated wording, and subjective assertions.\n6. Every number must come from the input information; if evidence is insufficient, explicitly state that the current results are insufficient to support the conclusion.\n7. Do not include recommendation sections such as exploration recommendations, prospecting recommendations, or next-step recommendations.'
        else:
            prompt = 'You are an academic writer specializing in geochemical data analysis. These results come from the ablation variant with the geology expert disabled. Generate a rigorous, verifiable report using only the all-element SOM, QE anomaly scores, and quantitative evaluation results.\n\n### Project Background\n- Analysis time: {current_time}\n\n### Quantitative Summary\n{quantitative_summary}\n\n### Prediction Model Results\n{prediction_summary}\n\n### Writing and Structure Requirements (mandatory)\n1. Return only the Markdown body, without additional commentary.\n2. Use the exact title: # GeochemMAF w/o geological expert: Ablation Experiment Report\n3. Use these sections: ## Abstract; ## 1. Data and All-Element Input; ## 2. SOM and QE Methods; ## 3. Quantitative Results; ## 4. Uncertainty and Limitations; ## 5. Conclusions.\n4. Do not provide geological interpretations, mineralization mechanisms, geological significance of elements, or exploration recommendations.\n5. All numerical values must come from the inputs. When information is insufficient, state: "The current results do not support this conclusion."'
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        key_findings_str = '\n'.join(key_findings)
        quantitative_summary = '\n'.join(quantitative_summary_lines)
        prompt = prompt.format(current_time=current_time, quantitative_summary=quantitative_summary, key_findings=key_findings_str, llm_interpretation=llm_interpretation, prediction_summary=prediction_summary)
        try:
            return self.decide(prompt, config=all_results.get('config') if isinstance(all_results, dict) else None)
        except Exception as e:
            self.logger.error(f'LLM report generation failed: {e}')
            return self.generate_report(all_results)
    def _output_predictions_data(self, data: Any, all_results: Dict[str, Any]) -> str:
        if 'prediction_model' not in all_results or all_results['prediction_model'] is None:
            return ''
        prediction_results = all_results['prediction_model']
        predictions = prediction_results.get('predictions', {})
        if isinstance(data, pd.DataFrame) and (not data.empty):
            result_df = data.copy()
            if 'probabilities' in predictions and predictions['probabilities'] is not None:
                probs = list(predictions['probabilities'])
                result_df['prediction_probability'] = (probs + [None] * len(result_df))[:len(result_df)]
            if 'predictions' in predictions and predictions['predictions'] is not None:
                pred_classes = list(predictions['predictions'])
                result_df['prediction_class'] = (pred_classes + [None] * len(result_df))[:len(result_df)]
            if 'confidence' in predictions and predictions['confidence'] is not None:
                conf = list(predictions['confidence'])
                result_df['confidence'] = (conf + [None] * len(result_df))[:len(result_df)]
            high_potential_indices = predictions.get('high_potential_indices', [])
            result_df['is_high_potential'] = 0
            valid_indices = [idx for idx in high_potential_indices if idx < len(result_df)]
            result_df.loc[valid_indices, 'is_high_potential'] = 1
            data_dir = os.path.join(self.output_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            predictions_path = os.path.join(data_dir, 'prediction_results.csv')
            _atomic_write_csv(result_df, predictions_path, index=False, encoding='utf-8-sig')
            high_potential_df = result_df[result_df['is_high_potential'] == 1]
            high_potential_path = os.path.join(data_dir, 'high_potential_areas.csv')
            _atomic_write_csv(high_potential_df, high_potential_path, index=False, encoding='utf-8-sig')
            self.logger.info(f'Prediction-result data saved to: {predictions_path}')
            self.logger.info(f'High-potential-area data saved to: {high_potential_path}')
            return predictions_path
        elif predictions:
            simple_data = {}
            if 'probabilities' in predictions:
                simple_data['prediction_probability'] = predictions['probabilities']
            if 'predictions' in predictions:
                simple_data['prediction_class'] = predictions['predictions']
            if 'confidence' in predictions:
                simple_data['confidence'] = predictions['confidence']
            if simple_data:
                min_length = min((len(v) for v in simple_data.values()))
                for key in simple_data:
                    simple_data[key] = simple_data[key][:min_length]
                simple_df = pd.DataFrame(simple_data)
                if 'high_potential_indices' in predictions:
                    simple_df['is_high_potential'] = 0
                    valid_indices = [idx for idx in predictions['high_potential_indices'] if idx < len(simple_df)]
                    simple_df.loc[valid_indices, 'is_high_potential'] = 1
                simple_path = os.path.join(self.output_dir, 'simple_predictions.csv')
                _atomic_write_csv(simple_df, simple_path, index=False, encoding='utf-8-sig')
                self.logger.info(f'Simplified prediction results saved to: {simple_path}')
                return simple_path
        return ''
    def _output_anomaly_analysis(self, all_results: Dict[str, Any]) -> str:
        element_anomalies = {}
        geo_results = all_results.get('geology_expert_results')
        if isinstance(geo_results, dict):
            key_element_analysis = geo_results.get('key_element_analysis', {})
            if isinstance(key_element_analysis, dict):
                element_anomalies = key_element_analysis.get('element_statistics', {}) or {}
            if (not element_anomalies) and isinstance(geo_results.get('ca_results'), dict):
                element_anomalies = geo_results.get('ca_results', {})
        if not element_anomalies:
            primary = all_results.get('geology_expert_primary', {})
            anomaly_analysis = primary.get('anomaly_analysis', {}) if isinstance(primary, dict) else {}
            if isinstance(anomaly_analysis, dict):
                element_anomalies = anomaly_analysis.get('element_anomalies', {}) or {}
            if isinstance(primary, dict) and (not element_anomalies) and isinstance(primary.get('element_anomalies'), dict):
                element_anomalies = primary.get('element_anomalies', {})
        if not isinstance(element_anomalies, dict) or not element_anomalies:
            return ''
        anomalies_data = []
        try:
            lang = _resolve_output_language()
            for element, stats in element_anomalies.items():
                if stats is not None:
                    anomalies_data.append(
                        {
                            _localize_text('Element', lang=lang): element,
                            _localize_text('Anomaly Threshold', lang=lang): stats.get('threshold', 0),
                            _localize_text('Anomaly Sample Count', lang=lang): stats.get('high_value_count', stats.get('anomaly_count', 0)),
                            _localize_text('Anomaly Percentage (%)', lang=lang): stats.get('high_value_percentage', stats.get('anomaly_percentage', 0)),
                        }
                    )
        except TypeError:
            return ''
        if anomalies_data:
            anomalies_df = pd.DataFrame(anomalies_data)
            data_dir = os.path.join(self.output_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            anomalies_path = os.path.join(data_dir, 'key_element_analysis.csv')
            _atomic_write_csv(anomalies_df, anomalies_path, index=False, encoding='utf-8-sig')
            self.logger.info(f'Key-element analysis results saved to: {anomalies_path}')
            return anomalies_path
        return ''
    def _output_feature_importance(self, all_results: Dict[str, Any]) -> str:
        if 'prediction_model' not in all_results or all_results['prediction_model'] is None:
            return ''
        if 'predictions' not in all_results['prediction_model'] or all_results['prediction_model']['predictions'] is None:
            return ''
        feature_importance = all_results['prediction_model']['predictions'].get('feature_importance', [])
        if feature_importance is None:
            return ''
        try:
            if feature_importance:
                importance_df = pd.DataFrame(feature_importance)
                data_dir = os.path.join(self.output_dir, 'data')
                os.makedirs(data_dir, exist_ok=True)
                importance_path = os.path.join(data_dir, 'feature_importance.csv')
                _atomic_write_csv(importance_df, importance_path, index=False, encoding='utf-8-sig')
                self.logger.info(f'Feature-importance results saved to: {importance_path}')
                return importance_path
        except Exception as e:
            self.logger.warning(f'Failed to save feature-importance results: {e}')
        return ''
    def _output_json_results(self, all_results: Dict[str, Any]) -> str:
        if all_results is None:
            return ''
        try:
            serializable_results = self._make_serializable(all_results)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            data_dir = os.path.join(self.output_dir, 'data')
            os.makedirs(data_dir, exist_ok=True)
            json_path = os.path.join(data_dir, 'complete_results.json')
            _atomic_write_json(serializable_results, json_path, encoding='utf-8')
            self.logger.info(f'Complete results saved as JSON to: {json_path}')
            return json_path
        except Exception as e:
            self.logger.error(f'Error while saving JSON results: {str(e)}')
            return ''
    def _make_serializable(self, data: Any) -> Any:
        if data is None:
            return None
        elif isinstance(data, dict):
            try:
                return {key: self._make_serializable(value) for key, value in data.items()}
            except TypeError:
                return {}
        elif isinstance(data, list):
            try:
                return [self._make_serializable(item) for item in data]
            except TypeError:
                return []
        elif isinstance(data, (np.ndarray, pd.Series)):
            try:
                return data.tolist()
            except Exception:
                return []
        elif isinstance(data, pd.DataFrame):
            try:
                return data.to_dict(orient='records')
            except Exception:
                return []
        elif isinstance(data, (np.int64, np.int32, np.float64, np.float32)):
            return float(data) if isinstance(data, (np.float64, np.float32)) else int(data)
        elif hasattr(data, '__dict__'):
            return f'<Object: {type(data).__name__}>'
        else:
            return data
    def _extract_key_findings(self, all_results: Dict[str, Any]) -> List[str]:
        findings = []
        geo_results = all_results.get('geology_expert_results')
        if isinstance(geo_results, dict):
            feature_summary = geo_results.get('feature_analysis', {}).get('summary')
            if feature_summary:
                findings.append(str(feature_summary))
            geo_interp_summary = geo_results.get('geological_interpretation', {}).get('summary')
            if geo_interp_summary:
                findings.append(str(geo_interp_summary))
        primary_results = all_results.get('geology_expert_primary')
        if isinstance(primary_results, dict):
            summary = primary_results.get('summary', '')
            if summary:
                findings.append(summary)
        if 'prediction_model' in all_results and all_results['prediction_model'] is not None:
            prediction_results = all_results['prediction_model']
            if prediction_results.get('predictions') is not None:
                high_potential_count = prediction_results['predictions'].get('high_potential_count', 0)
                findings.append(f'Identified {high_potential_count} high-potential mineralization areas')
        geo_interp = all_results.get('geological_interpretation')
        if geo_interp is None:
            geo_interp = all_results.get('geology_expert_results', {}).get('geological_interpretation')
        if isinstance(geo_interp, dict):
            mineralization_types = geo_interp.get('mineralization_types', {})
            if isinstance(mineralization_types, dict):
                primary_type = mineralization_types.get('primary_type', '')
                if primary_type and primary_type != 'Unidentified':
                    findings.append(f'Primary mineralization type: {primary_type}')
        if not findings and 'prediction_model' in all_results and (all_results['prediction_model'] is not None):
            findings.append('Completed predictive analysis of the geochemical data')
        return findings[:5]
    def _extract_main_recommendations(self, all_results: Dict[str, Any]) -> List[str]:
        recommendations = []
        geo_interp = all_results.get('geological_interpretation')
        if geo_interp is None:
            geo_interp = all_results.get('geology_expert_results', {}).get('geological_interpretation', {})
        if isinstance(geo_interp, dict):
            expl_recommendations = geo_interp.get('exploration_recommendations', [])
            if isinstance(expl_recommendations, list):
                for rec in expl_recommendations[:3]:
                    if rec is not None and isinstance(rec, dict) and ('description' in rec):
                        recommendations.append(rec['description'])
            plain_recommendations = geo_interp.get('recommendations', [])
            if isinstance(plain_recommendations, list) and len(recommendations) < 3:
                for rec in plain_recommendations[:3]:
                    if rec and isinstance(rec, str):
                        recommendations.append(rec)
        if len(recommendations) < 3:
            default_recommendations = ['Conduct detailed field geological surveys to validate the predictions', 'Perform geophysical measurements in high-potential areas', 'Establish a long-term monitoring program to track mineralization indicators']
            for rec in default_recommendations:
                if rec not in recommendations:
                    recommendations.append(rec)
                if len(recommendations) >= 3:
                    break
        return recommendations[:5]
    def _generate_prediction_map(
        self,
        raw_data: pd.DataFrame,
        all_results: Dict[str, Any],
        processed_data: pd.DataFrame=None,
        output_filename: str = "mineralization_potential_idw_map.png",
        title: str = 'Mineralization-Potential Prediction Map',
    ) -> str:
        return generate_prediction_map_idw(
            raw_data=raw_data,
            all_results=all_results,
            output_dir=self.output_dir,
            logger=self.logger,
            processed_data=processed_data,
            output_filename=output_filename,
            title=title,
        )
    def generate_report(self, all_results: Dict[str, Any]) -> str:
        report = '# Comprehensive Multi-Agent Report For Geochemical Mineralization-Potential Prediction\n\n'
        report += '## 1. Project Overview\n\n'
        stats = self._resolve_report_basic_stats(all_results)
        total_samples = stats.get('total_samples', 0)
        total_columns = stats.get('total_columns', 0)
        report += f'- **Data scale**: {total_samples} sampling points, '
        report += f'{total_columns} variables\n'
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report += f'- **Analysis date**: {current_time}\n'
        report += '- **Analysis platform**: LangGraph geochemical multi-agent software\n\n'
        has_target_analysis = False
        target_results = None
        geology_results = all_results.get('geology_expert_final', {})
        if not geology_results:
            geology_results = all_results.get('geology_expert_intermediate', {})
        if not geology_results:
            geology_results = all_results.get('geology_expert_primary', {})
        if not geology_results:
            geology_results = all_results.get('geology_expert_results', {})
        if isinstance(geology_results, dict):
            final_interpretation = geology_results.get('final_interpretation', {})
            if not final_interpretation and isinstance(geology_results.get('geological_interpretation'), dict):
                final_interpretation = geology_results.get('geological_interpretation', {})
            if isinstance(final_interpretation, dict) and 'target_metallogeny_analysis' in final_interpretation:
                has_target_analysis = True
                target_results = final_interpretation
        elif isinstance(geology_results, list):
            for result in geology_results:
                if isinstance(result, dict) and 'target_metallogeny_analysis' in result.get('final_interpretation', {}):
                    has_target_analysis = True
                    target_results = result.get('final_interpretation', {})
                    break
        if has_target_analysis:
            report += '## 2. Target-Deposit-Specific Analysis\n\n'
            mineralization_types = target_results.get('mineralization_types', [])
            if mineralization_types:
                report += '### 2.1 Identified Mineralization Types\n\n'
                for m_type in mineralization_types:
                    report += f'- **{m_type}**\n'
                report += '\n'
            confidence = target_results.get('confidence_level', 'Unknown')
            report += '### 2.2 Analysis Confidence\n\n'
            report += f'- **Confidence level**: {confidence}\n\n'
            if 'target_metallogeny_analysis' in target_results:
                target_analysis = target_results['target_metallogeny_analysis']
                has_high_potential = target_analysis.get('has_high_potential_areas', False)
                report += '### 2.3 High-Potential Area Analysis\n\n'
                if has_high_potential:
                    report += '- **High-potential mineralization areas are present** and should be prioritized for validation\n'
                else:
                    report += '- No significant high-potential mineralization areas have been identified\n'
                report += '\n'
        report += '## 3. Key-Element Analysis\n\n'
        anomaly_analysis = None
        primary_results = all_results.get('geology_expert_primary', {})
        if primary_results:
            if 'element_anomalies' in primary_results:
                anomaly_analysis = {'element_anomalies': primary_results['element_anomalies']}
            elif 'anomaly_analysis' in primary_results:
                anomaly_analysis = primary_results['anomaly_analysis']
        if anomaly_analysis is None:
            geo_bundle = all_results.get('geology_expert_results', {})
            if isinstance(geo_bundle, dict) and isinstance(geo_bundle.get('ca_results'), dict) and geo_bundle.get('ca_results'):
                anomaly_analysis = {'element_anomalies': geo_bundle.get('ca_results', {})}
        if anomaly_analysis:
            if 'element_anomalies' in anomaly_analysis:
                anomalies_list = []
                for elem, stats in anomaly_analysis['element_anomalies'].items():
                    anomalies_list.append((elem, stats))
                anomalies_list.sort(key=lambda x: x[1]['anomaly_count'], reverse=True)
                top_anomalies = anomalies_list[:10]
                report += '### 3.1 Statistics Of Major Key Elements\n\n'
                report += '| Element | Number of high-value samples | High-value ratio | High-value threshold |\n'
                report += '|------|------------|----------|----------|\n'
                for elem, stats in top_anomalies:
                    if isinstance(stats, dict):
                        anomaly_count = stats.get('high_value_count', stats.get('anomaly_count', 0))
                        anomaly_percentage = stats.get('high_value_percentage', stats.get('anomaly_percentage', 0))
                        threshold = stats.get('threshold', 0)
                        report += f'| {elem} | {anomaly_count} | {anomaly_percentage:.2f}% | {threshold:.4f} |\n'
                report += '\n### 3.2 Relationship Between Key Elements And The Target Deposit\n\n'
                for elem, stats in top_anomalies[:8]:
                    if not isinstance(stats, dict):
                        continue
                    relation_text = str(stats.get('relation_text') or '').strip()
                    overlap_count = stats.get('known_deposit_overlap_count', 0)
                    overlap_pct = stats.get('known_deposit_support_pct', 0)
                    if relation_text:
                        report += f'- **{elem}**: {relation_text}. Overlap with known deposits: {overlap_count}; deposit-support proportion: approximately {float(overlap_pct):.2f}%.\n'
            elif 'anomalies' in anomaly_analysis:
                anomalies_list = []
                for elem, stats in anomaly_analysis['anomalies'].items():
                    anomalies_list.append((elem, stats))
                anomalies_list.sort(key=lambda x: x[1]['anomaly_count'], reverse=True)
                top_anomalies = anomalies_list[:10]
                report += '### 3.1 Statistics Of Major Key Elements\n\n'
                report += '| Element | Number of high-value samples | High-value ratio | High-value threshold |\n'
                report += '|------|------------|----------|----------|\n'
                for elem, stats in top_anomalies:
                    if isinstance(stats, dict):
                        anomaly_count = stats.get('high_value_count', stats.get('anomaly_count', 0))
                        anomaly_percentage = stats.get('high_value_percentage', stats.get('anomaly_percentage', 0))
                        threshold = stats.get('threshold', 0)
                        report += f'| {elem} | {anomaly_count} | {anomaly_percentage:.2f}% | {threshold:.4f} |\n'
        report += '\n## 4. Analysis Of Element Associations\n\n'
        association_analysis = None
        if primary_results:
            if 'top_element_pairs' in primary_results or 'target_specific_analysis' in primary_results:
                association_analysis = {}
                if 'top_element_pairs' in primary_results:
                    association_analysis['top_element_pairs'] = primary_results['top_element_pairs']
                if 'target_specific_analysis' in primary_results:
                    association_analysis['target_specific_analysis'] = primary_results['target_specific_analysis']
            elif 'association_analysis' in primary_results:
                association_analysis = primary_results['association_analysis']
        if association_analysis is None:
            geo_bundle = all_results.get('geology_expert_results', {})
            if isinstance(geo_bundle, dict) and isinstance(geo_bundle.get('element_combos'), dict) and geo_bundle.get('element_combos'):
                association_analysis = geo_bundle.get('element_combos', {})
        if association_analysis and 'target_specific_analysis' in association_analysis:
            target_combos = association_analysis['target_specific_analysis'].get('identified_target_combos', [])
            if target_combos:
                report += '### 4.1 Characteristic Element Combinations For The Target Deposit\n\n'
                report += '| Element combination | Combination type | Mean correlation coefficient |\n'
                report += '|----------|----------|--------------|\n'
                for combo in target_combos:
                    report += f"| **{combo['combo']}** | {combo['description']} | {combo['avg_correlation']:.3f} |\n"
                report += '\n'
        if association_analysis:
            if 'top_element_pairs' in association_analysis:
                high_corr_pairs = []
                for pair_name, pair_data in association_analysis['top_element_pairs'].items():
                    if isinstance(pair_data, dict):
                        elements = pair_data.get('elements', [])
                        correlation = pair_data.get('correlation', 0)
                        if elements and len(elements) == 2:
                            high_corr_pairs.append({'elements': elements, 'correlation': correlation})
                if high_corr_pairs:
                    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    report += '### 4.2 Highly Correlated Element Pairs\n\n'
                    report += '| Element pair | Correlation coefficient |\n'
                    report += '|--------|----------|\n'
                    for pair in high_corr_pairs[:10]:
                        elem1, elem2 = pair['elements']
                        correlation = pair['correlation']
                        report += f'| {elem1}-{elem2} | {correlation:.3f} |\n'
            elif 'associations' in association_analysis:
                high_corr_pairs = association_analysis['associations'].get('high_correlation_pairs', [])
                if high_corr_pairs:
                    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    report += '### 4.2 Highly Correlated Element Pairs\n\n'
                    report += '| Element pair | Correlation coefficient |\n'
                    report += '|--------|----------|\n'
                    for pair in high_corr_pairs[:10]:
                        elem1, elem2 = pair['elements']
                        correlation = pair['correlation']
                        report += f'| {elem1}-{elem2} | {correlation:.3f} |\n'
        report += '\n## 5. Mineralization-Potential Prediction\n\n'
        prediction_results = all_results.get('prediction_model', {})
        if prediction_results:
            high_potential_count = 0
            if prediction_results.get('predictions') is not None:
                high_potential_count = prediction_results['predictions'].get('high_potential_count', 0)
                if high_potential_count == 0 and 'high_potential_indices' in prediction_results['predictions']:
                    high_potential_count = len(prediction_results['predictions']['high_potential_indices'])
            high_potential_percentage = high_potential_count / total_samples * 100 if total_samples > 0 else 0
            report += '### 5.1 Statistics Of High-Potential Areas\n\n'
            report += f'- **Number of high-potential areas**: {high_potential_count}\n'
            report += f'- **Proportion of all samples**: {high_potential_percentage:.2f}%\n'
            important_features = prediction_results.get('important_features', [])
            if important_features:
                report += '\n### 5.2 Important Predictive Features\n\n'
                report += '| Feature name | Importance score |\n'
                report += '|----------|------------|\n'
                for feature in important_features[:10]:
                    if isinstance(feature, dict) and 'feature' in feature and ('importance' in feature):
                        feature_name = feature['feature']
                        importance = feature['importance']
                        report += f'| {feature_name} | {importance:.4f} |\n'
        visualizations = all_results.get("visualizations") if isinstance(all_results, dict) else None
        if isinstance(visualizations, dict) and (visualizations.get("prediction_map_path") or visualizations.get("prediction_map_analysis")):
            report += '\n### 5.3 LLM-Based Geological Interpretation From Key-Element Spatial Patterns And The Mineralization-Potential Prediction Map\n\n'
            map_path = visualizations.get("prediction_map_path")
            map_analysis = visualizations.get("prediction_map_analysis")
            if map_path:
                try:
                    rel = os.path.relpath(str(map_path), os.path.join(self.output_dir, "reports")).replace("\\", "/")
                except Exception:
                    rel = str(map_path)
                report += f'![Mineralization-Potential Prediction Map]({rel})\n\n'
            if map_analysis:
                report += str(map_analysis).strip() + '\n'

        som_cluster_analysis = prediction_results.get("som_cluster_analysis") if isinstance(prediction_results, dict) else None
        som_geology_interpretation = prediction_results.get("som_geology_interpretation") if isinstance(prediction_results, dict) else None
        if isinstance(som_cluster_analysis, dict) and (som_cluster_analysis.get("all_elements") or som_cluster_analysis.get("filtered_elements")):
            report += '\n### 5.4 SOM Clustering Results And Geological Interpretation\n\n'

            def _fmt_rel(p: Any) -> str:
                if not p:
                    return ""
                try:
                    return os.path.relpath(str(p), os.path.join(self.output_dir, "reports")).replace("\\", "/")
                except Exception:
                    return str(p)

            def _append_run(run_container: Dict[str, Any], interp_container: Dict[str, Any], tag: str, title: str, elements_key: str) -> None:
                nonlocal report
                run_obj = run_container.get(tag)
                if not isinstance(run_obj, dict):
                    return
                qe = run_obj.get("qe")
                te = run_obj.get("te")
                final_k = run_obj.get("final_k")
                elbow_k = (run_obj.get("k_suggestion") or {}).get("elbow_k") if isinstance(run_obj.get("k_suggestion"), dict) else None
                artifacts_dir = run_obj.get("artifacts_dir") or run_obj.get("output_dir")
                elements = run_container.get(elements_key, [])
                elem_count = len(elements) if isinstance(elements, list) else 0
                report += f'#### {title}\n\n'
                report += f'- **Number of elements**: {elem_count}\n'
                report += f'- **Number of clusters (final_k)**: {final_k}\n'
                report += f'- **Suggested number of clusters (elbow_k)**: {elbow_k}\n'
                report += f'- **Quantization error (QE)**: {qe}\n'
                report += f'- **Topographic error (TE)**: {te}\n'
                if artifacts_dir:
                    report += f'- **Result directory**: {_fmt_rel(artifacts_dir)}\n'
                interp_text = None
                interp_path = None
                if isinstance(interp_container, dict) and isinstance(interp_container.get(tag), dict):
                    interp_text = interp_container[tag].get("text")
                    interp_path = interp_container[tag].get("path")
                if interp_path:
                    report += f'- **Interpretation file**: {_fmt_rel(interp_path)}\n'
                report += '\n'
                if interp_text:
                    report += str(interp_text).strip() + '\n\n'

            dual_source_runs = som_cluster_analysis.get("dual_source_runs")
            has_dual_source = isinstance(dual_source_runs, dict) and len(dual_source_runs) > 0
            if has_dual_source:
                primary_source = str(som_cluster_analysis.get("som_data_source") or "").strip() or "primary"
                report += 'The primary result uses the default main branch, while the remaining data sources are included for comparison.\n\n'
                report += f'#### Primary result data source: {primary_source}\n\n'
                _append_run(som_cluster_analysis, som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {}, "all_elements", 'All-Element SOM Run', "elements_all")
                _append_run(
                    som_cluster_analysis,
                    som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {},
                    "filtered_elements",
                    'Filtered-Element SOM Run',
                    "elements_filtered",
                )
                for source_name, run_container_obj in dual_source_runs.items():
                    if not isinstance(run_container_obj, dict):
                        continue
                    report += f'#### Comparative data source: {str(source_name)}\n\n'
                    _append_run(run_container_obj, {}, "all_elements", 'All-Element SOM Run', "elements_all")
                    _append_run(run_container_obj, {}, "filtered_elements", 'Filtered-Element SOM Run', "elements_filtered")
            else:
                _append_run(
                    som_cluster_analysis,
                    som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {},
                    "all_elements",
                    'All-Element SOM Run',
                    "elements_all",
                )
                _append_run(
                    som_cluster_analysis,
                    som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {},
                    "filtered_elements",
                    'Filtered-Element SOM Run',
                    "elements_filtered",
                )
        report += '\n## 6. Conclusions And Outlook\n\n'
        if has_target_analysis:
            report += '### 6.1 Summary Of Mineralization Potential\n\n'
            report += f'The geochemical analysis indicates a **{confidence}** level of mineralization potential in the study area.\n'
            report += f"The identified primary mineralization types are: {', '.join(mineralization_types)}.\n\n"
        report += '### 6.2 Overall Conclusions\n\n'
        report += '- The multi-agent workflow provides an integrated assessment of geochemical characteristics and mineralization potential in the study area\n'
        report += '- The combination of statistical analysis, machine learning, and geological expertise improves prediction reliability\n'
        report += '- The report outputs can support subsequent integrated analysis using multi-source geoscientific information\n\n'
        current_time = datetime.datetime.now()
        report += '---\n\n**Report generated at**: ' + current_time.strftime('%Y-%m-%d %H:%M:%S')
        return report
