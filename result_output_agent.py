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
            logger.error('无法生成预测图：原始数据不是有效的DataFrame或为空')
        except Exception:
            pass
        return ''
    if 'prediction_model' not in all_results or all_results['prediction_model'] is None:
        try:
            logger.error('无法生成预测图：没有预测结果数据')
        except Exception:
            pass
        return ''
    prediction_results = all_results['prediction_model']
    predictions = prediction_results.get('predictions', {})
    possible_x_cols = ['经度', 'longitude', 'LONGITUDE', 'Lon', 'lon']
    possible_y_cols = ['纬度', 'latitude', 'LATITUDE', 'Lat', 'lat']
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
                logger.warning(f'未找到标准坐标列，使用数值列 {x_col} 和 {y_col} 作为坐标')
            except Exception:
                pass
        else:
            try:
                logger.error('无法生成预测图：找不到足够的坐标或数值列')
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
        result_df, coord_meta = _normalize_coordinates(result_df, x_col=str(x_col), y_col=str(y_col), lon_col="经度", lat_col="纬度")
        x_col = "经度"
        y_col = "纬度"
        try:
            if isinstance(coord_meta, dict) and coord_meta.get("is_projected_input"):
                logger.info(f"坐标已规范化为经纬度: {coord_meta}")
        except Exception:
            pass
    except Exception as e:
        try:
            logger.error(f"无法将坐标规范化为经纬度，无法生成预测图: {e}")
        except Exception:
            pass
        return ""

    valid_data = result_df.dropna(subset=['prediction_probability', x_col, y_col])
    if valid_data.empty:
        try:
            logger.warning('无法生成预测图：预测概率数据为空')
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
            logger.warning("无法生成预测图：经纬度列无有效数值")
        except Exception:
            pass
        return ""
    x_label = _localize_text("经度", lang=lang)
    y_label = _localize_text("纬度", lang=lang)
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
        logger.info(f'使用IDW插值处理{len(x_values)}个样本点')
        logger.info(f'插值网格大小: {grid_size}x{grid_size}')
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
            logger.error(f'IDW插值失败，无法生成预测图: {str(e)}')
        except Exception:
            pass
        plt.close()
        return ''
    ax = plt.gca()
    ax.grid(False)
    im = ax.imshow(zi, extent=[x_min, x_max, y_min, y_max], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', alpha=0.9, aspect='equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    possible_label_cols = ['label', 'target', 'deposit', '矿床', '标签', 'label_encoded', 'target_encoded', 'labeled', 'has_deposit', 'is_deposit', 'Ore']
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
                    label=_localize_text('已知矿床', lang=lang),
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
    cbar.set_label(_localize_text('成矿潜力概率', lang=lang), fontsize=12)
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
            logger.exception(f'保存成矿潜力预测图失败: {str(e)}')
        except Exception:
            pass
        return ''
    finally:
        plt.close()
    try:
        logger.info(f'成矿潜力预测图已保存至：{map_path}')
    except Exception:
        pass
    return map_path
class ResultOutputAgent(BaseAgent):
    CAPABILITIES = [
        "综合报告：汇总全链路结果并生成 Markdown 报告",
        "可视化：生成成矿潜力预测图（IDW 插值）",
        "结果导出：导出预测表（CSV，包含概率/标签/高潜力标记）",
        "全链路归档：导出可序列化的全链路结果 JSON",
        "交付打包：将文件与输出目录打包为 zip",
    ]
    def __init__(self, output_dir: str='./output', llm=None):
        role_description = '负责生成综合地质分析报告和可视化结果。你需要基于所有智能体的分析结果，生成专业、清晰、易于理解的报告，并提供有价值的地质解释和找矿建议。'
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
        if low in {"som", "self organizing map", "self-organizing map", "selforganizingmap", "自组织映射", "自组织映射神经网络"}:
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
        ctx = f"样本总数={total_samples}；高潜力样本数={high_potential_count}。"
        prompt_text = (
            "你是一位资深地球化学/找矿制图专家。请仅基于我提供的“成矿潜力预测图（概率热力图）”进行解读，并输出可直接写入报告的 Markdown。\n"
            f"背景信息：{ctx}\n\n"
            "写作要求：\n"
            "1) 输出必须包含以下小节标题（按顺序）：\n"
            "   - 图面总体特征\n"
            "   - 高值区（热点）与低值区分布\n"
            "   - 与已知矿床点/样点分布的关系（若图中有标注）\n"
            "   - 不确定性与可能误判来源\n"
            "   - 找矿优先区划与下一步验证建议\n"
            "2) 解释应尽量具体：描述热点数量、相对方位/趋势、是否呈带状/团块状/孤立点；避免空泛。\n"
            "3) 不要编造不存在的图例/坐标信息；若无法从图上确定，请明确说明“不确定”。\n"
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
            self.logger.warning(f"预测图解译失败: {e}")
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
                raise RuntimeError("地质专家智能体不可用")
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
            self.logger.warning(f"地质专家靶区解译失败: {e}")
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

        lines.append("### 结果地质解译")
        lines.append("")
        summary_text = ""
        confidence_text = ""
        mineralization_text = ""
        if isinstance(geo_interp, dict):
            summary_text = str(geo_interp.get("summary") or "").strip()
            confidence_text = str(geo_interp.get("confidence_level") or "").strip()
            m_types = geo_interp.get("mineralization_types")
            if isinstance(m_types, list) and m_types:
                mineralization_text = "、".join([str(x) for x in m_types if str(x).strip()][:6])
            elif isinstance(m_types, dict):
                primary_type = str(m_types.get("primary_type") or "").strip()
                if primary_type:
                    mineralization_text = primary_type
        if summary_text:
            lines.append(f"研究区综合地球化学响应显示：{summary_text}")
            lines.append("")
        if confidence_text:
            lines.append(f"当前地质解译置信度判定为“{confidence_text}”。")
            lines.append("")
        if mineralization_text:
            lines.append(f"矿化类型识别结果以{mineralization_text}为主，表明异常组合与潜在成矿过程存在一定耦合关系。")
            lines.append("")
        if map_analysis:
            lines.append("基于成矿潜力图的空间解译显示：")
            lines.append("")
            lines.append(str(map_analysis).strip())
            lines.append("")
        geo_results = all_results.get("geology_expert_results", {}) if isinstance(all_results, dict) else {}
        key_element_analysis = geo_results.get("key_element_analysis", {}) if isinstance(geo_results, dict) else {}
        if isinstance(key_element_analysis, dict):
            relation_lines = key_element_analysis.get("relationship_summary") or []
            if isinstance(relation_lines, list) and relation_lines:
                lines.append("关键元素与目标矿种关系要点如下：")
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
            lines.append("SOM聚类地质解译要点如下：")
            lines.append("")
            for txt in som_lines[:2]:
                lines.append(txt)
                lines.append("")

        delineations = self._collect_target_delineations(all_results)
        success_items = [x for x in delineations if x.get("enabled") and x.get("map_path")]
        if success_items:
            lines.append("#### 基于最大Youden阈值的靶区圈定")
            lines.append("")
            lines.append("采用全区原始有效标签与QE得分计算ROC，以TPR−FPR最大值对应的阈值圈定靶区；QE不低于阈值的样本归为靶区。")
            lines.append("")
            lines.append("该结果为全区标签参与阈值选择的回顾性圈定结果，不作为独立预留测试集性能。图面使用最近邻显示二分类结果。")
            lines.append("")
            for item in success_items:
                run_tag = str(item.get("run_tag") or "run")
                lines.append(f"- {run_tag}: QE阈值={float(item['threshold']):.6g}，Youden={float(item['youden_index']):.4f}，敏感度={float(item['sensitivity']):.4f}，特异度={float(item['specificity']):.4f}。")
                for field, label in [("map_path", "靶区圈定图"), ("source_csv", "逐点分类数据"), ("metrics_json", "阈值与评价指标")]:
                    path = item.get(field)
                    if path:
                        rel_path = os.path.relpath(str(path), reports_dir).replace("\\", "/")
                        lines.append(f"- {run_tag} {label}: [{label}](<{rel_path}>)")
                interpretation = self._interpret_youden_targets(item, all_results)
                if interpretation.get("status") == "success":
                    lines.extend(["", f"##### {run_tag} 靶区地质解译", "", interpretation["markdown"], ""])
                elif interpretation.get("status") == "failed":
                    lines.extend(["", f"{run_tag} 大模型解译未完成；上述靶区图与数值指标仍为有效输出。", ""])
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

        for hdr in [r"\n##\s*3\.\s*结果\b", r"\n##\s*3\.\s*结果与分析\b", r"\n##\s*5\.\s*成矿潜力预测\b", r"\n##\s*结果\b"]:
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
        self.logger.info('正在生成结果...')
        try:
            if 'processing_history' not in state:
                state['processing_history'] = []
            if 'errors' not in state:
                state['errors'] = []
            processed_data = state.get('processed_data', state.get('preprocessed_data'))
            if processed_data is None:
                raise ValueError('没有找到处理后的数据，请先运行数据科学专家智能体')
            raw_data = state.get('data')
            if raw_data is None:
                raise ValueError('没有找到原始数据')
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
            state['processing_history'].append(f'{self.agent_name}: 结果输出完成')
            state['next_agent'] = None
            self.logger.info('结果输出完成')
            self.logger.info('输出文件路径:')
            for key, path in output_paths.items():
                if path:
                    self.logger.info(f'  - {key}: {path}')
        except Exception as e:
            self.logger.error(f'结果输出失败: {str(e)}')
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
        self.logger.info('结果输出智能体开始工作...')
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
                self.logger.error(f'生成综合报告时出错：{str(e)}')
                output_paths['comprehensive_report'] = ''
            try:
                feature_doc_path = self._output_feature_analysis_selection_doc(all_results)
                output_paths['feature_analysis_selection_doc'] = feature_doc_path
            except Exception as e:
                self.logger.error(f'生成特征分析与选择说明文档时出错：{str(e)}')
                output_paths['feature_analysis_selection_doc'] = ''
            if bool((all_results.get('config') or {}).get('geology_expert_enabled', True)):
                try:
                    element_doc_path = self._output_target_element_selection_doc(all_results)
                    output_paths['target_element_selection_doc'] = element_doc_path
                except Exception as e:
                    self.logger.error(f'生成目标相关元素说明文档时出错：{str(e)}')
                    output_paths['target_element_selection_doc'] = ''
            else:
                output_paths['target_element_selection_doc'] = ''
            try:
                predictions_path = self._output_predictions_data(data, all_results)
                output_paths['predictions_data'] = predictions_path
            except Exception as e:
                self.logger.error(f'输出预测结果时出错：{str(e)}')
                output_paths['predictions_data'] = ''
            try:
                anomalies_path = self._output_anomaly_analysis(all_results)
                output_paths['anomaly_analysis'] = anomalies_path
            except Exception as e:
                self.logger.error(f'输出异常分析结果时出错：{str(e)}')
                output_paths['anomaly_analysis'] = ''
            try:
                features_path = self._output_feature_importance(all_results)
                output_paths['feature_importance'] = features_path
            except Exception as e:
                self.logger.error(f'输出特征重要性时出错：{str(e)}')
                output_paths['feature_importance'] = ''
            try:
                json_path = self._output_json_results(all_results)
                output_paths['json_results'] = json_path
            except Exception as e:
                self.logger.error(f'输出JSON结果时出错：{str(e)}')
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
                            self.logger.info(f'详细Token分析报告已生成: {report_path}')
                    else:
                        self.logger.warning('TokenReporter模块未导入，跳过生成详细Token分析报告')
                except Exception as e:
                    self.logger.warning(f'无法生成详细Token分析报告: {e}')
            except Exception as e:
                self.logger.error(f'输出Token报告时出错：{str(e)}')
                output_paths['token_usage'] = ''
            self.logger.info(f'所有结果已输出到目录：{self.output_dir}')
        except Exception as e:
            self.logger.error(f'输出结果时出错：{str(e)}')
            output_paths = {'comprehensive_report': '', 'predictions_data': '', 'anomaly_analysis': '', 'feature_importance': '', 'json_results': '', 'token_usage': '', 'target_element_selection_doc': ''}
        return output_paths
    def _output_comprehensive_report(self, all_results: Dict[str, Any]) -> str:
        try:
            if self.llm:
                self.logger.info('正在使用 LLM 生成最终综合报告...')
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
            if (map_path or map_analysis) and ("LLM模型地质解译" not in str(report_content or "")) and ("成矿潜力预测图解译" not in str(report_content or "")):
                block_lines: List[str] = []
                block_lines.append("## 基于关键元素空间分布与成矿潜力预测图的LLM模型地质解译")
                block_lines.append("")
                if map_path:
                    try:
                        rel = os.path.relpath(str(map_path), reports_dir).replace("\\", "/")
                    except Exception:
                        rel = str(map_path)
                    block_lines.append(f"![成矿潜力预测图]({rel})")
                    block_lines.append("")
                if map_analysis:
                    block_lines.append(str(map_analysis).strip())
                    block_lines.append("")
                import re

                section_md = "\n".join(block_lines).rstrip() + "\n"
                text = str(report_content or "").rstrip() + "\n"
                m = re.search(r"\n##\s+(结论与建议|结论与展望|结论)\b", text)
                if m:
                    insert_at = int(m.start())
                    report_content = text[:insert_at].rstrip() + "\n\n" + section_md + "\n" + text[insert_at:].lstrip()
                else:
                    m2 = re.search(r"\n##\s+(成矿预测分析|成矿潜力预测)\b", text)
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
                appendix_lines.append("## 附：关键元素空间分布—成矿潜力耦合解译（地质专家/LLM）")
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
                        title = f"{elem}空间分布图" if elem else "关键元素空间分布图"
                    except Exception:
                        title = "关键元素空间分布图"
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
            self.logger.info(f'综合报告已保存至：{report_path}')
            return report_path
        except Exception as e:
            self.logger.error(f'生成综合报告时出错：{str(e)}')
            error_report = '# 地化数据成矿潜力预测综合报告\n\n'
            error_report += '## 生成报告时出现错误\n'
            error_report += f'- 错误信息：{str(e)}\n'
            error_report += '- 请检查数据格式和处理流程\n\n'
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            error_report_path = os.path.join(reports_dir, 'comprehensive_report_error.md')
            _atomic_write_text(error_report_path, error_report, encoding='utf-8')
            self.logger.info(f'错误报告已保存至：{error_report_path}')
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
        lines.append('# 特征分析与特征选择说明')
        lines.append('')
        lines.append(f'- 生成时间: {now}')
        lines.append(f'- 输出目录: {os.path.abspath(self.output_dir)}')
        lines.append('')
        lines.append('## 1. 特征分析（地质专家）')
        lines.append('')
        if stage:
            lines.append(f'- 分析阶段: {stage}')
        if element_count is not None:
            lines.append(f'- 分析元素数量: {element_count}')
        if sample_count is not None:
            lines.append(f'- 有效样本数量: {sample_count}')
        if isinstance(feature_analysis, dict) and feature_analysis.get('summary'):
            summary_text = str(feature_analysis.get('summary')).strip()
            if summary_text:
                lines.append('')
                lines.append('**摘要**')
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
                    lines.append('**相关性分析**')
                    lines.append('')
                    lines.append(f'- 强相关性对数量(>0.7, Pearson): {len(high_corrs)}')
            if heatmap_path:
                lines.append(f'- 相关性热力图: {heatmap_path}')
        hc = feature_analysis.get('hierarchical_clustering', {}) if isinstance(feature_analysis, dict) else {}
        if isinstance(hc, dict) and hc:
            lines.append('')
            lines.append('**层次聚类**')
            lines.append('')
            cc = hc.get('cluster_count')
            if cc is not None:
                try:
                    lines.append(f"- 聚类簇数量（maxclust）: {int(cc)}")
                except Exception:
                    pass
            if hc.get('dendrogram_path'):
                lines.append(f"- 树状图: {hc.get('dendrogram_path')}")
            if hc.get('reordered_correlation_heatmap_path'):
                lines.append(f"- 聚类重排相关性热力图: {hc.get('reordered_correlation_heatmap_path')}")
        factor = feature_analysis.get('factor_analysis', {}) if isinstance(feature_analysis, dict) else {}
        if isinstance(factor, dict) and factor:
            lines.append('')
            lines.append('**因子载荷**')
            lines.append('')
            if factor.get('factor_loading_plot_path'):
                lines.append(f"- 因子载荷热力图: {factor.get('factor_loading_plot_path')}")
        if isinstance(ca_results, dict) and ca_results:
            lines.append('')
            lines.append('## 2. 关键元素分析（地质专家）')
            lines.append('')
            lines.append(f'- 关键元素数量: {len(ca_results)}')
        lines.append('')
        lines.append('## 3. 目标相关元素提取（地质专家 → 数据专家）')
        lines.append('')
        if selected_elements:
            preview = ', '.join([str(x) for x in selected_elements[:50]])
            lines.append(f'- 目标相关元素清单: {preview}')
            if len(selected_elements) > 50:
                lines.append(f'- 目标相关元素数量: {len(selected_elements)}（已截断展示）')
        else:
            lines.append('- 目标相关元素清单: 空')
        if isinstance(sources, dict) and sources:
            lines.append('')
            lines.append('**元素来源分解**')
            lines.append('')
            for key, vals in sources.items():
                if isinstance(vals, list) and vals:
                    vals_preview = ', '.join([str(v) for v in vals[:30]])
                    tail = f'（共{len(vals)}个）' if len(vals) > 30 else ''
                    lines.append(f'- {key}: {vals_preview}{tail}')
        lines.append('')
        lines.append('## 4. 特征筛选与建模（数据专家）')
        lines.append('')
        if selected_features:
            lines.append(f'- 实际用于训练/预测的特征数: {len(selected_features)}')
            lines.append('')
            lines.append('**训练特征列表**')
            lines.append('')
            lines.append('```text')
            lines.append(', '.join([str(x) for x in selected_features]))
            lines.append('```')
        else:
            lines.append('- 实际用于训练/预测的特征数: 未记录')
        if best_model_name:
            lines.append('')
            lines.append('**模型选择结果**')
            lines.append('')
            best_model_display = self._display_model_name(best_model_name)
            best_low = str(best_model_name).strip().lower()
            model_type = "SOM(QE)评分（可选概率校准）" if best_low in {"som", "self organizing map", "self-organizing map", "selforganizingmap", "自组织映射", "自组织映射神经网络"} else "有监督二分类"
            lines.append(f'- 模型类型: {model_type}')
            if best_model_score is not None:
                lines.append(f'- 最佳模型: {best_model_display}（score={best_model_score}）')
            else:
                lines.append(f'- 最佳模型: {best_model_display}')
        content = '\n'.join(lines) + '\n'
        _atomic_write_markdown(doc_path, content, encoding='utf-8')
        self.logger.info(f'特征分析与选择说明文档已保存至：{doc_path}')
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
            'target_key_elements': '目标矿种关键元素',
            'correlation_related': '相关性分析',
            'factor_related': '因子分析',
            'cluster_related': '层次聚类',
            'anomaly_related': '关键元素分析'
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
        lines.append('# 目标相关元素筛选说明')
        lines.append('')
        lines.append(f'- 生成时间: {now}')
        lines.append(f'- 输出目录: {os.path.abspath(self.output_dir)}')
        lines.append('')
        lines.append('## 1. 三类特征分析结果')
        lines.append('')
        for source_key in ['correlation_related', 'factor_related', 'cluster_related']:
            vals_obj = sources.get(source_key, [])
            vals = [str(x) for x in vals_obj] if isinstance(vals_obj, list) else []
            lines.append(f"### {source_name_map.get(source_key, source_key)}")
            lines.append('')
            lines.append(f"- 元素数量: {len(vals)}")
            if vals:
                lines.append('')
                lines.append('```text')
                lines.append(', '.join(vals))
                lines.append('```')
            else:
                lines.append('- 元素清单: 空')
            lines.append('')
        lines.append('## 2. 最终选定元素')
        lines.append('')
        lines.append('- 当前筛选规则: 在「目标矿种关键元素 / 相关性分析 / 因子分析 / 层次聚类」四类来源中，至少被 2 类来源共同支持。')
        lines.append(f'- 最终元素数量: {len(selected_elements)}')
        if selected_elements:
            lines.append('')
            lines.append('```text')
            lines.append(', '.join(selected_elements))
            lines.append('```')
        else:
            lines.append('- 最终元素清单: 空')
        if selected_elements and source_hits:
            lines.append('')
            lines.append('## 3. 最终元素支持来源明细')
            lines.append('')
            lines.append('| 元素 | 支持来源数 | 支持来源 |')
            lines.append('|---|---:|---|')
            for elem in selected_elements:
                hit_list = source_hits.get(elem, [])
                lines.append(f"| {elem} | {counts_by_element.get(elem, 0)} | {', '.join(hit_list)} |")
        content = '\n'.join(lines).rstrip() + '\n'
        _atomic_write_markdown(doc_path, content, encoding='utf-8')
        self.logger.info(f'目标相关元素说明文档已保存至：{doc_path}')
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
            f'- 有效样本数: {total_samples}',
            f'- 变量总数: {feature_count}',
            f'- 高潜力样本数: {high_potential_count}',
            f'- 高潜力占比: {high_potential_ratio:.2f}%'
        ]
        if best_model_name:
            if best_model_score is None:
                quantitative_summary_lines.append(f'- 最佳模型: {best_model_name}')
            else:
                quantitative_summary_lines.append(f'- 最佳模型: {best_model_name}（score={best_model_score}）')
        geology_enabled = bool((all_results.get('config') or {}).get('geology_expert_enabled', True))
        if geology_enabled:
            prompt = '你是一位地球化学与矿床学方向的学术写作者。请基于输入结果，生成严谨、可复核、学术表达风格的综合报告。\n\n### 项目背景\n- 分析时间: {current_time}\n\n### 定量结果摘要\n{quantitative_summary}\n\n### 关键发现\n{key_findings}\n\n### 深度地质解译（由专家提供）\n{llm_interpretation}\n\n### 预测模型结果\n{prediction_summary}\n\n### 写作与结构要求（必须严格遵守）\n1. 仅输出 Markdown 正文，不要输出额外说明。\n2. 标题固定为：# 地球化学多智能体成矿潜力综合评估报告\n3. 章节结构固定为：\n   - ## 摘要\n   - ## 1. 研究背景与数据基础\n   - ## 2. 方法框架与判据\n   - ## 3. 结果\n   - ## 4. 讨论：不确定性与局限\n   - ## 5. 结论\n4. 在“结果”章节中必须包含三个三级标题：\n   - ### 3.1 地球化学异常组合与地质意义\n   - ### 3.2 成矿潜力空间格局\n   - ### 3.3 模型表现与可信度\n5. 全文使用学术语体：避免口号化表达、避免夸张措辞、避免主观判断。\n6. 所有数值必须来自输入信息；若信息不足，明确写“当前结果不足以支持该结论”。\n7. 不得出现“勘探建议”“找矿建议”“下一步工作建议”等建议类章节或小节。'
        else:
            prompt = '你是一位地球化学数据分析方向的学术写作者。当前结果来自“关闭地质专家”的消融版本。请仅基于全元素 SOM、QE 异常得分和定量评价结果生成严谨、可复核的报告。\n\n### 项目背景\n- 分析时间: {current_time}\n\n### 定量结果摘要\n{quantitative_summary}\n\n### 预测模型结果\n{prediction_summary}\n\n### 写作与结构要求（必须严格遵守）\n1. 仅输出 Markdown 正文，不要输出额外说明。\n2. 标题固定为：# GeochemMAF w/o geological expert 消融实验报告\n3. 章节结构固定为：## 摘要；## 1. 数据与全元素输入；## 2. SOM与QE方法；## 3. 定量结果；## 4. 不确定性与局限；## 5. 结论。\n4. 不得生成地质解译、成矿机理解释、元素成矿意义或找矿建议。\n5. 所有数值必须来自输入信息；信息不足时明确写“当前结果不足以支持该结论”。'
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        key_findings_str = '\n'.join(key_findings)
        quantitative_summary = '\n'.join(quantitative_summary_lines)
        prompt = prompt.format(current_time=current_time, quantitative_summary=quantitative_summary, key_findings=key_findings_str, llm_interpretation=llm_interpretation, prediction_summary=prediction_summary)
        try:
            return self.decide(prompt, config=all_results.get('config') if isinstance(all_results, dict) else None)
        except Exception as e:
            self.logger.error(f'LLM 报告生成失败: {e}')
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
            self.logger.info(f'预测结果数据已保存至：{predictions_path}')
            self.logger.info(f'高潜力区域数据已保存至：{high_potential_path}')
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
                self.logger.info(f'简单预测结果已保存至：{simple_path}')
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
                            _localize_text('元素', lang=lang): element,
                            _localize_text('异常阈值', lang=lang): stats.get('threshold', 0),
                            _localize_text('异常样本数', lang=lang): stats.get('high_value_count', stats.get('anomaly_count', 0)),
                            _localize_text('异常百分比(%)', lang=lang): stats.get('high_value_percentage', stats.get('anomaly_percentage', 0)),
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
            self.logger.info(f'关键元素分析结果已保存至：{anomalies_path}')
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
                self.logger.info(f'特征重要性分析结果已保存至：{importance_path}')
                return importance_path
        except Exception as e:
            self.logger.warning(f'保存特征重要性分析结果失败: {e}')
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
            self.logger.info(f'完整结果已以JSON格式保存至：{json_path}')
            return json_path
        except Exception as e:
            self.logger.error(f'保存JSON结果时出错：{str(e)}')
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
                findings.append(f'预测出{high_potential_count}个高潜力成矿区域')
        geo_interp = all_results.get('geological_interpretation')
        if geo_interp is None:
            geo_interp = all_results.get('geology_expert_results', {}).get('geological_interpretation')
        if isinstance(geo_interp, dict):
            mineralization_types = geo_interp.get('mineralization_types', {})
            if isinstance(mineralization_types, dict):
                primary_type = mineralization_types.get('primary_type', '')
                if primary_type and primary_type != '未明确识别':
                    findings.append(f'主要矿化类型：{primary_type}')
        if not findings and 'prediction_model' in all_results and (all_results['prediction_model'] is not None):
            findings.append('已完成地球化学数据的预测分析')
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
            default_recommendations = ['建议进行详细的野外地质调查验证预测结果', '针对高潜力区域进行地球物理测量', '建立长期监测系统，跟踪矿化指标变化']
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
        title: str = "成矿潜力预测图",
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
        report = '# 地球化学多智能体成矿潜力预测综合报告\n\n'
        report += '## 1. 项目概况\n\n'
        stats = self._resolve_report_basic_stats(all_results)
        total_samples = stats.get('total_samples', 0)
        total_columns = stats.get('total_columns', 0)
        report += f'- **数据规模**: {total_samples} 个采样点，'
        report += f'{total_columns} 个变量\n'
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report += f'- **分析日期**: {current_time}\n'
        report += '- **分析平台**: LangGraph 地球化学多智能体软件\n\n'
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
            report += '## 2. 目标矿种专项分析\n\n'
            mineralization_types = target_results.get('mineralization_types', [])
            if mineralization_types:
                report += '### 2.1 识别的矿化类型\n\n'
                for m_type in mineralization_types:
                    report += f'- **{m_type}**\n'
                report += '\n'
            confidence = target_results.get('confidence_level', '未知')
            report += '### 2.2 分析置信度\n\n'
            report += f'- **置信度级别**: {confidence}\n\n'
            if 'target_metallogeny_analysis' in target_results:
                target_analysis = target_results['target_metallogeny_analysis']
                has_high_potential = target_analysis.get('has_high_potential_areas', False)
                report += '### 2.3 高潜力区域分析\n\n'
                if has_high_potential:
                    report += '- **存在高潜力矿化区域**，建议优先验证\n'
                else:
                    report += '- 尚未发现显著的高潜力矿化区域\n'
                report += '\n'
        report += '## 3. 关键元素分析\n\n'
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
                report += '### 3.1 主要关键元素统计\n\n'
                report += '| 元素 | 高值样本数 | 高值比例 | 高值阈值 |\n'
                report += '|------|------------|----------|----------|\n'
                for elem, stats in top_anomalies:
                    if isinstance(stats, dict):
                        anomaly_count = stats.get('high_value_count', stats.get('anomaly_count', 0))
                        anomaly_percentage = stats.get('high_value_percentage', stats.get('anomaly_percentage', 0))
                        threshold = stats.get('threshold', 0)
                        report += f'| {elem} | {anomaly_count} | {anomaly_percentage:.2f}% | {threshold:.4f} |\n'
                report += '\n### 3.2 关键元素与目标矿种关系\n\n'
                for elem, stats in top_anomalies[:8]:
                    if not isinstance(stats, dict):
                        continue
                    relation_text = str(stats.get('relation_text') or '').strip()
                    overlap_count = stats.get('known_deposit_overlap_count', 0)
                    overlap_pct = stats.get('known_deposit_support_pct', 0)
                    if relation_text:
                        report += f'- **{elem}**: {relation_text} 与已知矿点叠加数为 {overlap_count}，矿点支撑比例约 {float(overlap_pct):.2f}%。\n'
            elif 'anomalies' in anomaly_analysis:
                anomalies_list = []
                for elem, stats in anomaly_analysis['anomalies'].items():
                    anomalies_list.append((elem, stats))
                anomalies_list.sort(key=lambda x: x[1]['anomaly_count'], reverse=True)
                top_anomalies = anomalies_list[:10]
                report += '### 3.1 主要关键元素统计\n\n'
                report += '| 元素 | 高值样本数 | 高值比例 | 高值阈值 |\n'
                report += '|------|------------|----------|----------|\n'
                for elem, stats in top_anomalies:
                    if isinstance(stats, dict):
                        anomaly_count = stats.get('high_value_count', stats.get('anomaly_count', 0))
                        anomaly_percentage = stats.get('high_value_percentage', stats.get('anomaly_percentage', 0))
                        threshold = stats.get('threshold', 0)
                        report += f'| {elem} | {anomaly_count} | {anomaly_percentage:.2f}% | {threshold:.4f} |\n'
        report += '\n## 4. 元素组合关系分析\n\n'
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
                report += '### 4.1 目标矿种特征元素组合\n\n'
                report += '| 元素组合 | 组合类型 | 平均相关系数 |\n'
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
                    report += '### 4.2 高相关性元素对\n\n'
                    report += '| 元素对 | 相关系数 |\n'
                    report += '|--------|----------|\n'
                    for pair in high_corr_pairs[:10]:
                        elem1, elem2 = pair['elements']
                        correlation = pair['correlation']
                        report += f'| {elem1}-{elem2} | {correlation:.3f} |\n'
            elif 'associations' in association_analysis:
                high_corr_pairs = association_analysis['associations'].get('high_correlation_pairs', [])
                if high_corr_pairs:
                    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    report += '### 4.2 高相关性元素对\n\n'
                    report += '| 元素对 | 相关系数 |\n'
                    report += '|--------|----------|\n'
                    for pair in high_corr_pairs[:10]:
                        elem1, elem2 = pair['elements']
                        correlation = pair['correlation']
                        report += f'| {elem1}-{elem2} | {correlation:.3f} |\n'
        report += '\n## 5. 成矿潜力预测\n\n'
        prediction_results = all_results.get('prediction_model', {})
        if prediction_results:
            high_potential_count = 0
            if prediction_results.get('predictions') is not None:
                high_potential_count = prediction_results['predictions'].get('high_potential_count', 0)
                if high_potential_count == 0 and 'high_potential_indices' in prediction_results['predictions']:
                    high_potential_count = len(prediction_results['predictions']['high_potential_indices'])
            high_potential_percentage = high_potential_count / total_samples * 100 if total_samples > 0 else 0
            report += '### 5.1 高潜力区域统计\n\n'
            report += f'- **高潜力区域数量**: {high_potential_count}\n'
            report += f'- **占总样本比例**: {high_potential_percentage:.2f}%\n'
            important_features = prediction_results.get('important_features', [])
            if important_features:
                report += '\n### 5.2 重要预测特征\n\n'
                report += '| 特征名称 | 重要性得分 |\n'
                report += '|----------|------------|\n'
                for feature in important_features[:10]:
                    if isinstance(feature, dict) and 'feature' in feature and ('importance' in feature):
                        feature_name = feature['feature']
                        importance = feature['importance']
                        report += f'| {feature_name} | {importance:.4f} |\n'
        visualizations = all_results.get("visualizations") if isinstance(all_results, dict) else None
        if isinstance(visualizations, dict) and (visualizations.get("prediction_map_path") or visualizations.get("prediction_map_analysis")):
            report += '\n### 5.3 基于关键元素空间分布与成矿潜力预测图的LLM模型地质解译\n\n'
            map_path = visualizations.get("prediction_map_path")
            map_analysis = visualizations.get("prediction_map_analysis")
            if map_path:
                try:
                    rel = os.path.relpath(str(map_path), os.path.join(self.output_dir, "reports")).replace("\\", "/")
                except Exception:
                    rel = str(map_path)
                report += f'![成矿潜力预测图]({rel})\n\n'
            if map_analysis:
                report += str(map_analysis).strip() + '\n'

        som_cluster_analysis = prediction_results.get("som_cluster_analysis") if isinstance(prediction_results, dict) else None
        som_geology_interpretation = prediction_results.get("som_geology_interpretation") if isinstance(prediction_results, dict) else None
        if isinstance(som_cluster_analysis, dict) and (som_cluster_analysis.get("all_elements") or som_cluster_analysis.get("filtered_elements")):
            report += '\n### 5.4 SOM聚类结果与地质解译\n\n'

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
                report += f'- **元素数**: {elem_count}\n'
                report += f'- **聚类数(final_k)**: {final_k}\n'
                report += f'- **建议聚类数(elbow_k)**: {elbow_k}\n'
                report += f'- **量化误差(QE)**: {qe}\n'
                report += f'- **拓扑误差(TE)**: {te}\n'
                if artifacts_dir:
                    report += f'- **结果目录**: {_fmt_rel(artifacts_dir)}\n'
                interp_text = None
                interp_path = None
                if isinstance(interp_container, dict) and isinstance(interp_container.get(tag), dict):
                    interp_text = interp_container[tag].get("text")
                    interp_path = interp_container[tag].get("path")
                if interp_path:
                    report += f'- **解译文件**: {_fmt_rel(interp_path)}\n'
                report += '\n'
                if interp_text:
                    report += str(interp_text).strip() + '\n\n'

            dual_source_runs = som_cluster_analysis.get("dual_source_runs")
            has_dual_source = isinstance(dual_source_runs, dict) and len(dual_source_runs) > 0
            if has_dual_source:
                primary_source = str(som_cluster_analysis.get("som_data_source") or "").strip() or "primary"
                report += '主结果采用默认主分支，其余数据源结果用于对比分析。\n\n'
                report += f'#### 主结果数据源：{primary_source}\n\n'
                _append_run(som_cluster_analysis, som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {}, "all_elements", "全元素SOM运行", "elements_all")
                _append_run(
                    som_cluster_analysis,
                    som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {},
                    "filtered_elements",
                    "筛选元素SOM运行",
                    "elements_filtered",
                )
                for source_name, run_container_obj in dual_source_runs.items():
                    if not isinstance(run_container_obj, dict):
                        continue
                    report += f'#### 对比数据源：{str(source_name)}\n\n'
                    _append_run(run_container_obj, {}, "all_elements", "全元素SOM运行", "elements_all")
                    _append_run(run_container_obj, {}, "filtered_elements", "筛选元素SOM运行", "elements_filtered")
            else:
                _append_run(
                    som_cluster_analysis,
                    som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {},
                    "all_elements",
                    "全元素SOM运行",
                    "elements_all",
                )
                _append_run(
                    som_cluster_analysis,
                    som_geology_interpretation if isinstance(som_geology_interpretation, dict) else {},
                    "filtered_elements",
                    "筛选元素SOM运行",
                    "elements_filtered",
                )
        report += '\n## 6. 结论与展望\n\n'
        if has_target_analysis:
            report += '### 6.1 成矿潜力总结\n\n'
            report += f'基于地球化学数据分析，研究区显示出**{confidence}**程度的成矿潜力。\n'
            report += f"识别出的主要矿化类型为：{', '.join(mineralization_types)}。\n\n"
        report += '### 6.2 总体结论\n\n'
        report += '- 通过多智能体协同分析，系统全面地评估了研究区的地球化学特征和矿化潜力\n'
        report += '- 结合统计分析、机器学习和地质专家知识，提高了预测的可靠性\n'
        report += '- 报告结果可作为后续多源地学信息综合分析的依据\n\n'
        current_time = datetime.datetime.now()
        report += '---\n\n**报告生成时间**: ' + current_time.strftime('%Y-%m-%d %H:%M:%S')
        return report
