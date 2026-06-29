import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
import os
import logging
import base64
import json
import re
from typing import Dict, List, Any, Optional, Tuple, cast
from sklearn.cluster import KMeans
try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent
try:
    from .utils.data_utils import localize_text as _localize_text
    from .utils.data_utils import resolve_output_language as _resolve_output_language
    from .utils.data_utils import setup_matplotlib_output_style as _setup_matplotlib_output_style
except Exception:
    from utils.data_utils import localize_text as _localize_text
    from utils.data_utils import resolve_output_language as _resolve_output_language
    from utils.data_utils import setup_matplotlib_output_style as _setup_matplotlib_output_style
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
class GeologyExpertAgent(BaseAgent):
    CAPABILITIES = [
        "Coordinate normalization: automatically detect geographic/projected coordinates and generate standardized longitude-latitude columns",
        "Target-deposit knowledge retrieval: extract key ore-forming elements, element associations, typical alteration, and geological significance",
        "Integrated geological analysis: identify key elements, element associations, geological implications, and exploration suggestions",
        "Key-element analysis: support target-related element analysis and high-value zone recognition",
        "Spatial analysis: identify spatial distribution patterns of key elements for target delineation",
        "Metallogenic inference: integrate multiple evidence sources to evaluate mineralization potential",
        "Coupled interpretation: jointly interpret key-element anomaly maps and mineral-potential prediction maps",
    ]
    def __init__(self, output_dir: str='./output', llm=None):
        role_description = (
            "You are a senior geology expert specializing in geochemical data analysis, geological interpretation, "
            "feature analysis, and mineralization-potential assessment. Use your domain knowledge to identify anomalous "
            "patterns, explain their geological meaning, and provide target-oriented exploration insights."
        )
        super().__init__('GeologyExpertAgent', role_description, llm)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_deposit_type: Optional[str] = None
        self.key_mineralization_elements: List[str] = []
        self.target_related_combos: Dict[str, Any] = {}
        self._register_skills()

    def _register_skills(self) -> None:
        reg = getattr(self, "skills", None)
        if reg is None or SkillSpec is None:
            return
        reg.register(
            SkillSpec(
                id="geo.normalize_coordinates",
                name="Coordinate normalization",
                description="Detect coordinate columns and generate standardized longitude-latitude fields with projected-coordinate conversion when possible",
                inputs={"df": "input DataFrame"},
                outputs={"df": "normalized DataFrame", "meta": "normalization metadata"},
                tags=("geo", "spatial", "preprocess"),
            ),
            lambda *, ctx, df, x_col=None, y_col=None, lon_col="Longitude", lat_col="Latitude": self.normalize_coordinates(
                df=df, x_col=x_col, y_col=y_col, lon_col=str(lon_col), lat_col=str(lat_col)
            ),
        )

    def normalize_coordinates(
        self,
        *,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        lon_col: str = "Longitude",
        lat_col: str = "Latitude",
    ) -> Dict[str, Any]:
        fn = self._get_skill_tool_callable("geo.normalize_coordinates", "tool.py", "normalize_coordinates")
        if fn is not None:
            return fn(df, x_col=x_col, y_col=y_col, lon_col=str(lon_col), lat_col=str(lat_col))
        try:
            from .utils.data_utils import normalize_coordinates as _normalize_coordinates
        except Exception:
            from utils.data_utils import normalize_coordinates as _normalize_coordinates
        out_df, meta = _normalize_coordinates(df, x_col=x_col, y_col=y_col, lon_col=str(lon_col), lat_col=str(lat_col))
        return {"df": out_df, "meta": meta}

    def _knowledge_enabled(self, config: Optional[Dict[str, Any]] = None) -> bool:
        if isinstance(config, dict):
            flag = config.get("knowledge_enabled")
            if flag is not None:
                return bool(flag)
        return True

    def retrieve_mineralization_characteristics(
        self, target_deposit_type: str, config: Optional[Dict[str, Any]] = None, study_area_location: Optional[str] = None
    ):
        self.target_deposit_type = target_deposit_type
        if not self._knowledge_enabled(config):
            self.key_mineralization_elements = []
            self.target_related_combos = {}
            try:
                self.logger.info("Target-domain knowledge retrieval is disabled by configuration. Downstream element screening will rely on statistical evidence only.")
            except Exception:
                logger.info("Target-domain knowledge retrieval is disabled by configuration. Downstream element screening will rely on statistical evidence only.")
            return
        try:
            self.logger.info(f"Retrieving mineralization characteristics for {target_deposit_type}...")
        except Exception:
            logger.info(f"Retrieving mineralization characteristics for {target_deposit_type}...")
        area_text = f"Study area: {study_area_location}\n" if study_area_location else ""
        prompt = (
            f"Act as a senior economic geologist and analyze the geochemical anomaly characteristics of {target_deposit_type}.\n"
            f"{area_text}"
            "Return a JSON object with the following fields:\n"
            "1. key_elements: a list of the principal ore-forming and indicator elements for this deposit type (for example ['Cu', 'Au', 'Mo']).\n"
            "2. element_associations: a dictionary where the key is an element association (for example 'Cu-Au') and the value is its geological meaning.\n"
            "3. typical_alteration: a list of typical wall-rock alteration types.\n"
            "4. geological_significance: a short statement on the geological significance of this deposit type; if the regional metallogenic background cannot be inferred from the study area, explicitly write 'uncertain'.\n\n"
            "Return JSON only, without any additional explanation.\n"
        )
        default_payload = {'key_elements': [], 'element_associations': {}, 'typical_alteration': [], 'geological_significance': ''}
        try:
            data = self.decide_json(prompt, default_payload, config=config)
            if not isinstance(data, dict):
                data = default_payload
            self.key_mineralization_elements = data.get('key_elements', [])
            self.target_related_combos = data.get('element_associations', {})
            try:
                self.logger.info(f"Successfully retrieved characteristics for {target_deposit_type}")
                self.logger.info(f"Key elements: {self.key_mineralization_elements}")
                self.logger.info(f"Element associations: {list(self.target_related_combos.keys())}")
            except Exception:
                logger.info(f"Successfully retrieved characteristics for {target_deposit_type}")
        except Exception as e:
            try:
                self.logger.exception(f"Failed to retrieve mineralization characteristics: {str(e)}")
            except Exception:
                logger.exception(f"Failed to retrieve mineralization characteristics: {str(e)}")
            self.key_mineralization_elements = []
            self.target_related_combos = {}
    def run(self, state: dict) -> dict:
        self.logger.info('Starting execution...')
        try:
            if 'processing_history' not in state:
                state['processing_history'] = []
            if 'errors' not in state:
                state['errors'] = []
            if 'analysis_results' not in state:
                state['analysis_results'] = {}
            target_deposit = state.get('target_deposit_type')
            config = state.get('config') if isinstance(state.get('config'), dict) else {}
            study_area_location = state.get('study_area_location')
            if not study_area_location and isinstance(config, dict):
                study_area_location = config.get("study_area_location")
            if target_deposit and target_deposit != self.target_deposit_type:
                self.logger.info(f'Detected target deposit: {target_deposit}. Retrieving mineralization characteristics...')
                self.retrieve_mineralization_characteristics(target_deposit, config=config, study_area_location=study_area_location)
            data = state.get('processed_data')
            if data is None:
                data = state.get('data')
            if data is None:
                raise ValueError('Data not found. Please load data first.')
            element_cols_obj = state.get('element_cols', [])
            if isinstance(element_cols_obj, list):
                element_cols = [str(c) for c in element_cols_obj]
            else:
                element_cols = []
            if not element_cols:
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                exclude_cols = ['FID', 'Ore', 'Longitude', 'Latitude']
                element_cols = [col for col in numeric_cols if col not in exclude_cols]
            stage = state.get('current_phase', 'primary')
            if 'geology' in stage.lower():
                stage = 'primary'
            feature_results_obj = state.get('feature_analysis_results')
            feature_results: Optional[Dict[str, Any]] = feature_results_obj if isinstance(feature_results_obj, dict) else None
            if feature_results is None:
                self.logger.info('No feature-analysis result (feature_analysis_results) was detected. Handing off to the data-science expert first.')
                state['processing_history'].append(f'{self.agent_name}: waiting for the data-science expert to complete feature analysis')
                state['next_agent'] = 'data_science_expert'
                return state
            prediction_results_obj = state.get('prediction_results')
            prediction_results: Optional[Dict[str, Any]] = prediction_results_obj if isinstance(prediction_results_obj, dict) else None
            cached_results_obj = state.get('geology_expert_results')
            cached_results: Optional[Dict[str, Any]] = cached_results_obj if isinstance(cached_results_obj, dict) else None
            step_overrides_obj: Any = None
            try:
                container = state.get("hitl_step_overrides")
                if isinstance(container, dict):
                    step_overrides_obj = container.get("geology_analysis")
            except Exception:
                step_overrides_obj = None
            step_overrides = step_overrides_obj if isinstance(step_overrides_obj, dict) else {}
            geology_results = self.analyze(
                data=data,
                element_cols=element_cols,
                stage=stage,
                feature_results=feature_results,
                prediction_results=prediction_results,
                cached_results=cached_results,
                config=config,
                step_overrides=step_overrides,
            )
            state['geology_expert_results'] = geology_results
            if 'feature_analysis' in geology_results:
                state['feature_analysis_results'] = geology_results['feature_analysis']
            state['analysis_results']['geology'] = geology_results
            state['processing_history'].append(f'{self.agent_name}: geological analysis completed')
            if not prediction_results:
                self.logger.info('Geological analysis completed. Handing off to the data-science expert for prediction...')
                state['next_agent'] = 'data_science_expert'
            else:
                self.logger.info('Geological analysis completed. Prediction results already exist, so the decision center will choose the next step...')
                state['next_agent'] = 'agent_decision'
            self.logger.info('Execution completed')
        except Exception as e:
            self.logger.exception(f'Execution failed: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'{self.agent_name}: {str(e)}')
            state['processing_history'].append(f'{self.agent_name}: execution failed - {str(e)}')
            state['next_agent'] = 'agent_decision'
        return state
    def is_compositional_data(self, data: pd.DataFrame, element_cols: List[str]) -> Tuple[bool, float]:
        valid_cols = [col for col in element_cols if col in data.columns]
        if not valid_cols:
            return (False, 0.0)
        is_non_negative = (data[valid_cols] >= 0).all().all()
        if not is_non_negative:
            return (False, 0.0)
        row_sums = data[valid_cols].sum(axis=1)
        avg_sum = row_sums.mean()
        is_close_to_1 = abs(avg_sum - 1.0) < 0.05
        is_close_to_100 = abs(avg_sum - 100.0) < 5.0
        if row_sums.std() > 0.01 * avg_sum:
            return (False, avg_sum)
        return (is_close_to_1 or is_close_to_100, avg_sum)
    def analyze(self, data: pd.DataFrame, element_cols: List[str], stage: str='primary', feature_results: Optional[Dict[str, Any]]=None, prediction_results: Optional[Dict[str, Any]]=None, cached_results: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]]=None, step_overrides: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        self.logger.info(f'Geology expert agent started {self._get_stage_name(stage)} analysis...')
        task = f'Analyze the geological data and decide the best {self._get_stage_name(stage)} analysis strategy'
        context = {'data_shape': data.shape, 'element_cols': element_cols}
        analysis_strategy = self.decide(task, context, config=config)
        if '```python' in analysis_strategy:
            text_part = analysis_strategy.split('```python')[0].strip()
            self.logger.info(f'Analysis strategy decision: {text_part} [code saved]')
        else:
            self.logger.info(f'Analysis strategy decision: {analysis_strategy}')
        is_compositional, avg_sum = self.is_compositional_data(data, element_cols)
        results: Dict[str, Any] = {'stage': stage, 'total_samples': len(data), 'is_compositional_data': is_compositional, 'avg_row_sum': avg_sum}
        if feature_results is None:
            raise ValueError('feature_analysis_results is empty. Let the data-science expert complete feature analysis first.')
        self.logger.info('Reusing feature-analysis results provided by the data-science expert')
        results['feature_analysis'] = feature_results
        target_element_selection = self._select_target_related_elements(element_cols, feature_results, {}, config=config)
        results['target_element_selection'] = target_element_selection
        target_element_selection_csv = self._save_target_element_sources_csv(target_element_selection)
        if target_element_selection_csv:
            results['target_element_selection_csv'] = target_element_selection_csv
        selected_elements_obj = target_element_selection.get('selected_elements', [])
        selected_elements: List[str] = [str(x) for x in selected_elements_obj] if isinstance(selected_elements_obj, list) else []
        results['target_related_elements'] = selected_elements
        if selected_elements:
            self.logger.info(f"Selected target-related elements from feature analysis: {', '.join(selected_elements)}")
        else:
            self.logger.info('No target-related elements could be selected from feature analysis')
        if cached_results and stage != 'primary':
            self.logger.info('Reusing previous key-element and element-association analysis results')
            results['anomaly_analysis'] = cached_results.get('anomaly_analysis', {})
            results['association_analysis'] = cached_results.get('association_analysis', {})
            anomaly_analysis_obj = results.get('anomaly_analysis') if isinstance(results, dict) else None
            element_anomalies = anomaly_analysis_obj.get('element_anomalies', {}) if isinstance(anomaly_analysis_obj, dict) else {}
            cached_anomaly_elements = list(element_anomalies.keys()) if isinstance(element_anomalies, dict) else []
            if set(cached_anomaly_elements) != set(selected_elements):
                self.logger.info('Cached key-element set differs from the target-related elements. Reanalyzing with target-related elements.')
                anomaly_analysis = self._analyze_element_anomalies(data, element_cols, prefer_elements=selected_elements)
                results['anomaly_analysis'] = anomaly_analysis
                results['key_element_analysis'] = anomaly_analysis
                results['ca_results'] = anomaly_analysis.get('element_anomalies', {})
            association_analysis_obj = results.get('association_analysis') if isinstance(results, dict) else None
            assoc_key_elements_obj = association_analysis_obj.get('key_elements_analyzed', []) if isinstance(association_analysis_obj, dict) else []
            assoc_key_elements = [str(x) for x in assoc_key_elements_obj] if isinstance(assoc_key_elements_obj, list) else []
            if set(assoc_key_elements) != set(selected_elements):
                self.logger.info('Cached association-analysis element set differs from the target-related elements. Reanalyzing with target-related elements.')
                association_analysis = self._analyze_element_associations(data, prefer_elements=selected_elements)
                results['association_analysis'] = association_analysis
        else:
            anomaly_analysis = self._analyze_element_anomalies(data, element_cols, prefer_elements=selected_elements)
            results['anomaly_analysis'] = anomaly_analysis
            results['key_element_analysis'] = anomaly_analysis
            results['ca_results'] = anomaly_analysis.get('element_anomalies', {})
            skip_visualizations = False
            if isinstance(step_overrides, dict):
                skip_visualizations = bool(step_overrides.get("skip_visualizations", False))
            if skip_visualizations:
                self.logger.info("HITL: key-element visualizations were skipped according to user settings")
            else:
                key_element_cols_obj = anomaly_analysis.get('key_element_cols') if isinstance(anomaly_analysis, dict) else None
                key_element_cols = [str(x) for x in key_element_cols_obj] if isinstance(key_element_cols_obj, list) else []
                self.logger.info(f"Generating visualizations for {len(key_element_cols)} key mineralization elements: {', '.join(key_element_cols)}")
                viz = CAVisualization(output_dir=self.output_dir)
                ca_results_obj = anomaly_analysis.get('element_anomalies', {}) if isinstance(anomaly_analysis, dict) else {}
                if isinstance(ca_results_obj, dict) and ca_results_obj:
                    viz_outputs = viz.plot_ca_result_images(data, key_element_cols, ca_results_obj, coord_cols=['Longitude', 'Latitude'])
                    results['ca_visualizations'] = viz_outputs
                    results['key_element_visualizations'] = viz_outputs
                    ca_img_count = sum((len(v) for v in viz_outputs.values() if isinstance(v, list)))
                    self.logger.info(f'Generated {ca_img_count} key-element spatial-distribution images')
            association_analysis = self._analyze_element_associations(data, prefer_elements=selected_elements)
            results['association_analysis'] = association_analysis
        self.logger.info(f'Running {self._get_stage_name(stage)} geological interpretation...')
        anomaly_analysis_for_interp = cast(Dict[str, Any], results.get('anomaly_analysis', {}))
        ca_results = cast(Dict[str, Any], anomaly_analysis_for_interp.get('element_anomalies', {}))
        interpretation_results = self.interpret(data=data, element_cols=element_cols, ca_results=ca_results, anomaly_analysis=anomaly_analysis_for_interp, prediction_results=prediction_results, stage=stage)
        results['geological_interpretation'] = interpretation_results
        if stage == 'primary':
            potential_areas = self._identify_potential_mineralization_areas(data, anomaly_analysis_for_interp)
            results['potential_areas'] = potential_areas
            association_analysis_for_summary = cast(Dict[str, Any], results.get('association_analysis', {}))
            results['summary'] = self._generate_primary_summary(anomaly_analysis_for_interp, association_analysis_for_summary, potential_areas)
            self.logger.info(f'Primary-stage analysis completed. Identified {len(potential_areas)} potential mineralization areas')
        elif stage == 'intermediate':
            results['detailed_assessment'] = self._perform_detailed_assessment(data, anomaly_analysis_for_interp, feature_results)
            detailed_assessment = cast(Dict[str, Any], results['detailed_assessment'])
            results['summary'] = self._generate_intermediate_summary(detailed_assessment)
            feature_contribution_obj = detailed_assessment.get('feature_contribution', {})
            feature_contribution = feature_contribution_obj if isinstance(feature_contribution_obj, dict) else {}
            self.logger.info(f"Intermediate-stage analysis completed. Built contribution assessments for {len(feature_contribution)} elements")
        elif stage == 'final':
            results['final_interpretation'] = self._provide_final_interpretation(data, anomaly_analysis_for_interp, prediction_results)
            results['summary'] = self._generate_final_summary(cast(Dict[str, Any], results['final_interpretation']))
        return results
    def _calculate_c_a_threshold(self, values: pd.Series) -> float:
        valid_values = values.dropna()
        if valid_values.empty:
            return 0.0
        valid_values = valid_values[valid_values > 0]
        if valid_values.empty:
            return 0.0
        sorted_values = np.sort(valid_values)
        n = int(len(sorted_values))
        cumulative_area = np.arange(1, n + 1) / n
        y = np.log(sorted_values).astype(float)
        x = np.log(1 - cumulative_area + 1e-10).astype(float)

        px = np.cumsum(x)
        py = np.cumsum(y)
        pxx = np.cumsum(x * x)
        pxy = np.cumsum(x * y)
        pyy = np.cumsum(y * y)

        total_sx = float(px[-1])
        total_sy = float(py[-1])
        total_sxx = float(pxx[-1])
        total_sxy = float(pxy[-1])
        total_syy = float(pyy[-1])

        def _sse(seg_n: int, sx: float, sy: float, sxx: float, sxy: float, syy: float) -> float:
            n_f = float(seg_n)
            if seg_n <= 1:
                return float('inf')
            denom = n_f * sxx - sx * sx
            if abs(denom) <= 1e-20:
                return syy - (sy * sy) / n_f
            b = (n_f * sxy - sx * sy) / denom
            a = (sy - b * sx) / n_f
            return syy - 2 * a * sy - 2 * b * sxy + a * a * n_f + 2 * a * b * sx + b * b * sxx
        best_threshold_idx = 0
        min_residual = float('inf')
        start = int(n * 0.1)
        end = int(n * 0.9)
        for i in range(start, end):
            if i < 5 or (n - i) < 5:
                continue
            sx1 = float(px[i - 1])
            sy1 = float(py[i - 1])
            sxx1 = float(pxx[i - 1])
            sxy1 = float(pxy[i - 1])
            syy1 = float(pyy[i - 1])
            sx2 = total_sx - sx1
            sy2 = total_sy - sy1
            sxx2 = total_sxx - sxx1
            sxy2 = total_sxy - sxy1
            syy2 = total_syy - syy1
            total_residual = _sse(i, sx1, sy1, sxx1, sxy1, syy1) + _sse(n - i, sx2, sy2, sxx2, sxy2, syy2)
            if total_residual < min_residual:
                min_residual = total_residual
                best_threshold_idx = i
        if best_threshold_idx == 0:
            threshold = float(np.median(sorted_values) * 1.5)
        else:
            threshold = float(sorted_values[best_threshold_idx])
        threshold = max(threshold, float(sorted_values.min() + 1e-10))
        return threshold
    def _perform_ca_analysis(self, values: pd.Series) -> Dict[str, Any]:
        valid_values = values.dropna()
        valid_values = valid_values[valid_values > 0]
        if valid_values.empty:
            return {'threshold': 0.0, 'sorted_values': np.array([]), 'log_concentration': np.array([]), 'log_area': np.array([]), 'background_slope': None, 'anomaly_slope': None, 'threshold_index': 0}
        sorted_values = np.sort(valid_values)
        n = len(sorted_values)
        cumulative_area = np.arange(1, n + 1) / n
        log_concentration = np.log(sorted_values)
        log_area = np.log(1 - cumulative_area + 1e-10)
        threshold = self._calculate_c_a_threshold(values)
        threshold_idx = np.searchsorted(sorted_values, threshold)
        bg_coef = None
        anom_coef = None
        if 0 < threshold_idx < n:
            bg_log_area = log_area[:threshold_idx]
            bg_log_conc = log_concentration[:threshold_idx]
            if len(bg_log_area) >= 5:
                bg_coef = np.polyfit(bg_log_area, bg_log_conc, 1)
        if 0 <= threshold_idx < n:
            anom_log_area = log_area[threshold_idx:]
            anom_log_conc = log_concentration[threshold_idx:]
            if len(anom_log_area) >= 5:
                anom_coef = np.polyfit(anom_log_area, anom_log_conc, 1)
        return {'threshold': threshold, 'sorted_values': sorted_values, 'log_concentration': log_concentration, 'log_area': log_area, 'background_slope': bg_coef[0] if bg_coef is not None else None, 'anomaly_slope': anom_coef[0] if anom_coef is not None else None, 'threshold_index': threshold_idx}
    def _calculate_high_value_threshold(self, values: pd.Series, quantile: float=0.9) -> float:
        valid_values = pd.to_numeric(values, errors='coerce').dropna()
        valid_values = valid_values[np.isfinite(valid_values)]
        if valid_values.empty:
            return 0.0
        try:
            q = float(valid_values.quantile(float(quantile)))
        except Exception:
            q = float(np.nanpercentile(valid_values.to_numpy(dtype=float), 90))
        if not np.isfinite(q):
            q = float(valid_values.median())
        min_val = float(valid_values.min())
        max_val = float(valid_values.max())
        if max_val <= min_val:
            return max_val
        return min(max(q, min_val), max_val)
    def _find_label_column(self, data: pd.DataFrame) -> Optional[str]:
        possible_label_cols = ['Ore', 'label', 'target', 'deposit', 'label_encoded', 'target_encoded', 'labeled', 'has_deposit', 'is_deposit']
        for col in possible_label_cols:
            if col in data.columns:
                return col
        for col in data.columns:
            col_lower = str(col).lower()
            if any((lc.lower() in col_lower for lc in possible_label_cols)):
                return str(col)
        return None
    def _infer_key_element_relation(self, element: str) -> Tuple[str, str]:
        elem_lower = str(element).strip().lower()
        target_name = str(self.target_deposit_type or 'Target Deposit').strip()
        target_keys = [str(k).strip().lower() for k in (self.key_mineralization_elements or []) if str(k).strip()]
        if any((k == elem_lower or k in elem_lower for k in target_keys)):
            return ('direct_indicator', f'{element} is a direct key element or primary mineralization indicator for {target_name}.')
        for combo, desc in (self.target_related_combos or {}).items():
            parts = [str(x).strip().lower() for x in str(combo).split('-') if str(x).strip()]
            if any((p == elem_lower or p in elem_lower for p in parts)):
                return ('associated_indicator', f'{element} is related to a typical element association of {target_name} and can serve as a cooperative prospecting indicator ({desc}).')
        return ('supporting_indicator', f'{element} is a supporting indicator in the analysis of {target_name} and can help identify wall-rock alteration, evolutionary differentiation, or peripheral mineralization responses.')
    def _analyze_element_anomalies(self, data: pd.DataFrame, element_cols: list, prefer_elements: Optional[List[str]]=None) -> Dict[str, Any]:
        anomalies = {}
        valid_element_cols = [elem for elem in element_cols if elem in data.columns]
        normalized_to_col = {str(elem).strip().lower(): elem for elem in valid_element_cols}
        key_element_cols: List[str] = []
        if prefer_elements is not None:
            prefer_lower = [str(k).strip().lower() for k in prefer_elements if str(k).strip()]
            for k in prefer_lower:
                col = normalized_to_col.get(k)
                if col is not None and col not in key_element_cols:
                    key_element_cols.append(col)
        else:
            key_targets_lower = [str(k).strip().lower() for k in (self.key_mineralization_elements or []) if str(k).strip()]
            if key_targets_lower:
                for elem in valid_element_cols:
                    elem_lower = elem.lower()
                    if any((k in elem_lower for k in key_targets_lower)):
                        key_element_cols.append(elem)
            else:
                key_element_cols = list(valid_element_cols)
        self.logger.info(f"Analyzing {len(key_element_cols)} key elements: {', '.join(key_element_cols)}")
        label_col = self._find_label_column(data)
        deposit_mask = None
        deposit_count = 0
        if label_col is not None and label_col in data.columns:
            try:
                deposit_mask = (pd.to_numeric(data[label_col], errors='coerce').fillna(0) == 1).to_numpy()
                deposit_count = int(np.sum(deposit_mask))
            except Exception:
                deposit_mask = None
                deposit_count = 0
        for element in key_element_cols:
            self.logger.info(f'Analyzing key element: {element}')
            values = pd.to_numeric(data[element], errors='coerce')
            threshold = self._calculate_high_value_threshold(values)
            mean_val = float(values.mean()) if values.notna().any() else 0.0
            std_val = float(values.std()) if values.notna().any() else 0.0
            max_val = float(values.max()) if values.notna().any() else 0.0
            mask = values >= threshold
            anomaly_count = int(mask.sum())
            anomaly_samples = data.index[mask].tolist()
            total_count = int(len(data))
            anomaly_pct = anomaly_count / total_count * 100 if total_count > 0 else 0.0
            overlap_count = 0
            overlap_pct = 0.0
            if deposit_mask is not None and len(deposit_mask) == len(mask):
                try:
                    overlap_count = int(np.sum(mask.to_numpy() & deposit_mask))
                    overlap_pct = overlap_count / deposit_count * 100 if deposit_count > 0 else 0.0
                except Exception:
                    overlap_count = 0
                    overlap_pct = 0.0
            relation_type, relation_text = self._infer_key_element_relation(element)
            anomalies[element] = {
                'threshold': threshold,
                'threshold_index': 0,
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_pct,
                'high_value_count': anomaly_count,
                'high_value_percentage': anomaly_pct,
                'anomaly_samples': anomaly_samples,
                'known_deposit_overlap_count': overlap_count,
                'known_deposit_support_pct': overlap_pct,
                'relation_type': relation_type,
                'relation_text': relation_text,
                'method': 'Key-element analysis',
            }
        relationship_summary = [str(v.get('relation_text')) for v in anomalies.values() if isinstance(v, dict) and v.get('relation_text')]
        return {
            'element_anomalies': anomalies,
            'element_statistics': anomalies,
            'key_element_cols': key_element_cols,
            'method': 'Key-element analysis',
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'relationship_summary': relationship_summary,
        }
    def _analyze_element_associations(self, data: pd.DataFrame, prefer_elements: Optional[List[str]]=None) -> Dict[str, Any]:
        associations = {}
        target_specific_results = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        exclude_cols = ['FID', 'Ore', 'Longitude', 'Latitude']
        all_element_cols = [col for col in numeric_cols if col not in exclude_cols]
        normalized_to_col = {str(elem).strip().lower(): elem for elem in all_element_cols}
        key_element_cols: List[str] = []
        target_key_elements = [str(k).strip().lower() for k in (self.key_mineralization_elements or []) if str(k).strip()]
        if prefer_elements is not None:
            prefer_lower = [str(k).strip().lower() for k in prefer_elements if str(k).strip()]
            for k in prefer_lower:
                col = normalized_to_col.get(k)
                if col is not None and col not in key_element_cols:
                    key_element_cols.append(col)
        else:
            if target_key_elements:
                for elem in all_element_cols:
                    elem_lower = elem.lower()
                    if any((k in elem_lower for k in target_key_elements)):
                        key_element_cols.append(elem)
            else:
                key_element_cols = list(all_element_cols)
        target_name = self.target_deposit_type if self.target_deposit_type else 'Target Deposit'
        self.logger.info(f'Analyzing associations among {len(key_element_cols)} key mineralization elements, focusing on {target_name} characteristic combinations')
        for i in range(len(key_element_cols)):
            for j in range(i + 1, len(key_element_cols)):
                elem1 = key_element_cols[i]
                elem2 = key_element_cols[j]
                assoc_name = f'{elem1}-{elem2}'
                correlation = data[[elem1, elem2]].corr().iloc[0, 1]
                associations[assoc_name] = {'correlation': correlation, 'elements': [elem1, elem2]}
        sorted_associations = sorted(associations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        top_associations = dict(sorted_associations)
        if len(key_element_cols) >= 2 and self.target_related_combos:
            corr_matrix = data[key_element_cols].corr()
            target_combos_found = []
            for combo, description in self.target_related_combos.items():
                elements = combo.split('-')
                if all((any((e.lower() in col.lower() for col in key_element_cols)) for e in elements)):
                    combo_elements_in_data = []
                    for e in elements:
                        for col in key_element_cols:
                            if e.lower() in col.lower():
                                combo_elements_in_data.append(col)
                                break
                    if len(combo_elements_in_data) >= 2:
                        combo_corr_sum = 0
                        combo_corr_count = 0
                        for i in range(len(combo_elements_in_data)):
                            for j in range(i + 1, len(combo_elements_in_data)):
                                combo_corr_sum += abs(corr_matrix.loc[combo_elements_in_data[i], combo_elements_in_data[j]])
                                combo_corr_count += 1
                        avg_corr = combo_corr_sum / combo_corr_count if combo_corr_count > 0 else 0
                        target_combos_found.append({'combo': combo, 'description': description, 'avg_correlation': avg_corr, 'elements_in_data': combo_elements_in_data})
            target_specific_results['identified_target_combos'] = target_combos_found
            main_element = target_key_elements[0] if target_key_elements else None
            if main_element:
                main_elements_in_data = [col for col in key_element_cols if main_element in col.lower()]
                if main_elements_in_data:
                    anomaly_associations = []
                    for m_elem in main_elements_in_data:
                        for other_elem in key_element_cols:
                            if other_elem != m_elem:
                                corr_val = corr_matrix.loc[m_elem, other_elem]
                                anomaly_associations.append({'main_element': m_elem, 'associated_element': other_elem, 'correlation': corr_val})
                    anomaly_associations.sort(key=lambda x: abs(x['correlation']), reverse=True)
                    target_specific_results['main_element_associations'] = anomaly_associations
        results = {'top_element_pairs': top_associations, 'total_pairs_analyzed': len(associations), 'key_elements_analyzed': key_element_cols, 'target_specific_analysis': target_specific_results}
        if target_specific_results.get('identified_target_combos'):
            self.logger.info(f"Detected {len(target_specific_results['identified_target_combos'])} characteristic element combinations for {target_name}")
        return results
    def _identify_potential_mineralization_areas(self, data: pd.DataFrame, anomaly_analysis: Dict) -> List[int]:
        potential_areas_set = set()
        for element, stats in anomaly_analysis['element_anomalies'].items():
            if element in data.columns and 'threshold' in stats:
                element_anomaly_areas = data[data[element] >= stats['threshold']].index.tolist()
                potential_areas_set.update(element_anomaly_areas)
        return list(potential_areas_set)
    def _calculate_anomaly_score(self, data: pd.DataFrame, element: str, threshold: float) -> pd.Series:
        if element not in data.columns:
            return pd.Series(index=data.index, dtype=float, name=f'{element}_anomaly_score')
        anomaly_score = np.where(data[element] >= threshold, (data[element] / threshold) ** 2, data[element] / threshold)
        return pd.Series(anomaly_score, index=data.index, name=f'{element}_anomaly_score')
    def _identify_ca_patterns(self, anomaly_analysis: Dict) -> Dict[str, Any]:
        patterns: Dict[str, Any] = {'strong_anomaly_elements': [], 'background_vs_anomaly_slopes': {}, 'element_groups': {'high_anomaly_percentage': [], 'medium_anomaly_percentage': [], 'low_anomaly_percentage': []}}
        for element, stats in anomaly_analysis['element_anomalies'].items():
            if stats['anomaly_percentage'] > 5:
                patterns['strong_anomaly_elements'].append({'element': element, 'anomaly_percentage': stats['anomaly_percentage'], 'threshold': stats['threshold']})
            if stats['background_slope'] is not None and stats['anomaly_slope'] is not None:
                patterns['background_vs_anomaly_slopes'][element] = {'background_slope': stats['background_slope'], 'anomaly_slope': stats['anomaly_slope'], 'slope_ratio': stats['anomaly_slope'] / stats['background_slope'] if stats['background_slope'] != 0 else float('inf')}
            if stats['anomaly_percentage'] > 10:
                patterns['element_groups']['high_anomaly_percentage'].append(element)
            elif stats['anomaly_percentage'] > 2:
                patterns['element_groups']['medium_anomaly_percentage'].append(element)
            else:
                patterns['element_groups']['low_anomaly_percentage'].append(element)
        patterns['strong_anomaly_elements'].sort(key=lambda x: x['anomaly_percentage'], reverse=True)
        return patterns
    def _perform_detailed_assessment(self, data: pd.DataFrame, anomaly_analysis: Dict, feature_results: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        element_anomalies_obj = anomaly_analysis.get('element_anomalies', {}) if isinstance(anomaly_analysis, dict) else {}
        element_anomalies = element_anomalies_obj if isinstance(element_anomalies_obj, dict) else {}
        sorted_elements = sorted(element_anomalies.items(), key=lambda x: x[1].get('anomaly_percentage', 0), reverse=True)
        feature_contribution = {}
        if feature_results and 'feature_importance' in feature_results:
            for element, _ in sorted_elements:
                if element in feature_results['feature_importance']:
                    all_importances = feature_results['feature_importance'][element].get('all_importances', {})
                    sorted_features = sorted(all_importances.items(), key=lambda x: x[1], reverse=True)
                    feature_contribution[element] = [f'{f}({imp:.2f})'.replace('_', '') for f, imp in sorted_features]
        return {'feature_contribution': feature_contribution, 'recommended_focus': self._determine_recommended_focus(anomaly_analysis)}
    def _select_target_related_elements(
        self,
        element_cols: List[str],
        feature_results: Dict[str, Any],
        anomaly_analysis: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        use_target_knowledge = self._knowledge_enabled(config)
        union_order: List[str] = []
        sources: Dict[str, List[str]] = {
            'target_key_elements': [],
            'correlation_related': [],
            'factor_related': [],
            'cluster_related': [],
            'anomaly_related': [],
        }
        def _add(items: List[str], key: str) -> None:
            for item in items:
                if item not in union_order:
                    union_order.append(item)
                if item not in sources[key]:
                    sources[key].append(item)
        keywords: List[str] = []
        if use_target_knowledge:
            for elem in self.key_mineralization_elements:
                if elem:
                    keywords.append(str(elem).strip().lower())
            for combo in self.target_related_combos.keys():
                for part in str(combo).split('-'):
                    part = part.strip().lower()
                    if part:
                        keywords.append(part)
        keywords = list(dict.fromkeys(keywords))
        base_targets: List[str] = []
        if use_target_knowledge and keywords:
            for col in element_cols:
                col_lower = str(col).lower()
                if any((k in col_lower for k in keywords)):
                    base_targets.append(col)
        _add(base_targets, 'target_key_elements')
        pearson = feature_results.get('correlation_analysis', {}).get('pearson', {})
        high_corrs = pearson.get('high_correlations', [])
        if high_corrs:
            corr_related: List[str] = []
            for item in high_corrs:
                pair = item.get('elements')
                if not pair or '-' not in pair:
                    continue
                elem1, elem2 = [p.strip() for p in pair.split('-', 1)]
                if not elem1 or not elem2:
                    continue
                elem1_lower = elem1.lower()
                elem2_lower = elem2.lower()
                is_elem1_target = (elem1 in base_targets) or (keywords and any((k in elem1_lower for k in keywords)))
                is_elem2_target = (elem2 in base_targets) or (keywords and any((k in elem2_lower for k in keywords)))
                if is_elem1_target and elem2 in element_cols:
                    corr_related.append(elem2)
                if is_elem2_target and elem1 in element_cols:
                    corr_related.append(elem1)
            _add(corr_related, 'correlation_related')
        def _infer_mineral_anchor_symbol() -> str:
            if not use_target_knowledge:
                return ''
            target_name = str(self.target_deposit_type or '').strip()
            mapping = [
                ('tungsten', 'W'),
                ('copper', 'Cu'),
                ('gold', 'Au'),
                ('silver', 'Ag'),
                ('lead', 'Pb'),
                ('zinc', 'Zn'),
                ('tin', 'Sn'),
                ('molybdenum', 'Mo'),
                ('antimony', 'Sb'),
                ('arsenic', 'As'),
                ('bismuth', 'Bi'),
            ]
            for key, symbol in mapping:
                if key and key in target_name:
                    return str(symbol)
            return ''
        hc = feature_results.get('hierarchical_clustering', {}) if isinstance(feature_results, dict) else {}
        if isinstance(hc, dict):
            cluster_labels_obj = hc.get('cluster_labels', {})
            if isinstance(cluster_labels_obj, dict) and cluster_labels_obj:
                cluster_labels: Dict[str, int] = {}
                for k, v in cluster_labels_obj.items():
                    try:
                        kk = str(k)
                        vv = int(v)
                        cluster_labels[kk] = vv
                    except Exception:
                        continue
                if cluster_labels:
                    key_map = {str(k).lower(): str(k) for k in cluster_labels.keys()}
                    anchor_symbol = _infer_mineral_anchor_symbol()
                    anchor_keys: List[str] = []
                    if anchor_symbol:
                        anchor_low = anchor_symbol.strip().lower()
                        if anchor_symbol in cluster_labels:
                            anchor_keys = [str(anchor_symbol)]
                        elif anchor_low in key_map:
                            anchor_keys = [str(key_map[anchor_low])]
                        else:
                            contains = [orig for low, orig in key_map.items() if anchor_low and anchor_low in low]
                            if contains:
                                anchor_keys = [str(contains[0])]
                    cluster_ids = {cluster_labels[k] for k in anchor_keys if k in cluster_labels}
                    if cluster_ids:
                        related = [e for e, cid in cluster_labels.items() if (cid in cluster_ids and e in element_cols)]
                        _add(related, 'cluster_related')
        def _infer_anchor_symbols() -> List[str]:
            if not use_target_knowledge:
                return []
            target_name = str(self.target_deposit_type or '').strip()
            anchors: List[str] = []
            mapping = [
                ('tungsten', 'W'),
                ('copper', 'Cu'),
                ('gold', 'Au'),
                ('silver', 'Ag'),
                ('lead', 'Pb'),
                ('zinc', 'Zn'),
                ('tin', 'Sn'),
                ('molybdenum', 'Mo'),
                ('antimony', 'Sb'),
                ('arsenic', 'As'),
                ('bismuth', 'Bi'),
            ]
            for key, symbol in mapping:
                if key and key in target_name:
                    anchors.append(symbol)
            if not anchors:
                for elem in (self.key_mineralization_elements or []):
                    s = str(elem).strip()
                    if s:
                        anchors.append(s)
            return list(dict.fromkeys(anchors))
        def _match_rows(df: pd.DataFrame, symbol: str) -> List[str]:
            sym = str(symbol).strip().lower()
            if not sym:
                return []
            idxs = [str(i) for i in df.index.tolist()]
            exact = [i for i in idxs if str(i).strip().lower() == sym]
            if exact:
                return exact
            contains = [i for i in idxs if sym in str(i).strip().lower()]
            return contains
        def _select_from_factor_loadings_target_positive(loadings: Dict[str, Any], max_factor_count: int=1) -> None:
            if not isinstance(loadings, dict) or not loadings:
                return
            load_df = pd.DataFrame(loadings)
            if load_df.empty:
                return
            for col in load_df.columns:
                load_df[col] = pd.to_numeric(load_df[col], errors='coerce')
            factors = list(load_df.columns)
            if not factors:
                return
            related: List[str] = []
            selected_factors: List[str] = []
            for anchor in _infer_anchor_symbols():
                rows = _match_rows(load_df, anchor)
                if not rows:
                    continue
                anchor_vals = {}
                for f in factors:
                    vals = pd.to_numeric(load_df.loc[rows, f], errors='coerce')
                    try:
                        v = float(np.nanmax(vals.values.astype(float)))
                    except Exception:
                        v = float('nan')
                    anchor_vals[f] = v
                valid_items = [(f, v) for f, v in anchor_vals.items() if v == v]
                if not valid_items:
                    continue
                pos_items = [(f, v) for f, v in valid_items if v > 0]
                if pos_items:
                    pos_items.sort(key=lambda x: x[1], reverse=True)
                    selected_factors = [pos_items[0][0]]
                else:
                    valid_items.sort(key=lambda x: abs(x[1]), reverse=True)
                    selected_factors = [valid_items[0][0]]
                break
            if not selected_factors:
                selected_factors = factors[:1]
            for f in selected_factors:
                series = load_df[f].dropna()
                series = series[series > 0].sort_values(ascending=False)
                if series.empty:
                    continue
                for elem in series.index.tolist():
                    if elem in element_cols:
                        related.append(elem)
            if element_cols and len(related) > (len(element_cols) / 2.0):
                keep_set: set = set()
                for f in selected_factors:
                    series = load_df[f].dropna()
                    series = series[series > 0.5].sort_values(ascending=False)
                    if series.empty:
                        continue
                    for elem in series.index.tolist():
                        if elem in element_cols:
                            keep_set.add(elem)
                target_keep: List[str] = []
                for anchor in _infer_anchor_symbols():
                    a = str(anchor).strip().lower()
                    if not a:
                        continue
                    for col in element_cols:
                        col_lower = str(col).lower()
                        if a == col_lower or a in col_lower:
                            target_keep.append(col)
                for elem in target_keep:
                    keep_set.add(elem)
                if keep_set:
                    filtered_related: List[str] = []
                    for elem in related:
                        if elem in keep_set and elem not in filtered_related:
                            filtered_related.append(elem)
                    for elem in target_keep:
                        if elem in keep_set and elem not in filtered_related:
                            filtered_related.append(elem)
                    related = filtered_related
            _add(related, 'factor_related')
        _select_from_factor_loadings_target_positive(feature_results.get('factor_analysis', {}).get('factor_loadings', {}))
        selection_keys = ['correlation_related', 'factor_related', 'cluster_related']
        if use_target_knowledge:
            selection_keys.insert(0, 'target_key_elements')
        selected_elements: List[str] = []
        counts_by_element: Dict[str, int] = {}
        for k in selection_keys:
            vals = sources.get(k)
            if not isinstance(vals, list):
                continue
            for v in vals:
                counts_by_element[v] = counts_by_element.get(v, 0) + 1
        min_votes = 2 if len(selection_keys) >= 2 else 1
        selected_elements = [e for e in union_order if counts_by_element.get(e, 0) >= min_votes]
        selected_elements = sorted(selected_elements, key=lambda x: str(x).upper())
        selection_mode = f'at_least_{min_votes}_of_{len(selection_keys)}'
        try:
            counts = {k: (len(v) if isinstance(v, list) else 0) for k, v in sources.items()}
            self.logger.info(
                f"Target-related element selection mode: {selection_mode}; "
                f"selection_keys={selection_keys}; "
                f"counts={counts}; "
                f"selected={len(selected_elements)}; "
                f"knowledge_enabled={use_target_knowledge}"
            )
        except Exception:
            pass
        return {
            'selected_elements': selected_elements,
            'sources': sources,
            'selection_keys': selection_keys,
            'selection_mode': selection_mode,
            'knowledge_enabled': use_target_knowledge,
        }
    def _save_target_element_sources_csv(self, target_element_selection: Dict[str, Any]) -> str:
        try:
            sources_obj = target_element_selection.get('sources', {}) if isinstance(target_element_selection, dict) else {}
            sources = sources_obj if isinstance(sources_obj, dict) else {}
            selected_elements_obj = target_element_selection.get('selected_elements', []) if isinstance(target_element_selection, dict) else []
            selected_elements = {str(x).strip() for x in selected_elements_obj if str(x).strip()} if isinstance(selected_elements_obj, list) else set()
            rows: List[Dict[str, Any]] = []
            source_name_map = {
                'target_key_elements': 'Target-deposit key elements',
                'correlation_related': 'Correlation analysis',
                'factor_related': 'Factor analysis',
                'cluster_related': 'Hierarchical clustering',
                'anomaly_related': 'Anomaly analysis',
            }
            for source_key, vals_obj in sources.items():
                vals = [str(x).strip() for x in vals_obj] if isinstance(vals_obj, list) else []
                for elem in vals:
                    if not elem:
                        continue
                    rows.append({
                        'source_key': str(source_key),
                        'source_name': source_name_map.get(str(source_key), str(source_key)),
                        'element': elem,
                        'is_selected': 1 if elem in selected_elements else 0,
                    })
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            out_path = os.path.join(reports_dir, 'target_element_selection_sources.csv')
            df_out = pd.DataFrame(rows, columns=['source_key', 'source_name', 'element', 'is_selected'])
            if not df_out.empty:
                df_out = df_out.sort_values(by=['source_key', 'element'], ascending=[True, True]).reset_index(drop=True)
            df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
            self.logger.info(f'CSV of target-related elements from each method saved to: {out_path}')
            return out_path
        except Exception as e:
            self.logger.warning(f'Failed to save the CSV of target-related elements from each method: {e}')
            return ''
    def _provide_final_interpretation(self, data: pd.DataFrame, anomaly_analysis: Dict, prediction_results: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        result: Dict[str, Any] = {'mineralization_types': [], 'confidence_level': 'low', 'prediction_integration': {}, 'recommendations': [], 'target_metallogeny_analysis': {}}
        element_anomalies = anomaly_analysis['element_anomalies']
        target_key_elements = [k.lower() for k in self.key_mineralization_elements]
        has_target_anomaly = False
        if not target_key_elements and element_anomalies:
            sorted_anomalies = sorted(element_anomalies.items(), key=lambda x: x[1]['anomaly_percentage'], reverse=True)
            target_key_elements = [k.lower() for k, v in sorted_anomalies]
        for elem in element_anomalies:
            if any((key_elem in elem.lower() for key_elem in target_key_elements)) and element_anomalies[elem]['anomaly_percentage'] > 3:
                has_target_anomaly = True
                break
        target_name = self.target_deposit_type if self.target_deposit_type else 'Unknown Deposit'
        if has_target_anomaly:
            self.logger.info(f'Running deposit-specific mineralization analysis for {target_name}...')
            mineralization_types: List[str] = []
            if self.target_related_combos:
                for combo, desc in self.target_related_combos.items():
                    elements = combo.split('-')
                    combo_anomaly = True
                    for e in elements:
                        e_found = False
                        for anomaly_elem in element_anomalies:
                            if e.lower() in anomaly_elem.lower() and element_anomalies[anomaly_elem]['anomaly_percentage'] > 3:
                                e_found = True
                                break
                        if not e_found:
                            combo_anomaly = False
                            break
                    if combo_anomaly:
                        mineralization_types.append(f'{combo} mineralization ({desc})')
            if not mineralization_types:
                mineralization_types.append(f'{target_name} mineralization')
            result['mineralization_types'] = mineralization_types
            result['confidence_level'] = 'high' if len(mineralization_types) > 1 else 'medium'
            result['target_metallogeny_analysis']['identified_types'] = mineralization_types
            self.logger.info(f"Identified mineralization types: {', '.join(mineralization_types)}")
        else:
            mineralization_types = self._infer_mineralization_types(anomaly_analysis)
            result['mineralization_types'] = mineralization_types
            result['confidence_level'] = self._calculate_confidence_level(anomaly_analysis)
        if prediction_results:
            high_potential_count = prediction_results.get('high_potential_count', 0)
            result['prediction_integration']['predicted_high_potential_areas'] = high_potential_count
            if high_potential_count > len(data) * 0.1:
                if result['confidence_level'] == 'low':
                    result['confidence_level'] = 'medium'
            if has_target_anomaly:
                result['target_metallogeny_analysis']['has_high_potential_areas'] = high_potential_count > 0
        if has_target_anomaly:
            result['recommendations'] = [f"Prioritize validation of the relationship between {', '.join(self.key_mineralization_elements)} and structural controls", 'Perform denser sampling for characteristic element combinations', f'Analyze the mineralization regularity of {target_name} together with the geological background', 'Carry out targeted alteration-zone mapping', 'Conduct deep validation in high-value zones of key elements']
        else:
            result['recommendations'] = self._generate_exploration_recommendations(anomaly_analysis, result['prediction_integration'])
        return result
    def _infer_mineralization_types(self, anomaly_analysis: Dict) -> List[str]:
        mineralization_types: List[str] = []
        element_anomalies = anomaly_analysis.get('element_anomalies', {}) if isinstance(anomaly_analysis, dict) else {}
        if not isinstance(element_anomalies, dict) or not element_anomalies:
            return mineralization_types
        significant: List[Tuple[str, float]] = []
        for elem, stats in element_anomalies.items():
            if not isinstance(stats, dict):
                continue
            try:
                pct = float(stats.get('anomaly_percentage', 0))
            except Exception:
                pct = 0.0
            if pct > 3.0:
                significant.append((str(elem), pct))
        if not significant:
            return mineralization_types
        significant.sort(key=lambda x: x[1], reverse=True)
        target_name = str(self.target_deposit_type or '').strip()
        lead_elements = [e for e, _ in significant[:3]]
        if target_name:
            mineralization_types.append(f'{target_name}-related mineralization anomaly')
        if len(lead_elements) >= 2:
            mineralization_types.append(f"{'-'.join(lead_elements)} multi-element mineralization anomaly")
        else:
            mineralization_types.append(f'{lead_elements[0]}-dominated mineralization anomaly')
        if len(significant) >= 4:
            mineralization_types.append('Regional multi-element cooperative mineralization anomaly')
        return mineralization_types
    def _calculate_confidence_level(self, anomaly_analysis: Dict) -> str:
        anomaly_elements_count = len(anomaly_analysis['element_anomalies'])
        max_anomaly_percentage = max((stats['anomaly_percentage'] for stats in anomaly_analysis['element_anomalies'].values()))
        if anomaly_elements_count >= 4 and max_anomaly_percentage > 10:
            return 'high'
        elif anomaly_elements_count >= 2 and max_anomaly_percentage > 5:
            return 'medium'
        else:
            return 'low'
    def _generate_exploration_recommendations(self, anomaly_analysis: Dict, prediction_integration: Optional[Dict[str, Any]]=None) -> List[str]:
        recommendations: List[str] = []
        prioritized_elements = [elem for elem, stats in anomaly_analysis['element_anomalies'].items() if stats['anomaly_percentage'] > 5]
        if prioritized_elements:
            recommendations.append(f"Prioritize exploration in high-value zones of {', '.join(prioritized_elements)}")
        strong_anomaly_elements: List[str] = []
        for element, stats in anomaly_analysis['element_anomalies'].items():
            if 'mean' in stats and 'max' in stats and (stats['max'] > stats['mean'] * 3):
                strong_anomaly_elements.append(element)
        if strong_anomaly_elements:
            recommendations.append(f"High-value enrichment of {', '.join(strong_anomaly_elements)} was detected; drilling verification is recommended")
        if prediction_integration:
            predicted_high_potential = prediction_integration.get('predicted_high_potential_areas') or prediction_integration.get('high_potential_count') or 0
            if predicted_high_potential > 0:
                recommendations.append(f'The predictive model identified {predicted_high_potential} high-potential areas; prioritize validation of these areas')
        return recommendations
    def _determine_recommended_focus(self, anomaly_analysis: Dict) -> List[str]:
        return ['Analyze spatial patterns of key elements', 'Study structural controls', 'Investigate geochronological constraints on mineralization']
    def interpret(self, data: pd.DataFrame, element_cols: List[str], ca_results: Dict[str, Any], anomaly_analysis: Dict[str, Any], prediction_results: Optional[Dict[str, Any]]=None, stage: str='final') -> Dict[str, Any]:
        self.logger.info(f'{stage.capitalize()} Geological Interpretation Started...')
        results: Dict[str, Any] = {'stage': stage, 'mineralization_types': [], 'metallogenic_regularity': {}, 'element_associations': {}, 'spatial_distribution': {}, 'confidence_level': 'low', 'recommendations': [], 'summary': ''}
        self.logger.info('Inferring mineralization types...')
        mineralization_types = self._infer_mineralization_types(anomaly_analysis)
        results['mineralization_types'] = mineralization_types
        self.logger.info('Analyzing metallogenic regularity...')
        metallogenic_regularity = self._analyze_metallogenic_regularity(anomaly_analysis, ca_results)
        results['metallogenic_regularity'] = metallogenic_regularity
        self.logger.info('Analyzing element associations...')
        prefer_elements = list(ca_results.keys()) if isinstance(ca_results, dict) else []
        element_associations = self._analyze_element_associations(data, prefer_elements=prefer_elements)
        results['element_associations'] = element_associations
        self.logger.info('Analyzing spatial distribution...')
        spatial_distribution = self._analyze_spatial_distribution(data, element_cols, ca_results)
        results['spatial_distribution'] = spatial_distribution
        results['confidence_level'] = self._calculate_confidence_level(anomaly_analysis)
        results['recommendations'] = self._generate_exploration_recommendations(anomaly_analysis, prediction_results)
        results['summary'] = self._generate_interpretation_summary(results)
        self.logger.info(f'{stage.capitalize()} Geological Interpretation Completed.')
        return results
    def _analyze_metallogenic_regularity(self, anomaly_analysis: Dict, ca_results: Dict[str, Any]) -> Dict[str, Any]:
        regularity: Dict[str, Any] = {'element_distribution_patterns': {}, 'anomaly_concentration_zones': [], 'metallogenic_types': []}
        element_anomalies = anomaly_analysis['element_anomalies']
        for element, stats in element_anomalies.items():
            regularity['element_distribution_patterns'][element] = {'anomaly_percentage': stats['anomaly_percentage'], 'mean_concentration': stats['mean'], 'max_concentration': stats['max'], 'threshold': ca_results[element]['threshold']}
        high_anomaly_elements = [elem for elem, stats in element_anomalies.items() if stats['anomaly_percentage'] > 5]
        regularity['anomaly_concentration_zones'] = high_anomaly_elements
        if len(high_anomaly_elements) >= 3:
            top_combo = '-'.join([str(e) for e in high_anomaly_elements[:3]])
            regularity['metallogenic_types'].append(f'{top_combo} multi-element cooperative mineralization anomaly')
        elif len(high_anomaly_elements) == 2:
            regularity['metallogenic_types'].append(f'{high_anomaly_elements[0]}-{high_anomaly_elements[1]} dual-element cooperative mineralization anomaly')
        elif len(high_anomaly_elements) == 1:
            regularity['metallogenic_types'].append(f'{high_anomaly_elements[0]}-dominated mineralization anomaly')
        target_name = str(self.target_deposit_type or '').strip()
        if target_name and high_anomaly_elements:
            regularity['metallogenic_types'].append(f'{target_name}-related anomaly assemblage')
        return regularity
    def _analyze_spatial_distribution(self, data: pd.DataFrame, element_cols: List[str], ca_results: Dict[str, Any]) -> Dict[str, Any]:
        spatial_results: Dict[str, Any] = {'has_coordinate_data': False, 'anomaly_clusters': {}, 'distribution_patterns': []}
        possible_coordinate_cols = ['Longitude', 'Latitude', 'longitude', 'latitude', 'X', 'Y', 'x', 'y']
        coord_cols = [col for col in data.columns if col in possible_coordinate_cols]
        if len(coord_cols) >= 2:
            spatial_results['has_coordinate_data'] = True
            for element in element_cols[:5]:
                if element in ca_results:
                    threshold = ca_results[element]['threshold']
                    anomaly_data = data[data[element] > threshold]
                    if len(anomaly_data) > 0:
                        coords = anomaly_data[coord_cols].values
                        if len(coords) > 2:
                            kmeans = KMeans(n_clusters=min(3, len(coords)), random_state=42)
                            clusters = kmeans.fit_predict(coords)
                            spatial_results['anomaly_clusters'][element] = {'cluster_count': len(set(clusters)), 'sample_count': len(anomaly_data)}
        spatial_results['distribution_patterns'] = self._determine_distribution_patterns(ca_results)
        return spatial_results
    def _determine_distribution_patterns(self, ca_results: Dict) -> List[str]:
        patterns = []
        percentages = [ca_results[elem]['anomaly_percentage'] for elem in ca_results]
        mean_percentage = np.mean(percentages)
        max_percentage = np.max(percentages)
        if max_percentage > 20:
            patterns.append('Strongly concentrated anomaly distribution')
        elif mean_percentage > 10:
            patterns.append('Broad anomaly distribution')
        else:
            patterns.append('Sparse anomaly distribution')
        return patterns
    def _generate_interpretation_summary(self, results: Dict[str, Any]) -> str:
        summary = f"Geological interpretation results ({results['stage']} stage):\n"
        summary += f"- Inferred mineralization types: {(', '.join(results['mineralization_types']) if results['mineralization_types'] else 'Unidentified')}\n"
        summary += f"- Metallogenic regularity: {(', '.join(results['metallogenic_regularity'].get('metallogenic_types', [])) if results['metallogenic_regularity'].get('metallogenic_types') else 'Unidentified')}\n"
        summary += f"- Key-element concentration zones: {', '.join(results['metallogenic_regularity'].get('anomaly_concentration_zones', []))}\n"
        summary += f"- Confidence level: {results['confidence_level']}\n"
        summary += f"- Number of exploration recommendations: {len(results['recommendations'])}\n"
        return summary
    def _get_stage_name(self, stage: str) -> str:
        stage_names = {'primary': 'primary', 'intermediate': 'intermediate', 'final': 'final'}
        return stage_names.get(stage, 'unknown')
    def _generate_primary_summary(self, anomaly_analysis: Dict, association_analysis: Dict, potential_areas: List) -> str:
        anomaly_count = len(anomaly_analysis['element_anomalies'])
        potential_count = len(potential_areas)
        summary = f'Primary-stage analysis identified high-value responses in {anomaly_count} key elements, '
        summary += f'and identified {potential_count} potential mineralization areas.'
        top_elements = sorted(anomaly_analysis['element_anomalies'].items(), key=lambda x: x[1]['anomaly_percentage'], reverse=True)[:3]
        if top_elements:
            element_strings = []
            for elem, stats in top_elements:
                element_strings.append(f"{elem}({stats['anomaly_percentage']:.2f}%)")
            summary += f"The main key elements are: {', '.join(element_strings)}."
        return summary
    def _generate_intermediate_summary(self, detailed_assessment: Dict) -> str:
        feature_contribution_obj = detailed_assessment.get('feature_contribution', {})
        feature_contribution = feature_contribution_obj if isinstance(feature_contribution_obj, dict) else {}
        assessed_elements = list(feature_contribution.keys())
        if assessed_elements:
            summary = f"Intermediate-stage analysis completed element-contribution assessment for: {', '.join(assessed_elements)}."
        else:
            summary = 'Intermediate-stage analysis completed the element-contribution assessment.'
        return summary
    def _generate_final_summary(self, final_interpretation: Dict) -> str:
        mineralization_types = final_interpretation['mineralization_types']
        confidence_level = final_interpretation['confidence_level']
        summary = f"Final interpretation result: identified {', '.join(mineralization_types)}, "
        summary += f'with a prediction confidence level of {confidence_level}.'
        return summary

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
        return "data:image/png;base64," + b64

    def analyze_spatial_potential_coupling(
        self,
        *,
        prediction_map_path: str,
        prediction_map_hotspots_path: Optional[str] = None,
        spatial_distribution_image_paths: Optional[List[str]] = None,
        element_ca_results: Optional[Dict[str, Any]] = None,
        target_related_elements: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not getattr(self, "llm", None):
            return {"markdown": "", "cot_steps": []}
        if not prediction_map_path:
            return {"markdown": "", "cot_steps": []}
        spatial_distribution_image_paths = spatial_distribution_image_paths or []
        element_ca_results = element_ca_results or {}
        target_related_elements = target_related_elements or []

        image_uris: List[Dict[str, Any]] = []
        pm_uri = self._image_to_data_uri(prediction_map_path)
        if pm_uri:
            image_uris.append({"type": "image_url", "image_url": {"url": pm_uri}})
        pmh_uri = self._image_to_data_uri(prediction_map_hotspots_path) if prediction_map_hotspots_path else None
        if pmh_uri:
            image_uris.append({"type": "image_url", "image_url": {"url": pmh_uri}})

        spatial_meta_lines: List[str] = []
        for p in spatial_distribution_image_paths:
            try:
                base = os.path.basename(str(p))
                if "_spatial_distribution" in base:
                    elem = base.split("_spatial_distribution")[0].strip()
                else:
                    elem = base.split("_spatial_anomaly")[0].strip()
                if elem:
                    spatial_meta_lines.append(f"- {elem}: {base}")
                else:
                    spatial_meta_lines.append(f"- {base}")
            except Exception:
                spatial_meta_lines.append(f"- {str(p)}")

        if len(image_uris) < 1:
            return {"markdown": "", "cot_steps": []}

        stats_lines: List[str] = []
        if isinstance(element_ca_results, dict) and element_ca_results:
            for elem in target_related_elements:
                if elem in element_ca_results:
                    st = element_ca_results.get(elem) or {}
                    try:
                        thr = st.get("threshold")
                        ap = st.get("anomaly_percentage")
                        rel = str(st.get("relation_text") or "").strip()
                        thr_s = f"{float(thr):.6g}" if thr is not None else "NA"
                        ap_s = f"{float(ap):.3g}%" if ap is not None else "NA"
                        stats_lines.append(f"- {elem}: high-value threshold={thr_s}; high-value proportion={ap_s}; relation={rel or 'not provided'}")
                    except Exception:
                        stats_lines.append(f"- {elem}: (failed to parse key-element statistics)")
            if not stats_lines:
                items = list(element_ca_results.items())[:6]
                for elem, st in items:
                    if not isinstance(st, dict):
                        continue
                    try:
                        thr = st.get("threshold")
                        ap = st.get("anomaly_percentage")
                        rel = str(st.get("relation_text") or "").strip()
                        thr_s = f"{float(thr):.6g}" if thr is not None else "NA"
                        ap_s = f"{float(ap):.3g}%" if ap is not None else "NA"
                        stats_lines.append(f"- {elem}: high-value threshold={thr_s}; high-value proportion={ap_s}; relation={rel or 'not provided'}")
                    except Exception:
                        stats_lines.append(f"- {elem}: (failed to parse key-element statistics)")

        target_elems_str = ', '.join([str(x) for x in target_related_elements if str(x).strip()][:15])
        if not target_elems_str:
            target_elems_str = "not provided"
        stats_block = "\n".join(stats_lines) if stats_lines else "- No key-element threshold or high-value proportion information was provided"
        spatial_meta = "\n".join(spatial_meta_lines) if spatial_meta_lines else "- No key-element spatial-distribution image list was provided"

        base_prompt_text = (
            "You are a senior economic geologist. Your task is to produce a coupled interpretation of key-element spatial distributions and the mineralization-potential prediction map, and return Markdown that can be inserted directly into the integrated report.\n\n"
            "Input figure description:\n"
            "1) Figure 1: mineralization-potential prediction map (probability heatmap).\n"
            "2) Figure 2 (if available): mineralization-potential prediction map with hotspot annotations.\n"
            "3) Key-element spatial-distribution maps: the model is not given raw image pixels for these maps in this step; only the image list and a summary of high-value thresholds, high-value proportions, and element relationships are provided.\n\n"
            f"Target-related elements (for reference): {target_elems_str}\n\n"
            "Key-element thresholds / high-value proportions / element relationships (for reference):\n"
            f"{stats_block}\n\n"
            "Spatial-anomaly image list (for reference):\n"
            f"{spatial_meta}\n\n"
            "Writing requirements:\n"
            "1) The output must contain the following subsection headings in this order:\n"
            "   - Overlay Consistency (hotspots with strong anomaly support)\n"
            "   - Overlay Inconsistency (hotspots that require caution and possible reasons)\n"
            "   - Re-ranked Target Priorities And Validation Suggestions\n"
            "2) Keep the overlay analysis as specific as possible: describe the relative position of hotspots (for example northeast or west-central), their geometry (for example belt-like, clustered, or isolated), and identify which elemental anomalies support them.\n"
            "3) Do not invent legends, coordinate values, map scales, or deposit-point information that is not provided. If something cannot be determined from the figures, explicitly state 'uncertain'.\n"
            "4) Avoid long mechanistic speculation and focus on the chain from map evidence to target decision-making.\n"
        )
        max_steps = 12
        if isinstance(config, dict):
            try:
                ms = int(config.get("cot_max_steps", max_steps))
                if 1 <= ms <= 25:
                    max_steps = ms
            except Exception:
                pass
        cot_enabled = True
        if isinstance(config, dict) and "cot_enabled" in config:
            cot_enabled = bool(config.get("cot_enabled"))
        if (os.environ.get("GEOCHEM_COT_ENABLED") or os.environ.get("AGENTS_COT_ENABLED")) is not None:
            cot_enabled = str(os.environ.get("GEOCHEM_COT_ENABLED") or os.environ.get("AGENTS_COT_ENABLED")).strip().lower() in {"1", "true", "yes", "on"}

        prompt_text = base_prompt_text
        if cot_enabled:
            prompt_text = (
                base_prompt_text
                + "\n\n"
                + "You must output JSON only. Do not output any other text and do not use Markdown code fences.\n"
                + "Required fields:\n"
                + "- markdown: string, the Markdown body that satisfies the writing requirements above\n"
                + f"- cot_steps: string[], a step-by-step reasoning chain with one sentence per step and at most {int(max_steps)} steps; describe reasoning only, do not invent data, and write 'uncertain' when the evidence is insufficient.\n"
            )

        try:
            from langchain_core.messages import HumanMessage

            msg = HumanMessage(content=[{"type": "text", "text": prompt_text}, *image_uris])
            llm = self.llm
            resp = llm.invoke([msg])
            text = getattr(resp, "content", resp)
            text_str = str(text or "").strip()
            if not cot_enabled:
                return {"markdown": text_str, "cot_steps": []}
            parsed: Optional[Dict[str, Any]] = None
            try:
                parsed = json.loads(text_str) if text_str.startswith("{") else None
            except Exception:
                parsed = None
            if parsed is None:
                try:
                    m = re.search(r"\{[\s\S]*\}", text_str)
                    if m:
                        parsed = json.loads(m.group(0))
                except Exception:
                    parsed = None
            if isinstance(parsed, dict):
                md = parsed.get("markdown")
                cot_steps = parsed.get("cot_steps")
                md_s = str(md or "").strip() if not isinstance(md, str) else md.strip()
                cleaned_steps: List[str] = []
                if isinstance(cot_steps, list):
                    for s in cot_steps[:max_steps]:
                        if not isinstance(s, str):
                            continue
                        ss = s.strip()
                        if ss:
                            cleaned_steps.append(ss)
                return {"markdown": md_s, "cot_steps": cleaned_steps}
            return {"markdown": text_str, "cot_steps": []}
        except Exception as e:
            try:
                self.logger.warning(f"Spatial anomaly / mineralization-potential coupling interpretation failed: {e}")
            except Exception:
                pass
            return {"markdown": "", "cot_steps": []}
class CAVisualization:
    def __init__(self, output_dir='output/visualization_ca'):
        self.output_dir = output_dir
        self.viz_root_dirname = 'Key element analysis results'
        self._lang = _resolve_output_language()
        self._setup_matplotlib()
        os.makedirs(output_dir, exist_ok=True)
    def _setup_matplotlib(self):
        _setup_matplotlib_output_style(plt)
        sns.set_style('whitegrid')
        sns.set_palette('husl')
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Micro Hei']
        found_chinese_font = any((font in available_fonts for font in chinese_fonts))
        if not found_chinese_font:
            logger.warning('No CJK-compatible font was found. Some localized chart text may not render correctly.')
    def _calculate_c_a_threshold(self, values: pd.Series) -> float:
        valid_values = values.dropna()
        if valid_values.empty:
            return 0.0
        valid_values = valid_values[valid_values > 0]
        if valid_values.empty:
            return 0.0
        sorted_values = np.sort(valid_values)
        n = int(len(sorted_values))
        cumulative_area = np.arange(1, n + 1) / n
        y = np.log(sorted_values).astype(float)
        x = np.log(1 - cumulative_area + 1e-10).astype(float)

        px = np.cumsum(x)
        py = np.cumsum(y)
        pxx = np.cumsum(x * x)
        pxy = np.cumsum(x * y)
        pyy = np.cumsum(y * y)

        total_sx = float(px[-1])
        total_sy = float(py[-1])
        total_sxx = float(pxx[-1])
        total_sxy = float(pxy[-1])
        total_syy = float(pyy[-1])

        def _sse(seg_n: int, sx: float, sy: float, sxx: float, sxy: float, syy: float) -> float:
            n_f = float(seg_n)
            if seg_n <= 1:
                return float('inf')
            denom = n_f * sxx - sx * sx
            if abs(denom) <= 1e-20:
                return syy - (sy * sy) / n_f
            b = (n_f * sxy - sx * sy) / denom
            a = (sy - b * sx) / n_f
            return syy - 2 * a * sy - 2 * b * sxy + a * a * n_f + 2 * a * b * sx + b * b * sxx
        best_threshold_idx = 0
        min_residual = float('inf')
        start = int(n * 0.1)
        end = int(n * 0.9)
        for i in range(start, end):
            if i < 5 or (n - i) < 5:
                continue
            sx1 = float(px[i - 1])
            sy1 = float(py[i - 1])
            sxx1 = float(pxx[i - 1])
            sxy1 = float(pxy[i - 1])
            syy1 = float(pyy[i - 1])
            sx2 = total_sx - sx1
            sy2 = total_sy - sy1
            sxx2 = total_sxx - sxx1
            sxy2 = total_sxy - sxy1
            syy2 = total_syy - syy1
            total_residual = _sse(i, sx1, sy1, sxx1, sxy1, syy1) + _sse(n - i, sx2, sy2, sxx2, sxy2, syy2)
            if total_residual < min_residual:
                min_residual = total_residual
                best_threshold_idx = i
        if best_threshold_idx == 0:
            threshold = float(np.median(sorted_values) * 1.5)
        else:
            threshold = float(sorted_values[best_threshold_idx])
        threshold = max(threshold, float(sorted_values.min() + 1e-10))
        return threshold
    def _perform_ca_analysis(self, values: pd.Series) -> Dict[str, Any]:
        valid_values = values.dropna()
        valid_values = valid_values[valid_values > 0]
        if valid_values.empty:
            return {'threshold': 0.0, 'sorted_values': np.array([]), 'log_concentration': np.array([]), 'log_area': np.array([]), 'background_slope': None, 'anomaly_slope': None, 'threshold_index': 0}
        sorted_values = np.sort(valid_values)
        n = len(sorted_values)
        cumulative_area = np.arange(1, n + 1) / n
        log_concentration = np.log(sorted_values)
        log_area = np.log(1 - cumulative_area + 1e-10)
        threshold = self._calculate_c_a_threshold(values)
        threshold_idx = int(np.searchsorted(sorted_values, threshold))
        bg_coef = None
        anom_coef = None
        if 0 < threshold_idx < n:
            bg_log_area = log_area[:threshold_idx]
            bg_log_conc = log_concentration[:threshold_idx]
            if len(bg_log_area) >= 5:
                bg_coef = np.polyfit(bg_log_area, bg_log_conc, 1)
        if 0 <= threshold_idx < n:
            anom_log_area = log_area[threshold_idx:]
            anom_log_conc = log_concentration[threshold_idx:]
            if len(anom_log_area) >= 5:
                anom_coef = np.polyfit(anom_log_area, anom_log_conc, 1)
        return {'threshold': threshold, 'sorted_values': sorted_values, 'log_concentration': log_concentration, 'log_area': log_area, 'background_slope': bg_coef[0] if bg_coef is not None else None, 'anomaly_slope': anom_coef[0] if anom_coef is not None else None, 'threshold_index': threshold_idx}
    def plot_all_elements_boxplot(self, data: pd.DataFrame, element_cols: List[str]):
        self._setup_matplotlib()
        valid_elements = [element for element in element_cols if element in data.columns]
        valid_elements = sorted(valid_elements, key=lambda x: str(x).lower())
        if not valid_elements:
            logger.warning('No valid element columns are available for the boxplot')
            return
        plt.figure(figsize=(15, 8))
        boxplot_data = []
        labels = []
        for element in valid_elements:
            element_data = data[element].dropna()
            if not element_data.empty:
                boxplot_data.append(element_data)
                labels.append(element)
        box = plt.boxplot(boxplot_data, tick_labels=labels, patch_artist=True)
        cmap = plt.get_cmap('tab20')
        colors = list(getattr(cmap, 'colors', []))
        if not colors:
            denom = max(1, len(boxplot_data) - 1)
            colors = [cmap(i / denom) for i in range(len(boxplot_data))]
        colors = colors[:len(boxplot_data)]
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        plt.title(_localize_text('Boxplot of All Element Concentrations', lang=self._lang), fontsize=26)
        plt.xlabel(_localize_text('Element', lang=self._lang), fontsize=24)
        plt.ylabel(_localize_text('Concentration', lang=self._lang), fontsize=24)
        plt.xticks(rotation=45, ha='right',fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        viz_dir = os.path.join(self.output_dir, self.viz_root_dirname, 'log_log')
        os.makedirs(viz_dir, exist_ok=True)
        output_path = os.path.join(viz_dir, 'all_elements_boxplot.png')
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.exception(f'Failed to save the boxplot for all elements: {str(e)}')
        finally:
            plt.close()
    def plot_element_distributions(self, data: pd.DataFrame, element_cols: List[str], ca_results: Dict) -> List[str]:
        saved_files = []
        for element in element_cols:
            if element not in data.columns:
                continue
            element_series = data[element].dropna()
            if element_series.empty:
                logger.warning(f'{element} has no valid data. Skipping the concentration-distribution plot')
                continue
            plt.figure(figsize=(10, 6))
            sns.histplot(element_series, kde=True, color='steelblue', edgecolor='dodgerblue', alpha=0.7)
            threshold = ca_results.get(element, {}).get('threshold')
            if threshold is not None:
                threshold_name = _localize_text('Anomaly Threshold', lang=self._lang)
                plt.axvline(x=threshold, color='red', linestyle='--', label=f'{threshold_name}: {threshold:.4f}')
            plt.title(f"{element} {_localize_text('Concentration Distribution', lang=self._lang)}", fontsize=18)
            plt.xlabel(f"{element} {_localize_text('Concentration', lang=self._lang)}", fontsize=15)
            plt.ylabel(_localize_text('Frequency', lang=self._lang), fontsize=15)
            plt.legend(fontsize=15)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.tick_params(axis='both', which='minor', labelsize=12)
            plt.tight_layout()
            viz_dir = os.path.join(self.output_dir, self.viz_root_dirname, 'Concentration distribution statistics')
            os.makedirs(viz_dir, exist_ok=True)
            output_file = os.path.join(viz_dir, f'{element}_distribution.png')
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                saved_files.append(output_file)
            except Exception as e:
                logger.exception(f'Failed to save the concentration-distribution plot for {element}: {str(e)}')
            finally:
                plt.close()
        return saved_files
    def plot_ca_log_log(self, data: pd.DataFrame, element_cols: List[str], ca_results: Dict) -> List[str]:
        saved_files = []
        for element in element_cols:
            if element not in data.columns:
                continue
            element_data = data[element].dropna()
            element_data = element_data[element_data > 0]
            if element_data.empty:
                logger.warning(f'{element} has no valid positive data. Skipping the C-A log-log plot')
                continue
            sorted_data = np.sort(element_data)[::-1]
            cumulative_area = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            non_zero_mask = sorted_data > 0
            sorted_data = sorted_data[non_zero_mask]
            cumulative_area = cumulative_area[non_zero_mask]
            plt.figure(figsize=(11, 9))
            plt.loglog(
                sorted_data,
                cumulative_area,
                'o-',
                color='#1f77b4',
                markersize=6,
                markeredgewidth=0.5,
                markeredgecolor='#1f77b4',
                markerfacecolor='white',
                alpha=0.8,
                linewidth=1.5,
                label=_localize_text('Cumulative Frequency Distribution', lang=self._lang),
            )
            threshold = ca_results.get(element, {}).get('threshold')
            if threshold is not None and threshold > 0:
                threshold_area = np.sum(element_data > threshold) / len(element_data) * 100
                threshold_name = _localize_text('Anomaly Threshold', lang=self._lang)
                plt.loglog(threshold, threshold_area, '^', color='#d62728', markersize=10, markeredgewidth=1.5, markeredgecolor='#d62728', markerfacecolor='white', label=f'{threshold_name}: {threshold:.4f}')
                plt.axvline(x=threshold, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.5, label=None)
                plt.axhline(y=threshold_area, color='#d62728', linestyle='--', alpha=0.8, linewidth=1.5, label=None)
            plt.title(f"{element} {_localize_text('C-A Log-Log Plot', lang=self._lang)}", fontsize=22, pad=6)
            plt.xlabel(_localize_text('Element Concentration (Log Scale)', lang=self._lang), fontsize=20, labelpad=10)
            plt.ylabel(_localize_text('Cumulative Area Percentage (%) (Log Scale)', lang=self._lang), fontsize=20, labelpad=10)
            plt.legend(fontsize=20, loc='lower left', frameon=True, framealpha=0.9, edgecolor='black', facecolor='white')
            plt.tick_params(axis='both', which='major', labelsize=20)
            plt.tick_params(axis='both', which='minor', labelsize=20)
            plt.tight_layout()
            viz_dir = os.path.join(self.output_dir, self.viz_root_dirname, 'log_log')
            os.makedirs(viz_dir, exist_ok=True)
            output_file = os.path.join(viz_dir, f'{element}_ca_log_log.png')
            try:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                saved_files.append(output_file)
            except Exception as e:
                logger.exception(f'Failed to save the C-A log-log plot for {element}: {str(e)}')
            finally:
                plt.close()
        return saved_files
    def plot_anomaly_percentages(self, ca_results: Dict) -> str:
        if not ca_results:
            logger.warning('C-A results are empty. Skipping the anomaly-percentage comparison plot')
            return ''
        elements = list(ca_results.keys())
        percentages = [ca_results.get(elem, {}).get('anomaly_percentage', 0.0) for elem in elements]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(elements, percentages, color='skyblue')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom')
        plt.title(_localize_text('Anomalous Sample Percentage by Element', lang=self._lang))
        plt.xlabel(_localize_text('Element', lang=self._lang))
        plt.ylabel(_localize_text('Anomalous Sample Percentage (%)', lang=self._lang))
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, 'anomaly_percentages.png')
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        except Exception as e:
            logger.exception(f'Failed to save the anomaly-percentage comparison plot: {str(e)}')
            return ''
        finally:
            plt.close()
        return output_file
    def plot_anomaly_spatial_distribution(self, data: pd.DataFrame, element_cols: List[str], ca_results: Dict, coord_cols: List[str]=['Longitude', 'Latitude']) -> List[str]:
        logger.info('Generating key-element spatial-distribution maps')
        possible_x_cols = coord_cols + ['Longitude', 'longitude', 'LONGITUDE', 'Lon', 'lon', 'X', 'x']
        possible_y_cols = coord_cols + ['Latitude', 'latitude', 'LATITUDE', 'Lat', 'lat', 'Y', 'y']
        x_col = None
        y_col = None
        if len(coord_cols) >= 2:
            if coord_cols[0] in data.columns and coord_cols[1] in data.columns:
                x_col = coord_cols[0]
                y_col = coord_cols[1]
            else:
                for col in data.columns:
                    if col.strip() == coord_cols[0] and x_col is None:
                        x_col = col
                    elif col.strip() == coord_cols[1] and y_col is None:
                        y_col = col
                if x_col and y_col:
                    logger.info(f'Using whitespace-normalized requested coordinate columns: {x_col}, {y_col}')
        if x_col is None or y_col is None:
            for col in data.columns:
                if col in possible_x_cols and x_col is None:
                    x_col = col
                elif col in possible_y_cols and y_col is None:
                    y_col = col
            if x_col is None or y_col is None:
                for col in data.columns:
                    col_stripped = col.strip()
                    if col_stripped in possible_x_cols and x_col is None:
                        x_col = col
                    elif col_stripped in possible_y_cols and y_col is None:
                        y_col = col
            if x_col and y_col:
                logger.info(f'Requested coordinate columns not found. Using standard coordinate columns: {x_col}, {y_col}')
        if x_col is None or y_col is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in element_cols]
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                logger.info(f'Standard coordinate columns not found. Using numeric columns {x_col} and {y_col} as coordinates')
            else:
                logger.warning('Could not find enough coordinate or numeric columns. Skipping spatial-distribution plotting')
                return []
        try:
            try:
                from utils.data_utils import normalize_coordinates as _normalize_coordinates_fn
            except Exception:
                from .utils.data_utils import normalize_coordinates as _normalize_coordinates_fn2

                _normalize_coordinates_fn = _normalize_coordinates_fn2
            data_norm, coord_meta = _normalize_coordinates_fn(data, x_col=str(x_col), y_col=str(y_col), lon_col="Longitude", lat_col="Latitude")
            lon_arr = pd.to_numeric(data_norm["Longitude"], errors="coerce").to_numpy(dtype=float)
            lat_arr = pd.to_numeric(data_norm["Latitude"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(lon_arr) & np.isfinite(lat_arr)
            if not np.any(ok):
                raise ValueError("Longitude/Latitude columns contain no valid values")
            lon_ok = (lon_arr[ok] >= -180.0) & (lon_arr[ok] <= 180.0)
            lat_ok = (lat_arr[ok] >= -90.0) & (lat_arr[ok] <= 90.0)
            if float(np.mean(lon_ok & lat_ok)) < 0.90:
                raise ValueError("Longitude/Latitude range validation failed")
            data = data_norm
            x_col = "Longitude"
            y_col = "Latitude"
            try:
                if isinstance(coord_meta, dict) and coord_meta.get("is_projected_input"):
                    logger.info(f"Coordinates were normalized to longitude/latitude: {coord_meta}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not normalize coordinates to longitude/latitude. Skipping spatial-distribution plotting: {e}")
            return []
        saved_files = []
        for element in element_cols:
            if element not in data.columns:
                continue
            plt.figure(figsize=(12, 10))
            valid_data = data.dropna(subset=[element, x_col, y_col])
            x = valid_data[x_col].values
            y = valid_data[y_col].values
            values = valid_data[element].values
            x_plot = x
            y_plot = y
            x_label = 'Longitude'
            y_label = 'Latitude'
            grid_size = 300
            x_min, x_max = (x_plot.min(), x_plot.max())
            y_min, y_max = (y_plot.min(), y_plot.max())
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
            grid_x, grid_y = np.meshgrid(xi, yi)
            try:
                pts = np.column_stack([x_plot.astype(float), y_plot.astype(float)])
                vals = values.astype(float)
                finite_mask = np.isfinite(pts).all(axis=1) & np.isfinite(vals)
                pts = pts[finite_mask]
                vals = vals[finite_mask]
                if pts.shape[0] == 0:
                    raise ValueError("No valid sample points are available for IDW")
                tree = cKDTree(pts)
                grid_pts = np.column_stack([grid_x.ravel().astype(float), grid_y.ravel().astype(float)])
                k = int(min(12, int(pts.shape[0])))
                dists, idxs = tree.query(grid_pts, k=k, workers=-1)
                if k == 1:
                    dists = dists[:, None]
                    idxs = idxs[:, None]
                neighbor_vals = vals[idxs]
                power = 2.0
                d_safe = np.maximum(dists, 1e-12)
                weights = 1.0 / (d_safe**power)
                wsum = weights.sum(axis=1)
                z = (weights * neighbor_vals).sum(axis=1) / wsum
                exact = (dists == 0).any(axis=1)
                if np.any(exact):
                    exact_rows = np.where(exact)[0]
                    pick = np.argmax((dists[exact_rows] == 0), axis=1)
                    z[exact_rows] = neighbor_vals[exact_rows, pick]
                grid_z = z.reshape(grid_x.shape)
                vmin = vals.min()
                vmax = vals.max()
                if vmax == vmin:
                    vmin = 0
                    vmax = 1
                cmap = plt.cm.get_cmap('jet')
                ax = plt.gca()
                ax.grid(False)
                im = ax.imshow(grid_z, extent=(x_min, x_max, y_min, y_max), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', alpha=0.9, aspect='equal')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02, fraction=0.06, aspect=20)
                cbar.set_label(f"{element} Concentration", fontsize=12)
                cbar.ax.tick_params(labelsize=10)
                possible_label_cols = ['Ore', 'label', 'target', 'deposit', 'label_encoded', 'target_encoded', 'labeled', 'has_deposit', 'is_deposit']
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
                        deposit_mask = (valid_data[label_col] == 1).to_numpy()
                        if not np.any(deposit_mask):
                            try:
                                deposit_mask = (pd.to_numeric(valid_data[label_col], errors='coerce') == 1).to_numpy()
                            except Exception:
                                deposit_mask = np.zeros(shape=(len(valid_data),), dtype=bool)
                        if np.any(deposit_mask):
                            ax.scatter(
                                x_plot[deposit_mask],
                                y_plot[deposit_mask],
                                color='cyan',
                                s=30,
                                edgecolor='black',
                                linewidth=2,
                                alpha=0.8,
                                label='Known Deposits',
                            )
                    except Exception:
                        pass
                handles, labels = ax.get_legend_handles_labels()
                if handles and labels:
                    by_label = dict(zip(labels, handles))
                    ax.legend(by_label.values(), by_label.keys(), fontsize=10, loc='lower right')
                title = f"{element} Spatial Distribution"
                plt.title(title, fontsize=16)
                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.tick_params(axis='both', which='major', labelsize=12)
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                viz_dir = os.path.join(self.output_dir, self.viz_root_dirname, 'Spatial distribution statistics')
                os.makedirs(viz_dir, exist_ok=True)
                output_file = os.path.join(viz_dir, f'{element}_spatial_distribution.png')
                plt.savefig(output_file, dpi=300, format='png', bbox_inches='tight', pil_kwargs={'optimize': True, 'compress_level': 9})
                plt.close()
                saved_files.append(output_file)
            except Exception as e:
                logger.exception(f'Error while plotting the key-element spatial-distribution map for {element}: {str(e)}')
                plt.close()
        return saved_files
    def plot_ca_result_images(self, data: pd.DataFrame, element_cols: List[str], ca_results: Dict, coord_cols: List[str]=['Longitude', 'Latitude']) -> Dict[str, List[str]]:
        visualization_results: Dict[str, List[str]] = {}
        visualization_results['spatial_distributions'] = self.plot_anomaly_spatial_distribution(data, element_cols, ca_results, coord_cols)
        return visualization_results
    def run(self, data_or_state, element_cols: Optional[List[str]]=None, coord_cols: List[str]=['Longitude', 'Latitude']) -> Dict[str, List[str]]:
        try:
            data = None
            cols = element_cols
            if isinstance(data_or_state, dict):
                state = data_or_state
                data = state.get('processed_data') or state.get('preprocessed_data') or state.get('data')
                if cols is None:
                    cols = state.get('element_cols')
                if cols is None and isinstance(data, pd.DataFrame):
                    cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    coord_exclude = set(['Longitude', 'Latitude', 'X', 'Y', 'x', 'y', 'lon', 'lat'])
                    cols = [c for c in cols if c not in coord_exclude]
            else:
                data = data_or_state
            if data is None or not isinstance(data, pd.DataFrame):
                raise ValueError('No valid DataFrame data was provided for visualization')
            if not cols:
                raise ValueError('No valid element columns were provided for visualization')
            logger.info('Starting generation of key-element visualization outputs')
            return self.run_all_visualizations(data, cols, coord_cols)
        except Exception as e:
            logger.exception(f'Key-element visualization failed: {str(e)}')
            return {}
    def run_all_visualizations(self, data: pd.DataFrame, element_cols: List[str], coord_cols: List[str]=['Longitude', 'Latitude']) -> Dict[str, List[str]]:
        ca_results = {}
        for element in element_cols:
            if element in data.columns:
                ca_result = self._perform_ca_analysis(data[element])
                non_null_count = int(data[element].dropna().shape[0])
                if non_null_count > 0:
                    anomaly_count = int((data[element] > ca_result['threshold']).sum())
                    anomaly_percentage = anomaly_count / non_null_count * 100
                else:
                    anomaly_count = 0
                    anomaly_percentage = 0.0
                ca_results[element] = {**ca_result, 'anomaly_count': anomaly_count, 'anomaly_percentage': anomaly_percentage}
        visualization_results = {}
        visualization_results['distributions'] = self.plot_element_distributions(data, element_cols, ca_results)
        visualization_results['ca_log_log'] = self.plot_ca_log_log(data, element_cols, ca_results)
        anomaly_percentages_path = self.plot_anomaly_percentages(ca_results)
        visualization_results['anomaly_percentages'] = [anomaly_percentages_path] if anomaly_percentages_path else []
        visualization_results['spatial_distributions'] = self.plot_anomaly_spatial_distribution(data, element_cols, ca_results, coord_cols)
        self.plot_all_elements_boxplot(data, element_cols)
        return visualization_results
