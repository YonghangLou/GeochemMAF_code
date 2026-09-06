import os
import json
import re
import sys
import time
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional, List, TypedDict, Callable, Tuple
import pandas as pd
from langgraph.graph import StateGraph, END
try:
    from agents.geology_expert_agent import GeologyExpertAgent
    from agents.result_output_agent import ResultOutputAgent
    from agents.data_science_expert_agent import DataScienceExpertAgent
except ImportError:
    try:
        from geology_expert_agent import GeologyExpertAgent
        from result_output_agent import ResultOutputAgent
        from data_science_expert_agent import DataScienceExpertAgent
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from geology_expert_agent import GeologyExpertAgent
        from result_output_agent import ResultOutputAgent
        from data_science_expert_agent import DataScienceExpertAgent
try:
    from .utils.llm_utils import get_llm
    from .utils.data_utils import load_data
except ImportError:
    from utils.llm_utils import get_llm
    from utils.data_utils import load_data
logger = logging.getLogger(__name__)


def _input_with_log_prefix(logger_obj: logging.Logger, level: str = "INFO") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    prompt = f"{ts} - {logger_obj.name} - {level} - "
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return input()
    except Exception:
        return input(prompt)


def _safe_json_loads(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    try:
        m = re.search(r"\{[\s\S]*\}", s)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _stdin_is_interactive() -> bool:
    try:
        return bool(getattr(sys.stdin, "isatty", lambda: False)())
    except Exception:
        return False


def _is_auth_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError"}:
        return True
    msg = str(exc)
    if "401" in msg and ("Unauthorized" in msg or "Authentication" in msg):
        return True
    if "Authentication Fails" in msg or "authentication_error" in msg:
        return True
    return False


def _is_billing_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "Arrearage" in msg or "overdue-payment" in msg or ("Access denied" in msg and "account" in msg and "good standing" in msg)


def _default_task_plan_markdown(
    target_deposit: str,
    available_agents: List[str],
    *,
    geology_expert_enabled: bool = True,
) -> str:
    agents_line = ", ".join([str(a) for a in available_agents]) if available_agents else '(unknown)'
    deposit = str(target_deposit or 'Unknown target deposit')
    geology_stage = (
        f"## Stage 2: Geological Interpretation and Anomaly Identification (geology_analysis)\n"
        f"- Objective: interpret anomalies and element assemblages in relation to the mineralization processes of {deposit}\n"
        f"- Outputs: anomaly summary, hypotheses about possible mineralization controls, and target delineation recommendations\n\n"
        if geology_expert_enabled
        else
        '## Stage 2: All-Element SOM Modeling (data_science_expert)\n'
        '- Objective: perform SOM modeling and QE anomaly scoring using all valid preprocessed geochemical elements\n'
        '- Outputs: all-element model, QE anomaly scores, and evaluation metrics\n\n'
    )
    return (
        f"# Default Multi-Agent Task Plan\n\n"
        f"- Target deposit: {deposit}\n"
        f"- Available agents: {agents_line}\n\n"
        f"## Stage 1: Data Preparation And Feature Extraction (data_science_expert)\n"
        f"- Goal: complete data-quality checks, field identification, basic cleaning, and usable feature preparation\n"
        f"- Deliverables: data summary, key field list, and processed data/features for downstream analysis\n\n"
        f"{geology_stage}"
        f"## Stage 3: Aggregation And Reporting (result_output)\n"
        f"- Goal: aggregate multi-agent results into structured figures and a comprehensive report\n"
        f"- Deliverables: report files, key figures, and conclusion summaries\n\n"
        f"## Stage 4: Final Evaluation (final_decision)\n"
        f"- Goal: assess result credibility and identify the next improvement direction\n"
        f"- Deliverables: final evaluation conclusions and recommended next steps\n"
    )
def _format_duration_cn(seconds: float) -> str:
    total_seconds = int(round(float(seconds)))
    hours = total_seconds // 3600
    minutes = total_seconds % 3600 // 60
    secs = total_seconds % 60
    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    if minutes > 0:
        return f'{minutes}m {secs}s'
    return f'{secs}s'
def update_comprehensive_report_runtime(report_path: str, duration_seconds: float) -> bool:
    if not report_path or not isinstance(report_path, str):
        return False
    if not os.path.exists(report_path):
        return False
    duration_human = _format_duration_cn(duration_seconds)
    runtime_line = f'- **Total Runtime**: {duration_human} ({float(duration_seconds):.2f}s)'
    runtime_pattern = re.compile('^\\\\s*-\\\\s*\\\\*\\\\*Total Runtime\\\\*\\\\*:\\\\s*.*\\\\s*$')
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception:
        return False
    replaced = False
    new_lines: List[str] = []
    for line in lines:
        if runtime_pattern.match(line):
            new_lines.append(runtime_line)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        inserted = False
        for i, line in enumerate(new_lines):
            if '## 1. Project Overview' in line:
                insert_at = i + 1
                for j in range(insert_at, min(insert_at + 30, len(new_lines))):
                    if '**Analysis Platform**' in new_lines[j]:
                        new_lines.insert(j + 1, runtime_line)
                        inserted = True
                        break
                if not inserted:
                    new_lines.insert(insert_at, runtime_line)
                    inserted = True
                break
        if not inserted:
            new_lines.insert(2 if len(new_lines) >= 2 else len(new_lines), runtime_line)
    file_path = os.path.abspath(report_path)
    target_dir = os.path.dirname(file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.md', dir=target_dir or None, text=True)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False
class workflow_state(TypedDict):
    data: Optional[pd.DataFrame]
    target_deposit_type: Optional[str]
    study_area_location: Optional[str]
    processed_data: Optional[pd.DataFrame]
    preprocessed_data: Optional[pd.DataFrame]
    model: Optional[Any]
    predictions: Optional[Dict]
    preprocessing_results: Optional[Dict]
    prediction_results: Optional[Dict]
    ca_results: Optional[Dict]
    anomaly_analysis: Optional[Dict]
    geology_expert_results: Optional[Dict]
    feature_analysis_results: Optional[Dict]
    feature_cols: Optional[List[str]]
    model_input_metadata: Optional[Dict[str, Any]]
    output_results: Optional[Dict]
    final_results: Optional[Dict]
    element_cols: Optional[List[str]]
    task_plan: Optional[List[str]]
    current_step: Optional[str]
    next_step: Optional[str]
    current_phase: Optional[str]
    next_agent: Optional[str]
    processing_history: Optional[List[str]]
    errors: Optional[List[str]]
    analysis_results: Optional[Dict]
    agent_messages: Optional[List[str]]
    decision_history: Optional[List[Dict]]
    error: Optional[str]
    status: Optional[str]
    config: Optional[Dict]
    hitl_step_overrides: Optional[Dict[str, Any]]
    governance: Optional[Dict[str, Any]]
    artifacts: Optional[Dict[str, Any]]
    checks: Optional[List[Dict[str, Any]]]
    replan: Optional[Dict[str, Any]]
class GeoChemistryWorkflow:
    def __init__(self, llm=None, output_dir: str='./output', progress_hook: Optional[Callable[[str, str, int, str], None]] = None):
        self.llm = llm or get_llm()
        if hasattr(self.llm, 'logger') and self.llm.logger is not None:
            self.logger = self.llm.logger
        else:
            self.logger = logger
        self.logger.debug(f'Initializing GeoChemistryWorkflow - output directory: {output_dir}')
        self.logger.info(f'LLM initialized: {type(self.llm).__name__}')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.progress_hook = progress_hook
        self._progress_value = 0
        self.logger.info('Initializing agents...')
        self.data_science_expert_agent = DataScienceExpertAgent(llm=self.llm, output_dir=self.output_dir)
        self.logger.info('Data-science expert agent initialized')
        self.geology_expert_agent = GeologyExpertAgent(llm=self.llm, output_dir=self.output_dir)
        self.logger.info('Geology expert agent initialized')
        self.result_output_agent = ResultOutputAgent(llm=self.llm, output_dir=self.output_dir)
        self.logger.info('Result-output agent initialized')
        self.agents = {'data_science_expert': self.data_science_expert_agent, 'geology_analysis': self.geology_expert_agent, 'result_output': self.result_output_agent}
        self.logger.info(f'Initialized {len(self.agents)} agents')
        self.logger.info('Building workflow graph...')
        self.workflow = self._build_workflow()
        self.logger.info('Workflow graph constructed')

    @staticmethod
    def _geology_expert_enabled(state: dict) -> bool:
        cfg = state.get("config") if isinstance(state, dict) else None
        if isinstance(cfg, dict) and "geology_expert_enabled" in cfg:
            return bool(cfg.get("geology_expert_enabled"))
        return True

    def _available_agent_names(self, state: dict) -> List[str]:
        names = list(self.agents.keys())
        if not self._geology_expert_enabled(state):
            names = [name for name in names if name != "geology_analysis"]
        return names

    def _governance_enabled(self, state: dict) -> bool:
        try:
            cfg = state.get("config")
        except Exception:
            cfg = None
        if isinstance(cfg, dict):
            flag = cfg.get("governance_enabled")
            if flag is not None:
                return bool(flag)
        return True

    def _governance_defaults(self, state: dict) -> Dict[str, Any]:
        cfg = state.get("config") if isinstance(state, dict) else None
        budgets = None
        if isinstance(cfg, dict):
            budgets = cfg.get("governance_budgets")
        if not isinstance(budgets, dict):
            budgets = {}
        max_node_steps = budgets.get("max_node_steps", 0)
        max_seconds = budgets.get("max_seconds", 0)
        max_rework_total = budgets.get("max_rework_total", 0)
        max_rework_per_node = budgets.get("max_rework_per_node", 0)
        try:
            max_node_steps = int(max_node_steps)
        except Exception:
            max_node_steps = 0
        try:
            max_seconds = int(max_seconds)
        except Exception:
            max_seconds = 0
        try:
            max_rework_total = int(max_rework_total)
        except Exception:
            max_rework_total = 0
        try:
            max_rework_per_node = int(max_rework_per_node)
        except Exception:
            max_rework_per_node = 0
        if max_node_steps < 0:
            max_node_steps = 0
        if max_seconds < 0:
            max_seconds = 0
        if max_rework_total < 0:
            max_rework_total = 0
        if max_rework_per_node < 0:
            max_rework_per_node = 0
        run_id = f"{int(time.time())}-{os.getpid()}"
        return {
            "run_id": run_id,
            "started_at": datetime.now().isoformat(),
            "started_perf": float(time.perf_counter()),
            "budgets": {
                "max_node_steps": max_node_steps,
                "max_seconds": max_seconds,
                "max_rework_total": max_rework_total,
                "max_rework_per_node": max_rework_per_node,
            },
            "counters": {
                "node_steps": 0,
                "rework_total": 0,
                "rework_by_node": {"data_science_expert": 0, "geology_analysis": 0, "result_output": 0},
                "last_error_count": 0,
            },
            "mode": "auto",
        }

    def _ensure_governance_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if not self._governance_enabled(state):
            return
        gov = state.get("governance")
        if not isinstance(gov, dict):
            gov = self._governance_defaults(state)
            state["governance"] = gov
        else:
            if "budgets" not in gov or "counters" not in gov:
                base = self._governance_defaults(state)
                merged = {**base, **gov}
                if isinstance(base.get("budgets"), dict) and isinstance(gov.get("budgets"), dict):
                    merged["budgets"] = {**base["budgets"], **gov["budgets"]}
                if isinstance(base.get("counters"), dict) and isinstance(gov.get("counters"), dict):
                    merged["counters"] = {**base["counters"], **gov["counters"]}
                gov = merged
                state["governance"] = gov
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            state["artifacts"] = {}
        checks = state.get("checks")
        if not isinstance(checks, list):
            state["checks"] = []
        replan = state.get("replan")
        if not isinstance(replan, dict):
            state["replan"] = {}

    def _gov_budget_exceeded(self, state: dict) -> Tuple[bool, str]:
        gov = state.get("governance") if isinstance(state, dict) else None
        if not isinstance(gov, dict):
            return False, ""
        budgets = gov.get("budgets")
        counters = gov.get("counters")
        if not isinstance(budgets, dict) or not isinstance(counters, dict):
            return False, ""
        try:
            elapsed = float(time.perf_counter()) - float(gov.get("started_perf", time.perf_counter()))
        except Exception:
            elapsed = 0.0
        try:
            max_seconds = int(budgets.get("max_seconds", 1200))
        except Exception:
            max_seconds = 1200
        if max_seconds >= 1 and elapsed > float(max_seconds):
            return True, "budget_exceeded:max_seconds"
        try:
            max_steps = int(budgets.get("max_node_steps", 30))
        except Exception:
            max_steps = 30
        try:
            steps = int(counters.get("node_steps", 0))
        except Exception:
            steps = 0
        if max_steps >= 1 and steps > max_steps:
            return True, "budget_exceeded:max_node_steps"
        return False, ""

    def _append_check(self, state: dict, *, check_id: str, ok: bool, severity: str, details: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        checks = state.get("checks")
        if not isinstance(checks, list):
            checks = []
            state["checks"] = checks
        payload = {
            "check_id": str(check_id),
            "ok": bool(ok),
            "severity": str(severity),
            "details": details if isinstance(details, dict) else {},
            "timestamp": datetime.now().isoformat(),
        }
        checks.append(payload)

    def _register_artifacts(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        artifacts = state.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
            state["artifacts"] = artifacts
        reports_dir = os.path.join(self.output_dir, "reports")
        candidates = {
            "task_plan_md": os.path.join(reports_dir, "task_plan.md"),
            "data_analysis_report_md": os.path.join(reports_dir, "data_analysis_report.md"),
            "preprocess_strategy_md": os.path.join(reports_dir, "preprocessing_strategy.md"),
            "model_selection_md": os.path.join(reports_dir, "model_selection_and_evaluation.md"),
        }
        for k, p in candidates.items():
            try:
                if p and os.path.exists(p):
                    artifacts[k] = os.path.abspath(p)
            except Exception:
                continue
        out = state.get("output_results")
        if isinstance(out, dict):
            for k, p in out.items():
                try:
                    if isinstance(p, str) and p.strip():
                        artifacts[str(k)] = os.path.abspath(p)
                except Exception:
                    continue

    def _verify_node(self, node_name: str, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if node_name == "initialize":
            task_plan = state.get("task_plan")
            ok = isinstance(task_plan, list) and bool([x for x in task_plan if isinstance(x, str) and x.strip()])
            self._append_check(
                state,
                check_id="initialize.task_plan",
                ok=ok,
                severity="error",
                details={"task_plan_present": bool(ok)},
            )
            return
        if node_name == "data_science_expert":
            processed = state.get("processed_data") or state.get("preprocessed_data")
            ok_df = isinstance(processed, pd.DataFrame) and (not processed.empty)
            self._append_check(
                state,
                check_id="data.preprocessed",
                ok=ok_df,
                severity="error",
                details={"processed_df": bool(ok_df), "shape": getattr(processed, "shape", None)},
            )
            geo = None
            try:
                ar = state.get("analysis_results")
                if isinstance(ar, dict):
                    geo = ar.get("geology")
            except Exception:
                geo = None
            has_geo = isinstance(geo, dict) and bool(geo)
            pred = state.get("prediction_results")
            has_pred = isinstance(pred, dict) and bool(pred)
            ok_pred = (not has_geo) or has_pred
            self._append_check(
                state,
                check_id="model.prediction_ready",
                ok=ok_pred,
                severity="error",
                details={"geology_present": bool(has_geo), "prediction_results_present": bool(has_pred)},
            )
            return
        if node_name == "geology_analysis":
            ar = state.get("analysis_results")
            geo = ar.get("geology") if isinstance(ar, dict) else None
            ok_geo = isinstance(geo, dict) and bool(geo)
            self._append_check(
                state,
                check_id="geology.results_present",
                ok=ok_geo,
                severity="error",
                details={"geology_present": bool(ok_geo), "keys": list(geo.keys())[:30] if isinstance(geo, dict) else []},
            )
            feat = state.get("feature_analysis_results")
            ok_feat = isinstance(feat, dict) and bool(feat)
            self._append_check(
                state,
                check_id="geology.feature_context_present",
                ok=ok_feat,
                severity="warn",
                details={"feature_analysis_present": bool(ok_feat), "keys": list(feat.keys())[:30] if isinstance(feat, dict) else []},
            )
            return
        if node_name == "result_output":
            out = state.get("output_results")
            ok_out = isinstance(out, dict) and bool(out)
            self._append_check(
                state,
                check_id="output.results_present",
                ok=ok_out,
                severity="error",
                details={"output_results_present": bool(ok_out), "keys": list(out.keys())[:30] if isinstance(out, dict) else []},
            )
            return

    def _rule_based_replan(self, node_name: str, state: dict) -> Optional[Dict[str, Any]]:
        if not isinstance(state, dict):
            return None
        gov = state.get("governance")
        budgets = gov.get("budgets") if isinstance(gov, dict) else None
        counters = gov.get("counters") if isinstance(gov, dict) else None
        if isinstance(budgets, dict) and isinstance(counters, dict):
            try:
                max_total = int(budgets.get("max_rework_total", 0))
            except Exception:
                max_total = 0
            try:
                total = int(counters.get("rework_total", 0))
            except Exception:
                total = 0
            if max_total > 0 and total > max_total:
                return {
                    "status": "hitl" if self._hitl_enabled(state) else "abort",
                    "rework_node": None,
                    "reason_codes": ["budget_exceeded:max_rework_total"],
                    "message_to_user": 'The rework budget has been exceeded. Human intervention or conservative termination is recommended.',
                    "step_overrides": {},
                    "config_overrides": {},
                }
        if node_name == "data_science_expert":
            processed = state.get("processed_data") or state.get("preprocessed_data")
            ok_df = isinstance(processed, pd.DataFrame) and (not processed.empty)
            if not ok_df:
                return {
                    "status": "needs_rework",
                    "rework_node": "data_science_expert",
                    "reason_codes": ["missing_artifact:processed_data"],
                    "message_to_user": 'No valid preprocessed data was produced. Rework the data-preprocessing step.',
                    "step_overrides": {},
                    "config_overrides": {},
                }
            ar = state.get("analysis_results")
            geo = ar.get("geology") if isinstance(ar, dict) else None
            has_geo = isinstance(geo, dict) and bool(geo)
            pred = state.get("prediction_results")
            has_pred = isinstance(pred, dict) and bool(pred)
            if has_geo and not has_pred:
                return {
                    "status": "needs_rework",
                    "rework_node": "data_science_expert",
                    "reason_codes": ["missing_artifact:prediction_results"],
                    "message_to_user": 'Geological analysis results exist, but no prediction results were produced. Rework the modeling/prediction step.',
                    "step_overrides": {},
                    "config_overrides": {},
                }
        if node_name == "geology_analysis":
            ar = state.get("analysis_results")
            geo = ar.get("geology") if isinstance(ar, dict) else None
            if not (isinstance(geo, dict) and bool(geo)):
                feat = state.get("feature_analysis_results")
                if not (isinstance(feat, dict) and bool(feat)):
                    return {
                        "status": "needs_rework",
                        "rework_node": "data_science_expert",
                        "reason_codes": ["missing_context:feature_analysis_results"],
                        "message_to_user": 'Feature-analysis context is missing. Rework the data-science step first to generate feature-analysis results.',
                        "step_overrides": {},
                        "config_overrides": {},
                    }
                return {
                    "status": "needs_rework",
                    "rework_node": "geology_analysis",
                    "reason_codes": ["missing_artifact:geology_results"],
                    "message_to_user": 'No geological analysis results were produced. Rework the geological-interpretation step.',
                    "step_overrides": {},
                    "config_overrides": {},
                }
        if node_name == "result_output":
            out = state.get("output_results")
            ok_out = isinstance(out, dict) and bool(out)
            if not ok_out:
                processed = state.get("processed_data") or state.get("preprocessed_data")
                ok_df = isinstance(processed, pd.DataFrame) and (not processed.empty)
                if not ok_df:
                    return {
                        "status": "needs_rework",
                        "rework_node": "data_science_expert",
                        "reason_codes": ["missing_artifact:processed_data"],
                        "message_to_user": 'The output stage is missing processed data. Rework preprocessing/modeling first.',
                        "step_overrides": {},
                        "config_overrides": {},
                    }
                return {
                    "status": "needs_rework",
                    "rework_node": "result_output",
                    "reason_codes": ["missing_artifact:output_results"],
                    "message_to_user": 'No valid output artifacts were produced. Rework the result-output step.',
                    "step_overrides": {},
                    "config_overrides": {},
                }
        return None

    def _llm_governor_replan(self, node_name: str, state: dict) -> Dict[str, Any]:
        default_payload: Dict[str, Any] = {
            "status": "ok",
            "rework_node": None,
            "reason_codes": [],
            "message_to_user": "",
            "step_overrides": {},
            "config_overrides": {},
        }
        if not isinstance(state, dict):
            return default_payload
        checks = state.get("checks")
        checks_tail = checks[-10:] if isinstance(checks, list) else []
        output_keys = []
        out = state.get("output_results")
        if isinstance(out, dict):
            output_keys = list(out.keys())[:30]
        ar = state.get("analysis_results")
        analysis_keys = list(ar.keys())[:30] if isinstance(ar, dict) else []
        gov = state.get("governance") if isinstance(state, dict) else None
        counters = gov.get("counters") if isinstance(gov, dict) else None
        budgets = gov.get("budgets") if isinstance(gov, dict) else None
        ctx = {
            "node": node_name,
            "phase": state.get("current_phase"),
            "next_agent": state.get("next_agent"),
            "errors_count": len(state.get("errors") or []),
            "analysis_keys": analysis_keys,
            "output_keys": output_keys,
            "checks_tail": checks_tail,
            "rework_total": counters.get("rework_total") if isinstance(counters, dict) else None,
            "rework_by_node": counters.get("rework_by_node") if isinstance(counters, dict) else None,
            "budgets": budgets if isinstance(budgets, dict) else None,
        }
        allowed_nodes = ["data_science_expert", "result_output"]
        if self._geology_expert_enabled(state):
            allowed_nodes.insert(1, "geology_analysis")
        prompt = (
            'You are the governor of a multi-agent geochemical workflow. '
            'Your task is to judge whether the latest node outcome requires rework, human intervention, or termination.'
            "\n\n"
            'You must output JSON only. Do not output any other text and do not use Markdown code fences.'
            "\n\n"
            'Hard constraints:\n'
            '1) status must be one of: ok | needs_rework | hitl | abort.\n'
            f"2) When status=needs_rework, rework_node must be one of: {allowed_nodes}.\n"
            '3) Do not invent files, metrics, or steps that do not exist. If information is insufficient, make a conservative judgment.\n'
            '4) step_overrides may only target the selected rework_node and must be an object; use {} when there is no executable change.\n'
            '5) config_overrides may only contain: structured_output_enabled, reflection_enabled, reflection_max_rounds.\n'
            "\n\n"
            f"Current context: {ctx}\n\n"
            'Return JSON with the following fields:\n'
            "- status: string\n"
            "- rework_node: string|null\n"
            "- reason_codes: string[]\n"
            "- message_to_user: string\n"
            "- step_overrides: object\n"
            "- config_overrides: object\n"
        )
        try:
            governor_agent = self.geology_expert_agent if self._geology_expert_enabled(state) else self.data_science_expert_agent
            parsed = governor_agent.decide_json(prompt, default_payload, config=state.get("config"))
            return parsed if isinstance(parsed, dict) else default_payload
        except Exception:
            return default_payload

    def _apply_replan(self, node_name: str, state: dict, replan: Dict[str, Any]) -> None:
        if not isinstance(state, dict) or not isinstance(replan, dict):
            return
        state["replan"] = dict(replan)
        status = str(replan.get("status") or "").strip().lower()
        rework_node = replan.get("rework_node")
        if rework_node is not None:
            rework_node = str(rework_node).strip()
        reason_codes_obj = replan.get("reason_codes")
        reason_codes = [str(x) for x in reason_codes_obj] if isinstance(reason_codes_obj, list) else []
        message = str(replan.get("message_to_user") or "").strip()
        if message:
            state.setdefault("processing_history", []).append(f"Governor: {message}")
        cfg_over = replan.get("config_overrides")
        applied_cfg = self._hitl_apply_config_overrides(state, cfg_over)
        step_over = replan.get("step_overrides")
        applied_step = {}
        if status == "needs_rework" and rework_node in {"data_science_expert", "geology_analysis", "result_output"}:
            applied_step = self._hitl_apply_step_overrides(state, node_name=rework_node, overrides=step_over)
        gov = state.get("governance")
        rework_allowed = True
        if isinstance(gov, dict):
            counters = gov.get("counters")
            budgets = gov.get("budgets")
            if isinstance(counters, dict) and status == "needs_rework" and rework_node:
                counters["rework_total"] = int(counters.get("rework_total", 0) or 0) + 1
                rbn = counters.get("rework_by_node")
                if not isinstance(rbn, dict):
                    rbn = {}
                    counters["rework_by_node"] = rbn
                rbn[rework_node] = int(rbn.get(rework_node, 0) or 0) + 1
                try:
                    max_per = int(budgets.get("max_rework_per_node", 0)) if isinstance(budgets, dict) else 0
                except Exception:
                    max_per = 0
                if max_per > 0 and int(rbn.get(rework_node, 0) or 0) > max_per:
                    rework_allowed = False
                    reason_codes.append("budget_exceeded:max_rework_per_node")
        if status == "needs_rework" and rework_node in {"data_science_expert", "geology_analysis", "result_output"}:
            if rework_allowed:
                state["next_agent"] = rework_node
            else:
                state["next_agent"] = "final_decision"
                state["current_phase"] = "budget_exceeded"
        elif status == "abort":
            state["next_agent"] = "final_decision"
            state["current_phase"] = "abort"
        elif status == "hitl":
            if isinstance(state.get("config"), dict):
                state["config"]["interaction_mode"] = "hitl"
        state.setdefault("decision_history", []).append(
            {
                "raw_decision": f"governor:{node_name}",
                "mapped_decision": state.get("next_agent"),
                "context": {
                    "status": status,
                    "rework_node": rework_node,
                    "reason_codes": reason_codes,
                    "applied_config_overrides": applied_cfg,
                    "applied_step_overrides": applied_step,
                },
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

    def _governance_post(self, node_name: str, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if not self._governance_enabled(state):
            return
        self._ensure_governance_state(state)
        self._register_artifacts(state)
        self._verify_node(node_name, state)
        if node_name in {"initialize"}:
            return
        exceeded, why = self._gov_budget_exceeded(state)
        if exceeded:
            replan: Dict[str, Any] = {
                "status": "hitl" if self._hitl_enabled(state) else "abort",
                "rework_node": None,
                "reason_codes": [why],
                "message_to_user": 'The budget limit has been reached. The workflow will terminate or switch to HITL mode.',
                "step_overrides": {},
                "config_overrides": {},
            }
            self._apply_replan(node_name, state, replan)
            return
        rule = self._rule_based_replan(node_name, state)
        if isinstance(rule, dict):
            self._apply_replan(node_name, state, rule)
            return
        llm_replan = self._llm_governor_replan(node_name, state)
        if isinstance(llm_replan, dict):
            self._apply_replan(node_name, state, llm_replan)

    def _governance_pre(self, node_name: str, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if not self._governance_enabled(state):
            return
        self._ensure_governance_state(state)
        gov = state.get("governance")
        if isinstance(gov, dict):
            counters = gov.get("counters")
            if isinstance(counters, dict):
                counters["node_steps"] = int(counters.get("node_steps", 0) or 0) + 1
                try:
                    counters["last_error_count"] = int(len(state.get("errors") or []))
                except Exception:
                    counters["last_error_count"] = int(counters.get("last_error_count", 0) or 0)
        exceeded, why = self._gov_budget_exceeded(state)
        if exceeded:
            can_finish_output = (
                node_name == "result_output"
                and isinstance(state.get("prediction_results"), dict)
                and bool(state.get("prediction_results"))
                and not (isinstance(state.get("output_results"), dict) and bool(state.get("output_results")))
            )
            if can_finish_output:
                state.setdefault("processing_history", []).append('Governor: budget exceeded, but final result-output cleanup is still allowed')
                return
            state["next_agent"] = "final_decision"
            state["current_phase"] = "budget_exceeded"
            state.setdefault("errors", []).append(f"Governor: {why}")
            state.setdefault("processing_history", []).append('Governor: budget exceeded, terminating downstream steps')

    def _hitl_enabled(self, state: dict) -> bool:
        cfg = state.get("config") if isinstance(state, dict) else None
        if not isinstance(cfg, dict):
            return False
        mode = cfg.get("interaction_mode")
        if isinstance(mode, str):
            return mode.strip().lower() in {"hitl", "human", "human_in_the_loop"}
        return bool(cfg.get("hitl_enabled", False))

    def _hitl_interpret(
        self,
        llm: Any,
        *,
        user_text: str,
        context: dict,
        suggested_next_agent: str,
        valid_decisions: List[str],
        no_clarification: bool = False,
    ) -> Dict[str, Any]:
        force_clause = ""
        if no_clarification:
            force_clause = 'Important constraint: you must not ask for clarification again. If information is insufficient, set need_clarification to false, proposed_next_agent to null, and explain the reason in assistant_reply.\n\n'
        prompt = (
            'You are the HITL interpreter for a multi-agent system. '
            'The user may ask questions or issue natural-language instructions. Your task is to understand the intent and map it to the next workflow action.'
            "\n\n"
            'You must output JSON only. Do not output any other text and do not use Markdown code fences.'
            "\n\n"
            'When assistant_reply answers a user question, it must stay grounded in the current multi-agent context: available agents, current phase, completed and pending tasks, and the system-recommended next step.'
            "\n\n"
            f"{force_clause}"
            f"System-recommended next step: {suggested_next_agent}\n"
            f"Allowed next-step options: {valid_decisions}\n\n"
            f"Current context: {context}\n\n"
            f"User input: {user_text}\n\n"
            'Return JSON with the following fields:\n'
            '- intent_summary: string; restate your understanding of the user intent in 1-3 sentences\n'
            '- proposed_next_agent: string|null, choose from the allowed options; use null if the user is only asking a question without changing the flow\n'
            '- assistant_reply: string, optional; answer the user if explanation is needed\n'
            '- need_clarification: boolean; whether a critical point must be clarified before execution\n'
            '- clarifying_question: string, optional; the single most important question when need_clarification=true\n'
            '- config_overrides: object, optional; only these keys are allowed: structured_output_enabled, reflection_enabled, reflection_max_rounds\n'
        )
        raw = llm.invoke(prompt).content
        parsed = _safe_json_loads(raw)
        if parsed is None:
            return {
                "intent_summary": 'The model output could not be parsed as structured intent. The workflow will continue with the default system suggestion.',
                "proposed_next_agent": None,
                "assistant_reply": "",
                "need_clarification": False,
                "clarifying_question": "",
                "config_overrides": {},
                "raw": str(raw),
            }
        parsed["raw"] = str(raw)
        return parsed

    def _hitl_interpret_task_plan(
        self,
        llm: Any,
        *,
        user_text: str,
        system_capabilities: str,
        current_task_plan: str,
        available_agents: List[str],
        data_summary: Optional[str] = None,
        no_clarification: bool = False,
    ) -> Dict[str, Any]:
        force_clause = ""
        if no_clarification:
            force_clause = 'Important constraint: you must not ask for clarification again. If information is insufficient, set need_clarification to false, updated_task_plan_markdown to null, and explain the reason in assistant_reply.\n\n'
        prompt = (
            'You are the HITL interpreter for a multi-agent system. '
            'The user may ask questions about the global task plan or request modifications. Your task is to understand the intent and help revise the task plan without introducing new agents.'
            "\n\n"
            'You must output JSON only. Do not output any other text and do not use Markdown code fences.'
            "\n\n"
            'Hard constraints:\n'
            '1) You may only use agent names that already exist in the current system. Do not invent new agents, tools, or external systems.\n'
            '2) Limit your response to the task plan itself. Do not ask for unrelated information.\n'
            '3) Data has already been loaded by the system. Do not ask the user for a data path or data file.\n'
            '4) If clarification is absolutely necessary, ask exactly one most important question.\n'
            "\n\n"
            f"{force_clause}"
            f"Available agents: {available_agents}\n\n"
            f"Data summary: {data_summary or '(loaded; no extra summary)'}\n\n"
            f"System capability summary:\n{system_capabilities}\n\n"
            f"Current task plan (Markdown):\n{current_task_plan}\n\n"
            f"User input: {user_text}\n\n"
            'Return JSON with the following fields:\n'
            '- intent_summary: string; restate your understanding of the user intent in 1-3 sentences\n'
            '- assistant_reply: string, optional; answer the user if explanation is needed, based on the existing multi-agent system and current plan\n'
            '- updated_task_plan_markdown: string|null; if the user requests an executable modification, return the full replacement Markdown plan, otherwise null\n'
            '- need_clarification: boolean, whether one key clarification is required before modifying the plan\n'
            '- clarifying_question: string, optional; the single most important question when need_clarification=true\n'
            '- config_overrides: object, optional; only these keys are allowed: structured_output_enabled, reflection_enabled, reflection_max_rounds\n'
        )
        raw = llm.invoke(prompt).content
        parsed = _safe_json_loads(raw)
        if parsed is None:
            return {
                "intent_summary": 'The model output could not be parsed as a task-plan revision request. The current plan will remain unchanged.',
                "assistant_reply": "",
                "updated_task_plan_markdown": None,
                "need_clarification": False,
                "clarifying_question": "",
                "config_overrides": {},
                "raw": str(raw),
            }
        parsed["raw"] = str(raw)
        return parsed

    def _maybe_hitl_confirm_task_plan(
        self,
        *,
        state: dict,
        llm: Any,
        system_capabilities: str,
        task_plan: str,
        available_agents: List[str],
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        if not self._hitl_enabled(state):
            return task_plan, None
        if not _stdin_is_interactive():
            return task_plan, None
        cfg = state.get("config") if isinstance(state, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        max_rounds_raw = cfg.get("hitl_max_rounds", 5)
        max_clarifications_raw = cfg.get("hitl_max_clarifications", 5)
        try:
            max_rounds = int(max_rounds_raw)
        except Exception:
            max_rounds = 5
        try:
            max_clarifications = int(max_clarifications_raw)
        except Exception:
            max_clarifications = 5
        if max_rounds < 1:
            max_rounds = 1
        if max_rounds > 10:
            max_rounds = 10
        if max_clarifications < 0:
            max_clarifications = 0
        if max_clarifications > 10:
            max_clarifications = 10
        current_plan = str(task_plan)
        self.logger.info('HITL mode: a global task plan has been generated (use show to inspect it)')
        data_summary = None
        data_obj = state.get("data") if isinstance(state, dict) else None
        try:
            if isinstance(data_obj, pd.DataFrame):
                cols = [str(c) for c in list(data_obj.columns)[:15]]
                data_summary = f"shape={data_obj.shape}, columns(head15)={cols}"
        except Exception:
            data_summary = None
        clarifications = 0
        user_text = ""
        for _ in range(max_rounds):
            if not user_text:
                self.logger.info('[INPUT] Confirm task plan: Enter=accept and continue; show=view plan; exit=leave HITL and continue; or type natural-language revisions/questions')
                user_text = _input_with_log_prefix(self.logger).strip()
                if not user_text:
                    return current_plan, {"mode": "accepted", "clarifications": clarifications}
                user_norm = user_text.strip().lower()
                if user_norm in {'start', 'start', 'confirm', 'confirm start', 'continue', "continue", "run", "start", "ok", 'okay', 'yes', 'yes', "y", "yes", "1"}:
                    return current_plan, {"mode": "accepted", "clarifications": clarifications, "user_text": user_text}
                if user_norm in {"show", "s"}:
                    self.logger.info(f"Current task plan:\n{current_plan}")
                    user_text = ""
                    continue
                if user_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return current_plan, {"mode": "disabled_by_user", "clarifications": clarifications}
            parsed = self._hitl_interpret_task_plan(
                llm,
                user_text=user_text,
                system_capabilities=system_capabilities,
                current_task_plan=current_plan,
                available_agents=available_agents,
                data_summary=data_summary,
            )
            need_clarification = bool(parsed.get("need_clarification", False))
            clarifying_question = str(parsed.get("clarifying_question") or "").strip()
            if need_clarification and clarifying_question:
                if clarifications >= max_clarifications:
                    self.logger.info('The task-plan clarification limit has been reached. Continue with the current plan without further questions.')
                    return current_plan, {
                        "mode": "clarification_limit_reached",
                        "user_text": user_text,
                        "parsed": parsed,
                        "clarifications": clarifications,
                        "max_clarifications": max_clarifications,
                    }
                self.logger.info(f"[INPUT] Task-plan clarification required: {clarifying_question} (exit=leave HITL)")
                answer = _input_with_log_prefix(self.logger).strip()
                answer_norm = answer.strip().lower()
                if answer_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return current_plan, {"mode": "disabled_by_user_on_clarification", "user_text": user_text, "parsed": parsed}
                if not answer:
                    return current_plan, {"mode": "cancelled_empty_clarification", "user_text": user_text, "parsed": parsed}
                clarifications += 1
                user_text = f"{user_text}\nAdditional information: {answer}"
                continue
            assistant_reply = str(parsed.get("assistant_reply") or "").strip()
            if assistant_reply:
                self.logger.info(f"HITL reply: {assistant_reply}")
            intent_summary = str(parsed.get("intent_summary") or "").strip()
            updated_plan_obj = parsed.get("updated_task_plan_markdown", None)
            updated_plan = None
            if isinstance(updated_plan_obj, str) and updated_plan_obj.strip():
                updated_plan = updated_plan_obj.strip()
            applied_overrides = self._hitl_apply_config_overrides(state, parsed.get("config_overrides"))
            if intent_summary:
                self.logger.info(f"HITL interpretation: {intent_summary}")
            if applied_overrides:
                self.logger.info(f"HITL config overrides: {applied_overrides}")
            if updated_plan:
                current_plan = updated_plan
                self.logger.info('HITL generated an updated task plan (use show to inspect it)')
            self.logger.info('[INPUT] Confirm: Enter/y/yes/1=accept the current plan and continue; exit=leave HITL and continue; otherwise enter further revision instructions')
            confirm = _input_with_log_prefix(self.logger).strip()
            confirm_norm = confirm.strip().lower()
            if not confirm_norm or confirm_norm in {"y", "yes", "1", 'yes', 'yes', "ok", 'confirm', 'confirm start', 'start', 'continue', "continue", "run", "start"}:
                return current_plan, {
                    "mode": "confirmed",
                    "user_text": user_text,
                    "parsed": parsed,
                    "applied_config_overrides": applied_overrides,
                    "clarifications": clarifications,
                }
            if confirm_norm in {"exit", "quit", "q"}:
                if isinstance(cfg, dict):
                    cfg["interaction_mode"] = "auto"
                return current_plan, {
                    "mode": "disabled_by_user",
                    "user_text": user_text,
                    "parsed": parsed,
                    "applied_config_overrides": applied_overrides,
                    "clarifications": clarifications,
                }
            user_text = confirm
        parsed_last = self._hitl_interpret_task_plan(
            llm,
            user_text=user_text,
            system_capabilities=system_capabilities,
            current_task_plan=current_plan,
            available_agents=available_agents,
            data_summary=data_summary,
            no_clarification=True,
        )
        return current_plan, {"mode": "max_rounds_reached", "user_text": user_text, "parsed": parsed_last, "clarifications": clarifications}

    def _hitl_apply_config_overrides(self, state: dict, overrides: Any) -> Dict[str, Any]:
        if not isinstance(overrides, dict):
            return {}
        cfg = state.get("config") if isinstance(state, dict) else None
        if not isinstance(cfg, dict):
            return {}
        allowed = {
            "structured_output_enabled",
            "reflection_enabled",
            "reflection_max_rounds",
        }
        applied: Dict[str, Any] = {}
        for k, v in overrides.items():
            key = str(k)
            if key not in allowed:
                continue
            cfg[key] = v
            applied[key] = v
        return applied

    def _hitl_allowed_step_overrides(self, node_name: str) -> Dict[str, Any]:
        if node_name == "data_science_expert":
            return {
                "skip_post_preprocess_feature_analysis": 'bool, whether to skip the post-preprocessing feature-analysis substep',
                "post_feature_analysis_parts": {
                    "correlation": 'bool, whether to run correlation analysis',
                    "hierarchical": 'bool, whether to run hierarchical clustering',
                    "factor": 'bool, whether to run factor analysis',
                },
                "write_preprocessed_report": 'bool, whether to write the post-preprocessing analysis report',
                "model_key": 'str, force a specific model (SOM)',
                "auto_programming_enabled": 'bool, whether to enable auto-programming',
                "programming_request": 'str, code task the user wants to auto-generate and execute',
                "programming_apply_mode": 'str, replace_df=replace current data, attach_artifact=attach artifact only',
            }
        if node_name == "geology_analysis":
            return {
                "skip_visualizations": 'bool, whether to skip key-element visualizations',
                "key_elements_limit": 'int, maximum number of key-element visualizations (1-30)',
            }
        return {}

    def _hitl_sanitize_step_overrides(self, node_name: str, overrides: Any) -> Dict[str, Any]:
        if not isinstance(overrides, dict):
            return {}
        allowed = self._hitl_allowed_step_overrides(node_name)
        if not allowed:
            return {}
        out: Dict[str, Any] = {}
        for k, v in overrides.items():
            key = str(k)
            if key not in allowed:
                continue
            if key == "post_feature_analysis_parts":
                if not isinstance(v, dict):
                    continue
                parts: Dict[str, bool] = {}
                for part_key in ("correlation", "hierarchical", "factor"):
                    if part_key in v:
                        parts[part_key] = bool(v.get(part_key))
                if parts:
                    out[key] = parts
                continue
            if key in {"skip_post_preprocess_feature_analysis", "write_preprocessed_report", "skip_visualizations"}:
                out[key] = bool(v)
                continue
            if key == "auto_programming_enabled":
                out[key] = bool(v)
                continue
            if key == "model_key":
                raw = str(v).strip().lower()
                if raw in {"som", "self organizing map", "self-organizing map", "selforganizingmap", '\u81ea\u7ec4\u7ec7\u6620\u5c04', '\u81ea\u7ec4\u7ec7\u6620\u5c04\u795e\u7ecf\u7f51\u7edc'}:
                    out[key] = "som"
                continue
            if key == "programming_request":
                raw = str(v).strip()
                if raw:
                    out[key] = raw[:2000]
                continue
            if key == "programming_apply_mode":
                raw = str(v).strip().lower()
                if raw in {"replace_df", "attach_artifact"}:
                    out[key] = raw
                continue
            if key == "key_elements_limit":
                try:
                    n = int(v)
                except Exception:
                    continue
                if n < 1:
                    n = 1
                if n > 30:
                    n = 30
                out[key] = n
                continue
        return out

    def _hitl_apply_step_overrides(self, state: dict, *, node_name: str, overrides: Any) -> Dict[str, Any]:
        sanitized = self._hitl_sanitize_step_overrides(node_name, overrides)
        if not sanitized:
            return {}
        if not isinstance(state, dict):
            return {}
        container = state.get("hitl_step_overrides")
        if not isinstance(container, dict):
            container = {}
            state["hitl_step_overrides"] = container
        existing = container.get(node_name)
        if not isinstance(existing, dict):
            existing = {}
        merged = {**existing, **sanitized}
        container[node_name] = merged
        return sanitized

    def _hitl_interpret_step(
        self,
        llm: Any,
        *,
        user_text: str,
        node_name: str,
        label: str,
        preview: str,
        allowed_overrides: Dict[str, Any],
        no_clarification: bool = False,
    ) -> Dict[str, Any]:
        force_clause = ""
        if no_clarification:
            force_clause = 'Important constraint: do not ask for further clarification. need_clarification must be false and step_overrides must be {}.\n\n'
        prompt = (
            ' The user may ask questions about, or request modifications to, the current step about to run.'
            " Your tasks are: 1) answer the user's question if needed; "
            '2) map requested method/content changes for the current step into executable structured instructions.'
            '1) Answer any user questions; 2) map requested changes to the current step methods or content into executable structured instructions. '
            "\n\n"
            'You must not change the overall task objective, request new data sources, or modify the global task plan. '
            'You may only emit changes allowed by the step_overrides whitelist.'
            "\n\n"
            'You must output JSON only. Do not output any other text and do not use Markdown code fences.'
            "\n\n"
            f"{force_clause}"
            f"Current step: {label} ({node_name})\n"
            f"Step preview:\n{preview}\n\n"
            f"Allowed step_overrides whitelist (keys and meanings): {allowed_overrides}\n\n"
            f"User input: {user_text}\n\n"
            'Return JSON with the following fields:\n'
            '- intent_summary: string; restate your understanding of the user intent in 1-3 sentences\n'
            '- assistant_reply: string, optional; answer here if the user asked a question or needs an explanation\n'
            '- action: string, must be one of: continue | skip | exit_hitl | modify_then_continue | modify_then_skip\n'
            '- step_overrides: object, must only contain whitelisted keys; use {} if no executable change exists\n'
            '- config_overrides: object, optional; may only modify system settings such as temperature, structured output, or reflection\n'
            '- need_clarification: boolean; whether a critical point must be clarified before execution\n'
            '- clarifying_question: string, if need_clarification=true, provide exactly one question\n'
            '- If the user asks the current step to auto-write code for a data-processing task, prefer auto_programming_enabled/programming_request/programming_apply_mode\n'
        )
        raw = llm.invoke(prompt).content
        parsed = _safe_json_loads(raw)
        if parsed is None:
            return {
                "intent_summary": 'Could not parse the model output. Ignore this modification and continue.',
                "assistant_reply": "",
                "action": "continue",
                "step_overrides": {},
                "config_overrides": {},
                "need_clarification": False,
                "clarifying_question": "",
                "raw": str(raw),
            }
        parsed["raw"] = str(raw)
        return parsed

    def _maybe_hitl_dialog_for_node(self, *, state: dict, node_name: str, label: str, preview: str) -> tuple[str, Optional[Dict[str, Any]]]:
        if not self._hitl_enabled(state):
            return "continue", None
        if not _stdin_is_interactive():
            return "continue", None
        cfg = state.get("config") if isinstance(state, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        max_rounds_raw = cfg.get("hitl_max_rounds", 5)
        max_clarifications_raw = cfg.get("hitl_max_clarifications", 5)
        try:
            max_rounds = int(max_rounds_raw)
        except Exception:
            max_rounds = 5
        try:
            max_clarifications = int(max_clarifications_raw)
        except Exception:
            max_clarifications = 5
        if max_rounds < 1:
            max_rounds = 1
        if max_rounds > 10:
            max_rounds = 10
        if max_clarifications < 0:
            max_clarifications = 0
        if max_clarifications > 10:
            max_clarifications = 10
        allowed_overrides = self._hitl_allowed_step_overrides(node_name)
        clarifications = 0
        user_text = ""
        for _ in range(max_rounds):
            if not user_text:
                self.logger.info(
                    f"[INPUT] HITL mode: this step is editable (Enter=execute; skip=skip this step; exit=leave HITL and continue; or type natural-language changes to this step's method/content)\n{preview}"
                )
                user_text = _input_with_log_prefix(self.logger).strip()
                if not user_text:
                    return "continue", {"mode": "accepted_default"}
                user_norm = user_text.strip().lower()
                if user_norm in {"skip", 'skip'}:
                    return "skip", {"mode": "skipped_by_user"}
                if user_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return "continue", {"mode": "disabled_by_user"}
            if node_name == "data_science_expert":
                user_norm2 = user_text.strip().lower()
                forced_model = None
                if "som" in user_norm2 or "self organizing map" in user_norm2 or "self-organizing map" in user_norm2 or 'self-organizing map' in user_norm2:
                    forced_model = "som"
                if forced_model:
                    applied_step = self._hitl_apply_step_overrides(state, node_name=node_name, overrides={"model_key": forced_model})
                    if applied_step:
                        self.logger.info(f"HITL step overrides: {applied_step}")
                    return "continue", {"mode": "forced_model", "model_key": forced_model, "user_text": user_text}
            parsed = self._hitl_interpret_step(
                self.llm,
                user_text=user_text,
                node_name=node_name,
                label=label,
                preview=preview,
                allowed_overrides=allowed_overrides,
            )
            need_clarification = bool(parsed.get("need_clarification", False))
            clarifying_question = str(parsed.get("clarifying_question") or "").strip()
            if need_clarification and clarifying_question:
                if clarifications >= max_clarifications:
                    self.logger.info('The HITL clarification limit has been reached. Continue with the current settings.')
                    return "continue", {"mode": "clarification_limit_reached", "user_text": user_text, "parsed": parsed}
                self.logger.info(f"[INPUT] HITL clarification required: {clarifying_question} (skip=skip; exit=leave HITL)")
                answer = _input_with_log_prefix(self.logger).strip()
                answer_norm = answer.strip().lower()
                if answer_norm in {"skip", 'skip'}:
                    return "skip", {"mode": "skipped_on_clarification", "user_text": user_text, "parsed": parsed}
                if answer_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return "continue", {"mode": "disabled_by_user_on_clarification", "user_text": user_text, "parsed": parsed}
                if not answer:
                    return "continue", {"mode": "cancelled_empty_clarification", "user_text": user_text, "parsed": parsed}
                clarifications += 1
                user_text = f"{user_text}\nAdditional information: {answer}"
                continue
            assistant_reply = str(parsed.get("assistant_reply") or "").strip()
            if assistant_reply:
                self.logger.info(f"HITL reply: {assistant_reply}")
            intent_summary = str(parsed.get("intent_summary") or "").strip()
            if intent_summary:
                self.logger.info(f"HITL interpretation: {intent_summary}")
            applied_cfg = self._hitl_apply_config_overrides(state, parsed.get("config_overrides"))
            if applied_cfg:
                self.logger.info(f"HITL config overrides: {applied_cfg}")
            applied_step = self._hitl_apply_step_overrides(state, node_name=node_name, overrides=parsed.get("step_overrides"))
            if applied_step:
                self.logger.info(f"HITL step overrides: {applied_step}")
            action = str(parsed.get("action") or "").strip()
            if action == "skip" or action == "modify_then_skip":
                return "skip", {"mode": "parsed_skip", "action": action, "user_text": user_text, "parsed": parsed, "clarifications": clarifications}
            if action == "exit_hitl":
                if isinstance(cfg, dict):
                    cfg["interaction_mode"] = "auto"
                return "continue", {"mode": "disabled_by_user_parsed", "action": action, "user_text": user_text, "parsed": parsed, "clarifications": clarifications}
            if action in {"continue", "modify_then_continue"}:
                return "continue", {"mode": "parsed_continue", "action": action, "user_text": user_text, "parsed": parsed, "clarifications": clarifications}
            user_text = ""
        parsed_last = self._hitl_interpret_step(
            self.llm,
            user_text=user_text,
            node_name=node_name,
            label=label,
            preview=preview,
            allowed_overrides=allowed_overrides,
            no_clarification=True,
        )
        return "continue", {"mode": "max_rounds_reached", "parsed": parsed_last, "clarifications": clarifications}

    def _maybe_hitl_override_decision(self, *, state: dict, context: dict, llm: Any, suggested_decision: str, valid_decisions: List[str]) -> tuple[str, Optional[Dict[str, Any]]]:
        if not self._hitl_enabled(state):
            return suggested_decision, None
        if not _stdin_is_interactive():
            return suggested_decision, None
        cfg = state.get("config") if isinstance(state, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        if not bool(cfg.get("hitl_decision_enabled", False)):
            return suggested_decision, None
        max_rounds_raw = cfg.get("hitl_max_rounds", 5)
        max_clarifications_raw = cfg.get("hitl_max_clarifications", 5)
        try:
            max_rounds = int(max_rounds_raw)
        except Exception:
            max_rounds = 5
        try:
            max_clarifications = int(max_clarifications_raw)
        except Exception:
            max_clarifications = 5
        if max_rounds < 1:
            max_rounds = 1
        if max_rounds > 10:
            max_rounds = 10
        if max_clarifications < 0:
            max_clarifications = 0
        if max_clarifications > 10:
            max_clarifications = 10
        clarifications = 0
        user_text = ""
        for _ in range(max_rounds):
            if not user_text:
                self.logger.info('[INPUT] HITL mode: enter your instruction or question (Enter=accept the system suggestion; skip=skip; exit=leave HITL)')
                user_text = _input_with_log_prefix(self.logger).strip()
                if not user_text:
                    return suggested_decision, {"mode": "accepted_default", "suggested_next_agent": suggested_decision}
                user_norm = user_text.strip().lower()
                if user_norm in {"skip", 'skip'}:
                    return suggested_decision, {"mode": "skipped", "suggested_next_agent": suggested_decision}
                if user_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return suggested_decision, {"mode": "disabled_by_user", "suggested_next_agent": suggested_decision}
            parsed = self._hitl_interpret(
                llm,
                user_text=user_text,
                context=context,
                suggested_next_agent=suggested_decision,
                valid_decisions=valid_decisions,
            )
            need_clarification = bool(parsed.get("need_clarification", False))
            clarifying_question = str(parsed.get("clarifying_question") or "").strip()
            if need_clarification and clarifying_question:
                if clarifications >= max_clarifications:
                    self.logger.info('The HITL clarification limit has been reached. Continue with the system suggestion without further questions. You can provide a clearer instruction at the next decision point.')
                    return suggested_decision, {
                        "mode": "clarification_limit_reached",
                        "suggested_next_agent": suggested_decision,
                        "user_text": user_text,
                        "parsed": parsed,
                        "clarifications": clarifications,
                        "max_clarifications": max_clarifications,
                    }
                self.logger.info(f"[INPUT] HITL clarification required: {clarifying_question} (skip=skip; exit=leave HITL)")
                answer = _input_with_log_prefix(self.logger).strip()
                answer_norm = answer.strip().lower()
                if answer_norm in {"skip", 'skip'}:
                    return suggested_decision, {"mode": "skipped_on_clarification", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed}
                if answer_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return suggested_decision, {"mode": "disabled_by_user_on_clarification", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed}
                if not answer:
                    return suggested_decision, {"mode": "cancelled_empty_clarification", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed}
                clarifications += 1
                user_text = f"{user_text}\nAdditional information: {answer}"
                continue
            assistant_reply = str(parsed.get("assistant_reply") or "").strip()
            if assistant_reply:
                self.logger.info(f"HITL reply: {assistant_reply}")
            intent_summary = str(parsed.get("intent_summary") or "").strip()
            proposed_next_agent_obj = parsed.get("proposed_next_agent", None)
            proposed_next_agent = None
            if isinstance(proposed_next_agent_obj, str):
                proposed_next_agent = proposed_next_agent_obj.strip()
            if proposed_next_agent not in valid_decisions:
                proposed_next_agent = None
            applied_overrides = self._hitl_apply_config_overrides(state, parsed.get("config_overrides"))
            next_agent_preview = proposed_next_agent or suggested_decision
            summary_line = intent_summary if intent_summary else '(no intent summary provided)'
            self.logger.info(f"HITL interpretation: {summary_line}")
            self.logger.info(f"HITL suggested next step: {next_agent_preview}")
            if applied_overrides:
                self.logger.info(f"HITL config overrides: {applied_overrides}")
            self.logger.info('[INPUT] Confirm: Enter/y/yes/1=execute; skip=skip this time; exit=leave HITL and accept the system suggestion; otherwise enter corrections')
            confirm = _input_with_log_prefix(self.logger).strip()
            confirm_norm = confirm.strip().lower()
            if not confirm_norm or confirm_norm in {"y", "yes", "1", 'yes', 'yes', "ok"}:
                final_decision = proposed_next_agent or suggested_decision
                return final_decision, {
                    "mode": "confirmed",
                    "suggested_next_agent": suggested_decision,
                    "final_next_agent": final_decision,
                    "user_text": user_text,
                    "parsed": parsed,
                    "applied_config_overrides": applied_overrides,
                    "clarifications": clarifications,
                }
            if confirm_norm in {"skip", 'skip'}:
                return suggested_decision, {"mode": "skipped", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed, "applied_config_overrides": applied_overrides}
            if confirm_norm in {"exit", "quit", "q"}:
                if isinstance(cfg, dict):
                    cfg["interaction_mode"] = "auto"
                return suggested_decision, {
                    "mode": "disabled_by_user",
                    "suggested_next_agent": suggested_decision,
                    "user_text": user_text,
                    "parsed": parsed,
                    "applied_config_overrides": applied_overrides,
                }
            user_text = confirm
        parsed_last = self._hitl_interpret(
            llm,
            user_text=user_text,
            context=context,
            suggested_next_agent=suggested_decision,
            valid_decisions=valid_decisions,
            no_clarification=True,
        )
        return suggested_decision, {"mode": "max_rounds_reached", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed_last, "clarifications": clarifications}

    def _emit_progress(self, event: str, node: str, progress: int, message: str) -> None:
        try:
            progress_int = int(progress)
        except Exception:
            progress_int = 0
        if progress_int < self._progress_value:
            progress_int = self._progress_value
        else:
            self._progress_value = progress_int
        hook = self.progress_hook
        if not hook:
            return
        try:
            hook(str(event), str(node), int(progress_int), str(message))
        except Exception:
            return
    def _node_progress_range(self, node: str) -> Optional[tuple]:
        progress_map = {
            "initialize": (5, 20, 'Initialization'),
            "agent_decision": (20, 25, 'Decision'),
            "data_science_expert": (25, 55, 'Data Science Expert'),
            "geology_analysis": (55, 75, 'Geology Expert'),
            "result_output": (75, 95, 'result_output'),
            "final_decision": (95, 100, 'Final Evaluation'),
        }
        return progress_map.get(node)

    def _node_preview_text(self, node_name: str, state: dict, label: str) -> str:
        try:
            data_obj = state.get("data") if isinstance(state, dict) else None
        except Exception:
            data_obj = None
        data_shape = None
        try:
            if isinstance(data_obj, pd.DataFrame):
                data_shape = data_obj.shape
        except Exception:
            data_shape = None
        processed_data = state.get("processed_data") if isinstance(state, dict) else None
        preprocessed_data = state.get("preprocessed_data") if isinstance(state, dict) else None
        processed_shape = None
        preprocessed_shape = None
        try:
            if isinstance(processed_data, pd.DataFrame):
                processed_shape = processed_data.shape
        except Exception:
            processed_shape = None
        try:
            if isinstance(preprocessed_data, pd.DataFrame):
                preprocessed_shape = preprocessed_data.shape
        except Exception:
            preprocessed_shape = None
        analysis_results = state.get("analysis_results") if isinstance(state, dict) else None
        analysis_keys: List[str] = []
        try:
            if isinstance(analysis_results, dict):
                analysis_keys = [str(k) for k in list(analysis_results.keys())[:12]]
        except Exception:
            analysis_keys = []
        show_task_plan_summary = False
        cfg = state.get("config") if isinstance(state, dict) else None
        if isinstance(cfg, dict):
            show_task_plan_summary = bool(cfg.get("hitl_show_task_plan_summary", False))
        task_plan = state.get("task_plan") if isinstance(state, dict) else None
        task_plan_head = ""
        if show_task_plan_summary:
            try:
                if isinstance(task_plan, list) and task_plan and isinstance(task_plan[0], str):
                    task_plan_head = task_plan[0].strip().replace("\r\n", "\n").replace("\r", "\n")
                    task_plan_head = task_plan_head[:240]
                    task_plan_head = " ".join(task_plan_head.split())
            except Exception:
                task_plan_head = ""
        task_plan_summary_shown = False
        if isinstance(state, dict):
            task_plan_summary_shown = bool(state.get("_hitl_task_plan_summary_shown", False))
        model_obj = state.get("model") if isinstance(state, dict) else None
        preds_obj = state.get("predictions") if isinstance(state, dict) else None
        has_model = model_obj is not None
        has_predictions = preds_obj is not None
        output_results = state.get("output_results") if isinstance(state, dict) else None
        has_output_results = isinstance(output_results, dict) and bool(output_results)
        has_geology_results = isinstance(analysis_results, dict) and ("geology" in analysis_results)

        action = f"Execute step: {label}"
        if node_name == "initialize":
            action = 'Generate the global task plan and write it to reports/task_plan.md'
        elif node_name == "agent_decision":
            action = 'Decide which agent to call next based on the current state and task plan'
        elif node_name == "data_science_expert":
            geology_ctx_present = False
            try:
                geology_ctx_present = bool(state.get("geology_expert_results")) or bool((isinstance(analysis_results, dict) and analysis_results.get("geology")))
            except Exception:
                geology_ctx_present = bool(isinstance(analysis_results, dict) and analysis_results.get("geology"))
            if not geology_ctx_present:
                action = 'Run raw-data quality/distribution analysis, derive a preprocessing strategy, and complete cleaning/transformation/scaling'
            elif (processed_shape is None and preprocessed_shape is None) and (not has_model and not has_predictions):
                action = 'Run quick preprocessing, then build a prediction model using geological analysis results and generate predictions/scores'
            elif not has_model and not has_predictions:
                action = 'Build a prediction model using geological analysis results and generate predictions/scores'
            elif has_model and not has_predictions:
                action = 'Generate predictions/scores from the existing model'
            elif not has_model and has_predictions:
                action = 'Train the missing model and reconcile it with existing predictions/scores'
            else:
                action = 'Add feature engineering, statistical analysis, or diagnostics to improve results for geological interpretation'
        elif node_name == "geology_analysis":
            if not has_predictions and not has_model:
                action = 'Use preprocessing/statistical results for preliminary geochemical anomaly assessment and element-combination interpretation'
            elif not has_geology_results:
                action = 'Use modeling/prediction results for integrated geological interpretation and produce geology results'
            else:
                action = 'Extend existing geology results with spatial interpretation and metallogenic-pattern inference'
        elif node_name == "result_output":
            if has_output_results:
                action = 'Update reports and visualizations from existing results'
            else:
                action = 'Aggregate existing results and generate reports and visualizations'
        elif node_name == "final_decision":
            action = 'Perform the final evaluation and end the workflow'
        lines = [f"This step will execute: {action}"]
        if data_shape is not None:
            lines.append(f"Current data: shape={data_shape}")
        if processed_shape is not None or preprocessed_shape is not None:
            lines.append(f"Processed data: processed={processed_shape}, preprocessed={preprocessed_shape}")
        if has_model or has_predictions:
            lines.append(f"Model status: model={'yes' if has_model else 'no'}, predictions={'yes' if has_predictions else 'no'}")
        if node_name == "data_science_expert" and ('Build the prediction model' in action or 'predictions/scores' in action or 'Train the missing model' in action):
            lines.append('Available model: SOM (QE)')
        if analysis_keys:
            lines.append(f"Existing result keys: {analysis_keys}")
        if task_plan_head and not task_plan_summary_shown:
            lines.append(f"Task-plan summary: {task_plan_head}")
            if isinstance(state, dict):
                state["_hitl_task_plan_summary_shown"] = True
        return "\n".join(lines)

    def _wrap_node(self, node_name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        def wrapped(state: dict) -> dict:
            rng = self._node_progress_range(node_name)
            label = node_name
            if rng:
                start, end, label = rng
                self._emit_progress("node_start", node_name, start, f"Entering step: {label}")
            self._governance_pre(node_name, state)
            if (
                node_name != "final_decision"
                and isinstance(state, dict)
                and state.get("next_agent") == "final_decision"
                and state.get("current_phase") in {"budget_exceeded", "abort"}
            ):
                if rng:
                    _, end, label = rng
                    self._emit_progress("node_end", node_name, end, f"Completed step: {label}")
                return state
            if self._hitl_enabled(state) and _stdin_is_interactive() and node_name not in {"agent_decision", "initialize", "geology_analysis", "result_output", "final_decision"}:
                preview = self._node_preview_text(node_name, state, label)
                decision, meta = self._maybe_hitl_dialog_for_node(state=state, node_name=node_name, label=label, preview=preview)
                if meta is not None:
                    state.setdefault("processing_history", []).append(f"HITL({node_name}): {meta.get('mode')}")
                if decision == "skip":
                    state.setdefault("processing_history", []).append(f"HITL skipped step: {label}")
                    if node_name not in {"final_decision"}:
                        state["next_agent"] = "agent_decision"
                    if rng:
                        _, end, _ = rng
                        self._emit_progress("node_end", node_name, end, f"Completed step: {label}")
                    return state
            try:
                result = fn(state)
            except Exception:
                rng2 = self._node_progress_range(node_name)
                if rng2:
                    _, _, label = rng2
                    self._emit_progress("node_error", node_name, self._progress_value, f"Step error: {label}")
                raise
            if node_name not in {"agent_decision", "final_decision"}:
                try:
                    self._governance_post(node_name, result)
                except Exception:
                    pass
            rng3 = self._node_progress_range(node_name)
            if rng3:
                _, end, label = rng3
                self._emit_progress("node_end", node_name, end, f"Completed step: {label}")
            return result
        return wrapped
    def _build_workflow(self):
        workflow = StateGraph(workflow_state)
        workflow.add_node('initialize', self._wrap_node('initialize', self._initialize_workflow))
        workflow.add_node('agent_decision', self._wrap_node('agent_decision', self._agent_decision))
        workflow.add_node('data_science_expert', self._wrap_node('data_science_expert', self._run_data_science_expert))
        workflow.add_node('geology_analysis', self._wrap_node('geology_analysis', self._run_geology_analysis))
        workflow.add_node('result_output', self._wrap_node('result_output', self._run_result_output))
        workflow.add_node('final_decision', self._wrap_node('final_decision', self._final_decision))
        workflow.set_entry_point('initialize')
        workflow.add_conditional_edges(
            'initialize',
            self._route_after_initialize,
            {
                'agent_decision': 'agent_decision',
                'final_decision': 'final_decision',
            },
        )
        workflow.add_conditional_edges(
            'agent_decision',
            self._decide_next_step,
            {
                'data_science_expert': 'data_science_expert',
                'geology_analysis': 'geology_analysis',
                'result_output': 'result_output',
                'final_decision': 'final_decision',
            },
        )
        workflow.add_conditional_edges(
            'data_science_expert',
            self._route_p2p,
            {
                'data_science_expert': 'data_science_expert',
                'geology_analysis': 'geology_analysis',
                'result_output': 'result_output',
                'agent_decision': 'agent_decision',
                'final_decision': 'final_decision',
            },
        )
        workflow.add_conditional_edges(
            'geology_analysis',
            self._route_p2p,
            {
                'data_science_expert': 'data_science_expert',
                'geology_analysis': 'geology_analysis',
                'result_output': 'result_output',
                'agent_decision': 'agent_decision',
                'final_decision': 'final_decision',
            },
        )
        workflow.add_conditional_edges(
            'result_output',
            self._route_p2p,
            {
                'data_science_expert': 'data_science_expert',
                'geology_analysis': 'geology_analysis',
                'result_output': 'result_output',
                'agent_decision': 'agent_decision',
                'final_decision': 'final_decision',
            },
        )
        workflow.add_edge('final_decision', END)
        compiled_workflow = workflow.compile()
        return compiled_workflow
    def _initialize_workflow(self, state: dict) -> dict:
        try:
            self.logger.info('Starting workflow initialization...')
            if 'processing_history' not in state:
                state['processing_history'] = []
            if 'errors' not in state:
                state['errors'] = []
            if 'analysis_results' not in state:
                state['analysis_results'] = {}
            if 'decision_history' not in state:
                state['decision_history'] = []
            geology_enabled = self._geology_expert_enabled(state)
            active_agents = [self.data_science_expert_agent, self.result_output_agent]
            if geology_enabled:
                active_agents.insert(1, self.geology_expert_agent)
            cap_blocks: List[str] = ['The system provides the following agents and detailed capabilities:']
            for agent in active_agents:
                text = ""
                try:
                    text = agent.describe_capabilities_text(include_skills=True)
                except Exception:
                    try:
                        text = agent.describe_skills_text()
                    except Exception:
                        text = ""
                if isinstance(text, str) and text.strip():
                    cap_blocks.append(text.strip())
            system_capabilities = "\n\n".join(cap_blocks).strip() + "\n"
            self.logger.info('Generating the task plan with the LLM...')
            target_deposit = state.get('target_deposit_type', 'Unknown target deposit')
            task = (
                f"Based on the geochemical analysis goals (identify anomalies related to {target_deposit}, predict mineralization potential, and perform geological interpretation), generate a stage-level collaboration plan strictly from the existing multi-agent capabilities below.\n\n"
                f"{system_capabilities}\n\n"
                'Output requirements:\n'
                '1) Output Markdown only. Do not add any extra explanation.\n'
                '2) The plan must be stage-level and contain 4-6 stages.\n'
                '3) For each stage, include only: responsible agent name (must come from the available agents), stage objective, and 1-3 key deliverables.\n'
                '4) Do not mention specific algorithms/models (for example LightGBM, XGBoost, or neural networks) unless explicitly required by the system capabilities.\n'
                '5) Do not break stages into substeps such as 1.1/1.2, and do not include code or parameters.\n'
                '6) The stage order should follow: data quality and feature preparation -> geological interpretation -> result aggregation -> final evaluation.\n'
            )
            self.logger.info('Requesting a collaboration task plan from the geology expert agent...')
            task_plan_final = ""
            try:
                if geology_enabled:
                    task_plan = self.geology_expert_agent.decide(task, config=state.get('config'))
                    if not isinstance(task_plan, str) or not task_plan.strip() or str(task_plan).strip().startswith("Error:"):
                        raise ValueError(f"invalid_task_plan: {str(task_plan)[:120]}")
                    task_plan_final = str(task_plan).strip()
                    self.logger.info('Task-plan generation completed.')
                else:
                    task_plan_final = _default_task_plan_markdown(
                        str(target_deposit),
                        self._available_agent_names(state),
                        geology_expert_enabled=False,
                    )
                    self.logger.info('Geology expert disabled; using the fixed ablation plan without geological interpretation')
            except Exception as e:
                if _is_billing_error(e) or _is_auth_error(e):
                    self.logger.error(f"Task-plan generation failed (model service unavailable, billing issue, or permission issue). Continue with the built-in default plan. Error summary: {str(e)[:300]}")
                    task_plan_final = _default_task_plan_markdown(str(target_deposit), self._available_agent_names(state), geology_expert_enabled=geology_enabled)
                else:
                    self.logger.warning(f"Task-plan generation failed. Continue with the built-in default plan. Error summary: {str(e)[:300]}")
                    task_plan_final = _default_task_plan_markdown(str(target_deposit), self._available_agent_names(state), geology_expert_enabled=geology_enabled)
            try:
                task_plan_final, hitl_meta = self._maybe_hitl_confirm_task_plan(
                    state=state,
                    llm=self.llm,
                    system_capabilities=system_capabilities,
                    task_plan=task_plan_final,
                    available_agents=self._available_agent_names(state),
                )
                if hitl_meta is not None:
                    state.setdefault("processing_history", []).append(f"HITL task-plan confirmation: {hitl_meta.get('mode')}")
            except Exception as e:
                self.logger.warning(f"HITL task-plan confirmation failed. Continue with the current plan. Error summary: {str(e)[:200]}")
            reports_dir = os.path.join(self.output_dir, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            task_plan_path = os.path.join(reports_dir, 'task_plan.md')
            file_path = os.path.abspath(task_plan_path)
            fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.md', dir=reports_dir, text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    f.write(task_plan_final)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, file_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            self.logger.info(f'Task plan saved to: {task_plan_path}')
            state['task_plan'] = [task_plan_final]
            state['processing_history'].append('System initialization: generated task plan')
            state['current_phase'] = 'initialization'
            self.logger.info('System initialization: multi-agent task planning completed')
        except Exception as e:
            if _is_auth_error(e):
                raise
            self.logger.exception(f'Initialization failed: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'Initialization failed: {str(e)}')
            geology_enabled = self._geology_expert_enabled(state)
            state['task_plan'] = [_default_task_plan_markdown(str(state.get("target_deposit_type") or 'Unknown target deposit'), self._available_agent_names(state), geology_expert_enabled=geology_enabled)]
        return state

    def _route_after_initialize(self, state: dict) -> str:
        if not isinstance(state, dict):
            return "agent_decision"
        if state.get("next_agent") == "final_decision" or state.get("current_phase") in {"budget_exceeded", "abort"}:
            return "final_decision"
        return "agent_decision"

    def _route_p2p(self, state: dict) -> str:
        history = state.get('processing_history', [])
        if len(history) > 50:
            self.logger.warning('Potential infinite loop detected (history > 50). Routing to final_decision.')
            return 'final_decision'
        next_agent = state.get('next_agent')
        if not self._geology_expert_enabled(state) and next_agent == 'geology_analysis':
            has_pred = isinstance(state.get('prediction_results'), dict) and bool(state.get('prediction_results'))
            next_agent = 'result_output' if has_pred else 'data_science_expert'
            state['next_agent'] = next_agent
        if isinstance(state, dict):
            try:
                phase = state.get("current_phase")
                has_pred = isinstance(state.get("prediction_results"), dict) and bool(state.get("prediction_results"))
                has_out = isinstance(state.get("output_results"), dict) and bool(state.get("output_results"))
                has_result_output = any(('ResultOutputAgent' in item or 'result_output' in item for item in history))
                if has_pred and (not has_out) and (not has_result_output) and phase != "abort":
                    return "result_output"
            except Exception:
                pass
        if not next_agent or next_agent == 'agent_decision':
            return 'agent_decision'
        valid_nodes = list(self.agents.keys()) + ['final_decision', 'agent_decision']
        if next_agent not in valid_nodes:
            self.logger.warning(f"Invalid next_agent '{next_agent}'. Routing to agent_decision.")
            return 'agent_decision'
        return next_agent
    def _agent_decision(self, state: dict) -> dict:
        try:
            if not self._geology_expert_enabled(state):
                has_processed = state.get('processed_data') is not None or state.get('preprocessed_data') is not None
                has_prediction = isinstance(state.get('prediction_results'), dict) and bool(state.get('prediction_results'))
                has_output = isinstance(state.get('output_results'), dict) and bool(state.get('output_results'))
                if not has_processed or not has_prediction:
                    decision = 'data_science_expert'
                elif not has_output:
                    decision = 'result_output'
                else:
                    decision = 'final_decision'
                state['next_agent'] = decision
                state['current_phase'] = decision
                state.setdefault('decision_history', []).append(
                    {
                        'raw_decision': 'rule:geology_expert_disabled',
                        'mapped_decision': decision,
                        'context': {'geology_expert_enabled': False},
                        'timestamp': pd.Timestamp.now().isoformat(),
                    }
                )
                return state
            if state.get('processed_data') is None and state.get('preprocessed_data') is None:
                suggested_decision = 'data_science_expert'
                processed_data = state.get('processed_data')
                processed_shape = processed_data.shape if processed_data is not None else None
                analysis_results = state.get('analysis_results', {})
                analysis_results_brief = {k: v.keys() if isinstance(v, dict) else str(v) for k, v in analysis_results.items()} if isinstance(analysis_results, dict) else {}
                context = {
                    'current_state': {
                        'phase': state.get('current_phase'),
                        'processed_data': processed_shape,
                        'analysis_results': analysis_results_brief,
                        'errors': state.get('errors'),
                        'processing_history': state.get('processing_history'),
                    },
                    'available_agents': list(self.agents.keys()),
                    'task_plan': state.get('task_plan'),
                }
                llm = self.llm
                valid_decisions = ['data_science_expert', 'geology_analysis', 'result_output', 'final_decision']
                hitl_decision, hitl_meta = self._maybe_hitl_override_decision(
                    state=state,
                    context=context,
                    llm=llm,
                    suggested_decision=suggested_decision,
                    valid_decisions=valid_decisions,
                )
                decision_to_use = hitl_decision if hitl_decision in valid_decisions else suggested_decision
                state['next_agent'] = decision_to_use
                state['current_phase'] = decision_to_use
                state.setdefault('decision_history', []).append(
                    {
                        'raw_decision': 'rule:data_science_first',
                        'mapped_decision': decision_to_use,
                        'context': {'reason': 'processed_data/preprocessed_data is None'},
                        'timestamp': pd.Timestamp.now().isoformat(),
                    }
                )
                if hitl_meta is not None:
                    state.setdefault('decision_history', []).append(
                        {
                            'raw_decision': 'hitl',
                            'mapped_decision': decision_to_use,
                            'context': {'hitl': hitl_meta, 'current_state': context.get('current_state')},
                            'timestamp': pd.Timestamp.now().isoformat(),
                        }
                    )
                return state
            history_len = len(state.get('processing_history', []))
            if history_len > 20:
                self.logger.warning('Too many workflow steps. Forcing final evaluation to avoid an infinite loop.')
                state['next_agent'] = 'final_decision'
                state['current_phase'] = 'force_termination'
                return state
            processed_data = state.get('processed_data')
            processed_shape = processed_data.shape if processed_data is not None else None
            analysis_results = state.get('analysis_results', {})
            analysis_results_brief = {k: v.keys() if isinstance(v, dict) else str(v) for k, v in analysis_results.items()} if isinstance(analysis_results, dict) else {}
            context = {
                'current_state': {
                    'phase': state.get('current_phase'),
                    'processed_data': processed_shape,
                    'analysis_results': analysis_results_brief,
                    'errors': state.get('errors'),
                    'processing_history': state.get('processing_history'),
                },
                'available_agents': list(self.agents.keys()),
                'task_plan': state.get('task_plan'),
            }
            prompt = f"As the decision center of a multi-agent geochemical analysis system, decide which agent should be called next, or whether the workflow should end, based on the current state and task plan.\n\nCurrent state: {context['current_state']}\n\nAvailable agents and responsibilities:\n- data_science_expert: data preprocessing, cleaning, normalization, predictive modeling\n- geology_analysis: geological analysis, anomaly detection, element-combination identification, spatial analysis\n- result_output: report generation, visualization, and data export\n- final_decision: end the workflow\n\nTask plan: {context['task_plan']}\n\nImportant guidance:\n1. Return data_science_expert if data preprocessing, cleaning, or modeling is needed.\n2. Return geology_analysis if geological analysis, anomaly detection, or element analysis is needed.\n3. Return result_output if reports or visualizations need to be generated.\n4. Return final_decision if all tasks are complete.\n\nReturn exactly one of the following and nothing else:\ndata_science_expert\ngeology_analysis\nresult_output\nfinal_decision"
            llm = self.llm
            raw_decision = llm.invoke(prompt).content.strip()
            agent_name_mapping = {'data_preprocessing': 'data_science_expert', 'data_science': 'data_science_expert', 'data_processing': 'data_science_expert', 'preprocessing': 'data_science_expert', 'modeling': 'data_science_expert', 'prediction': 'data_science_expert', 'data_science_expert': 'data_science_expert', 'geology': 'geology_analysis', 'geology_expert': 'geology_analysis', 'geological_analysis': 'geology_analysis', 'anomaly_detection': 'geology_analysis', 'element_analysis': 'geology_analysis', 'geology_analysis': 'geology_analysis', 'output': 'result_output', 'report': 'result_output', 'visualization': 'result_output', 'result': 'result_output', 'result_output': 'result_output', 'final': 'final_decision', 'end': 'final_decision', 'finish': 'final_decision', 'final_decision': 'final_decision'}
            decision = agent_name_mapping.get(raw_decision.lower(), raw_decision)
            valid_decisions = ['data_science_expert', 'geology_analysis', 'result_output', 'final_decision']
            if decision not in valid_decisions:
                self.logger.warning(f'The LLM returned an invalid decision: {raw_decision}. Falling back to the default decision logic.')
                decision = self._default_decision(state)
            hitl_decision, hitl_meta = self._maybe_hitl_override_decision(
                state=state,
                context=context,
                llm=llm,
                suggested_decision=decision,
                valid_decisions=valid_decisions,
            )
            decision_to_use = hitl_decision if hitl_decision in valid_decisions else decision
            self.logger.info(f'Decision result: {decision}')
            state['decision_history'].append({'raw_decision': raw_decision, 'mapped_decision': decision, 'context': context, 'timestamp': pd.Timestamp.now().isoformat()})
            if hitl_meta is not None:
                state['decision_history'].append(
                    {
                        'raw_decision': 'hitl',
                        'mapped_decision': decision_to_use,
                        'context': {'hitl': hitl_meta, 'current_state': context.get('current_state')},
                        'timestamp': pd.Timestamp.now().isoformat(),
                    }
                )
            state['next_agent'] = decision_to_use
            state['current_phase'] = decision_to_use
        except Exception as e:
            if _is_auth_error(e):
                raise
            self.logger.exception(f'Agent decision failed: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'Agent decision failed: {str(e)}')
            state['next_agent'] = self._default_decision(state)
        return state
    def _decide_next_step(self, state: dict) -> str:
        return state.get('next_agent', 'final_decision')
    def _default_decision(self, state: dict) -> str:
        history = state.get('processing_history', [])
        has_data_science = any(('DataScienceExpertAgent' in item or 'data preprocessing' in item or 'data science' in item for item in history))
        has_geology = any(('GeologyExpertAgent' in item or 'geology analysis' in item or 'Geology' in item for item in history))
        has_result_output = any(('ResultOutputAgent' in item or 'result_output' in item for item in history))
        self.logger.debug(f'Default decision check - data science: {has_data_science}, geology: {has_geology}, result output: {has_result_output}')
        if not has_data_science:
            return 'data_science_expert'
        elif self._geology_expert_enabled(state) and not has_geology:
            return 'geology_analysis'
        elif not has_result_output:
            return 'result_output'
        else:
            return 'final_decision'
    def _run_data_science_expert(self, state: dict) -> dict:
        return self.data_science_expert_agent.run(state)
    def _run_geology_analysis(self, state: dict) -> dict:
        if not self._geology_expert_enabled(state):
            state.setdefault('processing_history', []).append('GeologyExpertAgent: skipped according to the ablation configuration')
            state['geology_expert_results'] = None
            if isinstance(state.get('analysis_results'), dict):
                state['analysis_results'].pop('geology', None)
            state['next_agent'] = 'data_science_expert' if not state.get('prediction_results') else 'result_output'
            return state
        result = self.geology_expert_agent.run(state)
        if isinstance(result, dict) and 'next_agent' in result:
            state['next_agent'] = result['next_agent']
        else:
            state['next_agent'] = None
        return result
    def _run_result_output(self, state: dict) -> dict:
        return self.result_output_agent.run(state)
    def _final_decision(self, state: dict) -> dict:
        try:
            task = 'Evaluate the results of the following geochemical data-analysis workflow, judge whether the task was completed successfully, and identify possible improvement directions.\n\nResult summary:\n'
            summary = {'processing_history': state.get('processing_history'), 'errors': state.get('errors'), 'analysis_results': {k: v.keys() if isinstance(v, dict) else str(v) for k, v in state.get('analysis_results', {}).items()}}
            llm = self.llm
            evaluation = llm.invoke(task + str(summary)).content
            state['final_evaluation'] = evaluation
            state['processing_history'].append('Workflow completed: final evaluation finished')
        except Exception as e:
            state['errors'].append(f'Final decision failed: {str(e)}')
            state['final_evaluation'] = 'Unable to evaluate workflow results'
        return state
    def run(
        self,
        data_path: str,
        target_deposit_type: Optional[str] = None,
        study_area_location: Optional[str] = None,
        config: Optional[dict] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> dict:
        run_start_perf = time.perf_counter()
        self.logger.info(f'Starting geochemical workflow - data file: {data_path}, target deposit: {target_deposit_type}')
        self._emit_progress("run_start", "run", 1, 'Starting workflow')
        cfg = config or {}
        if study_area_location is None and isinstance(cfg, dict):
            study_area_location = cfg.get("study_area_location")
        if data is None:
            if not data_path:
                raise ValueError('data_path is empty and no in-memory data was provided')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'Data file does not exist: {data_path}')
        if data is None:
            self.logger.info('Loading data file...')
            try:
                self._emit_progress("data_loading", "run", 3, 'Loading data')
                data = load_data(data_path)
                self.logger.info(f'Data loading completed - shape: {data.shape}')
                self.logger.debug(f'Data columns: {list(data.columns)}')
                self.logger.debug(f"Data type summary: {len(data.dtypes)} columns total (int64: {(data.dtypes == 'int64').sum()}, float64: {(data.dtypes == 'float64').sum()})")
                self._emit_progress("data_loaded", "run", 5, 'Data loading completed')
            except Exception as e:
                self.logger.exception(f'Data loading failed: {str(e)}')
                raise
        else:
            self.logger.info(f'Using preloaded data - shape: {data.shape}')
            self._emit_progress("data_loaded", "run", 5, 'Data is ready')
        self.logger.info('Initializing workflow state...')
        initial_state = {
            'data': data,
            'target_deposit_type': target_deposit_type,
            'study_area_location': study_area_location,
            'processed_data': None,
            'preprocessed_data': None,
            'model': None,
            'predictions': None,
            'analysis_results': {},
            'geology_expert_results': None,
            'feature_cols': [],
            'model_input_metadata': {},
            'errors': [],
            'processing_history': [],
            'agent_messages': [],
            'current_phase': None,
            'next_agent': None,
            'task_plan': [],
            'decision_history': [],
            'config': cfg,
            'hitl_step_overrides': {},
            'governance': None,
            'artifacts': {},
            'checks': [],
            'replan': {},
        }
        self.logger.info('Workflow state initialization completed')
        self.logger.info('Executing workflow...')
        try:
            result = self.workflow.invoke(initial_state, {'recursion_limit': 100})
            self.logger.info('Workflow execution completed')
            if 'analysis_results' in result:
                self.logger.info(f"Analysis result summary - result count: {len(result['analysis_results'])}")
                for key, value in result['analysis_results'].items():
                    self.logger.debug(f"Analysis result '{key}': {type(value).__name__}")
            if 'errors' in result and result['errors']:
                self.logger.warning(f"The workflow produced {len(result['errors'])} error(s) during execution")
                for idx, err in enumerate(result['errors'], start=1):
                    self.logger.warning(f"Error[{idx}]: {err}")
        except Exception as e:
            if _is_auth_error(e):
                self.logger.error(f'Workflow execution failed: {str(e)}')
            else:
                self.logger.exception(f'Workflow execution failed: {str(e)}')
            raise
        run_duration_seconds = float(time.perf_counter() - run_start_perf)
        result['workflow_runtime_seconds'] = run_duration_seconds
        result['workflow_runtime_human'] = _format_duration_cn(run_duration_seconds)
        report_path = None
        if isinstance(result.get('output_results'), dict):
            report_path = result['output_results'].get('comprehensive_report')
        if report_path:
            update_comprehensive_report_runtime(report_path, run_duration_seconds)
        try:
            final_json_path = self.result_output_agent._output_json_results(result)
            if isinstance(result.get('output_results'), dict) and isinstance(final_json_path, str) and final_json_path.strip():
                result['output_results']['json_results'] = final_json_path
        except Exception as e:
            self.logger.warning(f'Failed to write back the final JSON result: {e}')
        self.logger.info('Workflow run completed. Preparing return value.')
        self._emit_progress("run_end", "run", 100, 'Workflow completed')
        return result
    def visualize_workflow(self, output_path: str='./workflow_graph.png'):
        try:
            graph_image = self.workflow.get_graph().draw_mermaid_png()
            file_path = os.path.abspath(output_path)
            target_dir = os.path.dirname(file_path)
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix='.tmp_', suffix='.png', dir=target_dir or None)
            os.close(fd)
            try:
                with open(tmp_path, 'wb') as f:
                    f.write(graph_image)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, file_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            logger.info(f'Workflow graph saved to: {output_path}')
        except Exception as e:
            logger.exception(f'Workflow visualization failed: {str(e)}')
def create_workflow(output_dir: str='./output', progress_hook: Optional[Callable[[str, str, int, str], None]] = None):
    workflow = GeoChemistryWorkflow(output_dir=output_dir, progress_hook=progress_hook)
    logger.info(f'Geochemical workflow instance created successfully. Output directory: {output_dir}')
    return workflow
if __name__ == '__main__':
    workflow = create_workflow()
    workflow.visualize_workflow()
