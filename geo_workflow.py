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
    agents_line = ", ".join([str(a) for a in available_agents]) if available_agents else "（未知）"
    deposit = str(target_deposit or "未知矿种")
    geology_stage = (
        f"## 阶段 2：地质解译与异常识别（geology_analysis）\n"
        f"- 目标：结合 {deposit} 的成矿地质机理进行异常与元素组合解译\n"
        f"- 产出：异常要点、可能控矿因素假设、靶区圈定建议\n\n"
        if geology_expert_enabled
        else
        "## 阶段 2：全元素 SOM 建模（data_science_expert）\n"
        "- 目标：使用预处理后的全部有效地球化学元素执行 SOM 建模与 QE 异常评分\n"
        "- 产出：全元素模型、QE 异常得分与评价指标\n\n"
    )
    return (
        f"# 多智能体协作概要计划（默认）\n\n"
        f"- 目标矿种：{deposit}\n"
        f"- 可用智能体：{agents_line}\n\n"
        f"## 阶段 1：数据准备与特征提取（data_science_expert）\n"
        f"- 目标：完成数据质量检查、字段识别、基础清洗与可用特征准备\n"
        f"- 产出：数据摘要、关键字段清单、可用于后续分析的处理后数据/特征\n\n"
        f"{geology_stage}"
        f"## 阶段 3：汇总与报告（result_output）\n"
        f"- 目标：将多智能体结果结构化汇总为图表与综合报告\n"
        f"- 产出：报告文件、关键图件与结论摘要\n\n"
        f"## 阶段 4：最终评估（final_decision）\n"
        f"- 目标：评估结果可信度与下一步改进方向\n"
        f"- 产出：最终评估结论与建议清单\n"
    )
def _format_duration_cn(seconds: float) -> str:
    total_seconds = int(round(float(seconds)))
    hours = total_seconds // 3600
    minutes = total_seconds % 3600 // 60
    secs = total_seconds % 60
    if hours > 0:
        return f'{hours}小时{minutes}分{secs}秒'
    if minutes > 0:
        return f'{minutes}分{secs}秒'
    return f'{secs}秒'
def update_comprehensive_report_runtime(report_path: str, duration_seconds: float) -> bool:
    if not report_path or not isinstance(report_path, str):
        return False
    if not os.path.exists(report_path):
        return False
    duration_human = _format_duration_cn(duration_seconds)
    runtime_line = f'- **运行总耗时**: {duration_human} ({float(duration_seconds):.2f}s)'
    runtime_pattern = re.compile('^\\s*-\\s*\\*\\*运行总耗时\\*\\*:\\s*.*\\s*$')
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
            if '## 1. 项目概况' in line:
                insert_at = i + 1
                for j in range(insert_at, min(insert_at + 30, len(new_lines))):
                    if '**分析平台**' in new_lines[j]:
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
        self.logger.debug(f'初始化GeoChemistryWorkflow - 输出目录: {output_dir}')
        self.logger.info(f'语言模型初始化完成: {type(self.llm).__name__}')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.progress_hook = progress_hook
        self._progress_value = 0
        self.logger.info('开始初始化智能体...')
        self.data_science_expert_agent = DataScienceExpertAgent(llm=self.llm, output_dir=self.output_dir)
        self.logger.info('数据科学专家智能体初始化完成')
        self.geology_expert_agent = GeologyExpertAgent(llm=self.llm, output_dir=self.output_dir)
        self.logger.info('地质专家智能体初始化完成')
        self.result_output_agent = ResultOutputAgent(llm=self.llm, output_dir=self.output_dir)
        self.logger.info('结果输出智能体初始化完成')
        self.agents = {'data_science_expert': self.data_science_expert_agent, 'geology_analysis': self.geology_expert_agent, 'result_output': self.result_output_agent}
        self.logger.info(f'共初始化 {len(self.agents)} 个智能体')
        self.logger.info('开始构建工作流图...')
        self.workflow = self._build_workflow()
        self.logger.info('工作流图构建完成')

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
                    "message_to_user": "返工次数已超过预算，建议人工介入或接受保守输出并结束流程。",
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
                    "message_to_user": "未生成有效的预处理数据，需返工数据预处理步骤。",
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
                    "message_to_user": "已有地质分析结果但未生成预测结果，需返工建模/预测步骤。",
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
                        "message_to_user": "缺少特征分析上下文，需先返工数据科学步骤生成特征分析结果。",
                        "step_overrides": {},
                        "config_overrides": {},
                    }
                return {
                    "status": "needs_rework",
                    "rework_node": "geology_analysis",
                    "reason_codes": ["missing_artifact:geology_results"],
                    "message_to_user": "未生成地质分析结果，需返工地质解译步骤。",
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
                        "message_to_user": "输出阶段缺少处理后数据，需先返工数据预处理/建模步骤。",
                        "step_overrides": {},
                        "config_overrides": {},
                    }
                return {
                    "status": "needs_rework",
                    "rework_node": "result_output",
                    "reason_codes": ["missing_artifact:output_results"],
                    "message_to_user": "未生成有效输出文件，需返工结果输出步骤。",
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
            "你是多智能体地球化学工作流的治理中枢（Governor）。"
            "你的任务是：基于最新节点执行后的状态与校验结果，判断是否需要返工、人工介入或结束。"
            "\n\n"
            "你必须只输出 JSON（不要输出其他文本，不要使用 Markdown 代码块）。"
            "\n\n"
            "硬约束：\n"
            "1) status 只能是 ok | needs_rework | hitl | abort。\n"
            f"2) 当 status=needs_rework 时，rework_node 必须是以下之一：{allowed_nodes}。\n"
            "3) 严禁编造不存在的文件、指标或步骤；如信息不足只能给出保守判断。\n"
            "4) step_overrides 只能用于 rework_node 对应步骤，且必须是对象；无可执行修改则为 {}。\n"
            "5) config_overrides 只能包含以下键：structured_output_enabled, reflection_enabled, reflection_max_rounds。\n"
            "\n\n"
            f"当前上下文：{ctx}\n\n"
            "请输出 JSON，字段：\n"
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
                "message_to_user": "已达到预算上限，流程将结束或转为人工介入。",
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
                state.setdefault("processing_history", []).append("Governor: 预算超限，但允许执行结果输出收尾")
                return
            state["next_agent"] = "final_decision"
            state["current_phase"] = "budget_exceeded"
            state.setdefault("errors", []).append(f"Governor: {why}")
            state.setdefault("processing_history", []).append("Governor: 预算超限，终止后续步骤")

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
            force_clause = "重要限制：你不得再要求澄清。若信息不足，need_clarification 必须为 false，proposed_next_agent 必须为 null，并在 assistant_reply 里说明原因与建议。\n\n"
        prompt = (
            "你是多智能体系统的人机交互解释器（HITL Interpreter）。"
            "用户会输入自然语言的指令或疑问。你的任务是：理解用户意图，并把它映射成下一步工作流动作。"
            "\n\n"
            "你必须只输出 JSON（不要输出其他任何文本，不要使用 Markdown 代码块）。"
            "\n\n"
            "当你在 assistant_reply 里回答用户疑问时，必须结合当前上下文中的多智能体系统：可用智能体、当前阶段、已完成/未完成任务、系统建议下一步。不要给脱离系统的泛泛建议。"
            "\n\n"
            f"{force_clause}"
            f"当前系统建议下一步：{suggested_next_agent}\n"
            f"允许的下一步选项：{valid_decisions}\n\n"
            f"当前上下文：{context}\n\n"
            f"用户输入：{user_text}\n\n"
            "请输出 JSON，字段要求：\n"
            "- intent_summary: string，1-3 句，复述你对用户意图的理解\n"
            "- proposed_next_agent: string|null，从允许选项中选一个；如果只是提问且不改变流程，可为 null\n"
            "- assistant_reply: string，可选；若用户是疑问或需要先解释原因，在此给出回答\n"
            "- need_clarification: boolean，是否必须先问清一个关键点才能执行\n"
            "- clarifying_question: string，可选；当 need_clarification=true 时给出一个最关键的问题\n"
            "- config_overrides: object，可选；只允许以下键：structured_output_enabled, reflection_enabled, reflection_max_rounds\n"
        )
        raw = llm.invoke(prompt).content
        parsed = _safe_json_loads(raw)
        if parsed is None:
            return {
                "intent_summary": "无法从模型输出中解析结构化意图，将按系统默认建议继续。",
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
            force_clause = "重要限制：你不得再要求澄清。若信息不足，need_clarification 必须为 false，updated_task_plan_markdown 必须为 null，并在 assistant_reply 里说明原因与建议。\n\n"
        prompt = (
            "你是多智能体系统的人机交互解释器（HITL Interpreter）。"
            "用户会针对“全局任务计划”提出疑问或修改指令。你的任务是：理解用户意图，并在不引入新智能体的前提下，协助用户修订任务计划。"
            "\n\n"
            "你必须只输出 JSON（不要输出其他任何文本，不要使用 Markdown 代码块）。"
            "\n\n"
            "硬约束：\n"
            "1) 你只能使用当前系统中存在的智能体名称来编排计划，绝对禁止虚构新智能体、工具或外部系统。\n"
            "2) 你只能围绕“任务计划”本身回答与修改，不要询问与系统无关的内容。\n"
            "3) 数据已经由系统加载完毕，绝对不得询问用户提供数据文件路径或 data_path。\n"
            "4) 如确需澄清，只能提出 1 个最关键问题。\n"
            "\n\n"
            f"{force_clause}"
            f"当前可用智能体：{available_agents}\n\n"
            f"当前数据摘要：{data_summary or '（已加载；无额外摘要）'}\n\n"
            f"系统能力说明：\n{system_capabilities}\n\n"
            f"当前任务计划（Markdown）：\n{current_task_plan}\n\n"
            f"用户输入：{user_text}\n\n"
            "请输出 JSON，字段要求：\n"
            "- intent_summary: string，1-3 句，复述你对用户意图的理解\n"
            "- assistant_reply: string，可选；若用户是疑问或需要先解释原因，在此给出回答（必须基于现有多智能体系统与当前计划）\n"
            "- updated_task_plan_markdown: string|null；若用户要求修改且可执行，给出“完整的、可直接替换”的新计划 Markdown；若无需/无法修改，填 null\n"
            "- need_clarification: boolean，是否必须先问清一个关键点才能修改计划\n"
            "- clarifying_question: string，可选；当 need_clarification=true 时给出一个最关键的问题\n"
            "- config_overrides: object，可选；只允许以下键：structured_output_enabled, reflection_enabled, reflection_max_rounds\n"
        )
        raw = llm.invoke(prompt).content
        parsed = _safe_json_loads(raw)
        if parsed is None:
            return {
                "intent_summary": "无法从模型输出中解析计划修改意图，将保持原计划不变。",
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
        self.logger.info("HITL 模式：已生成全局任务计划（如需查看请使用 show）")
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
                self.logger.info("【INPUT】任务计划确认：回车=接受并继续；show=显示计划；exit=退出HITL并继续；或输入自然语言进行修改/提问")
                user_text = _input_with_log_prefix(self.logger).strip()
                if not user_text:
                    return current_plan, {"mode": "accepted", "clarifications": clarifications}
                user_norm = user_text.strip().lower()
                if user_norm in {"开始", "开始吧", "确认", "确认开始", "继续", "continue", "run", "start", "ok", "好的", "好", "对", "y", "yes", "1"}:
                    return current_plan, {"mode": "accepted", "clarifications": clarifications, "user_text": user_text}
                if user_norm in {"show", "s"}:
                    self.logger.info(f"当前任务计划：\n{current_plan}")
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
                    self.logger.info("任务计划澄清次数已达上限，将不再追问，并保持当前计划继续。")
                    return current_plan, {
                        "mode": "clarification_limit_reached",
                        "user_text": user_text,
                        "parsed": parsed,
                        "clarifications": clarifications,
                        "max_clarifications": max_clarifications,
                    }
                self.logger.info(f"【INPUT】任务计划需要澄清：{clarifying_question}（exit=退出HITL）")
                answer = _input_with_log_prefix(self.logger).strip()
                answer_norm = answer.strip().lower()
                if answer_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return current_plan, {"mode": "disabled_by_user_on_clarification", "user_text": user_text, "parsed": parsed}
                if not answer:
                    return current_plan, {"mode": "cancelled_empty_clarification", "user_text": user_text, "parsed": parsed}
                clarifications += 1
                user_text = f"{user_text}\n补充信息：{answer}"
                continue
            assistant_reply = str(parsed.get("assistant_reply") or "").strip()
            if assistant_reply:
                self.logger.info(f"HITL 回复：{assistant_reply}")
            intent_summary = str(parsed.get("intent_summary") or "").strip()
            updated_plan_obj = parsed.get("updated_task_plan_markdown", None)
            updated_plan = None
            if isinstance(updated_plan_obj, str) and updated_plan_obj.strip():
                updated_plan = updated_plan_obj.strip()
            applied_overrides = self._hitl_apply_config_overrides(state, parsed.get("config_overrides"))
            if intent_summary:
                self.logger.info(f"HITL 理解：{intent_summary}")
            if applied_overrides:
                self.logger.info(f"HITL 配置调整：{applied_overrides}")
            if updated_plan:
                current_plan = updated_plan
                self.logger.info("HITL 已生成更新后的任务计划（如需查看请使用 show）")
            self.logger.info("【INPUT】请确认：回车/对/y/yes/1=确认采用当前计划并继续；exit=退出HITL并继续；否则请输入进一步修改说明")
            confirm = _input_with_log_prefix(self.logger).strip()
            confirm_norm = confirm.strip().lower()
            if not confirm_norm or confirm_norm in {"y", "yes", "1", "对", "好", "ok", "确认", "确认开始", "开始", "继续", "continue", "run", "start"}:
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
                "skip_post_preprocess_feature_analysis": "bool，是否跳过“预处理后特征分析”子步骤",
                "post_feature_analysis_parts": {
                    "correlation": "bool，是否进行相关性分析",
                    "hierarchical": "bool，是否进行层次聚类分析",
                    "factor": "bool，是否进行因子分析",
                },
                "write_preprocessed_report": "bool，是否写入预处理后数据分析报告文件",
                "model_key": "str，强制使用指定模型（SOM）",
                "auto_programming_enabled": "bool，是否启用自动编程",
                "programming_request": "str，用户希望自动生成并执行的代码任务",
                "programming_apply_mode": "str，replace_df=替换当前数据，attach_artifact=仅挂载产物",
            }
        if node_name == "geology_analysis":
            return {
                "skip_visualizations": "bool，是否跳过关键元素可视化输出",
                "key_elements_limit": "int，关键元素可视化数量上限（1-30）",
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
                if raw in {"som", "self organizing map", "self-organizing map", "selforganizingmap", "自组织映射", "自组织映射神经网络"}:
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
            force_clause = "重要限制：你不得再要求澄清。need_clarification 必须为 false，step_overrides 必须为 {}。\n\n"
        prompt = (
            "你是多智能体系统的人机交互解释器（HITL Step Interpreter）。"
            "用户会对“当前即将执行的步骤”提出疑问或修改意见。"
            "你的任务是："
            "1) 回答用户疑问（若有）；2) 把用户对“当前步骤的方法/内容修改”映射为可执行的结构化指令。"
            "\n\n"
            "严格约束：你只能修改当前步骤的方法/内容，不能改变整体任务目标、不能要求新增数据源、不能更改全局任务计划。"
            "你只能在允许的 step_overrides 白名单内给出修改。"
            "\n\n"
            "你必须只输出 JSON（不要输出其他任何文本，不要使用 Markdown 代码块）。"
            "\n\n"
            f"{force_clause}"
            f"当前步骤：{label}（{node_name}）\n"
            f"步骤预览信息：\n{preview}\n\n"
            f"允许的 step_overrides 白名单（键与含义）：{allowed_overrides}\n\n"
            f"用户输入：{user_text}\n\n"
            "请输出 JSON，字段要求：\n"
            "- intent_summary: string，1-3 句，复述你对用户意图的理解\n"
            "- assistant_reply: string，可选；若用户在提问或需要解释原因，在此回答\n"
            "- action: string，必须为以下之一：continue | skip | exit_hitl | modify_then_continue | modify_then_skip\n"
            "- step_overrides: object，必须只包含白名单允许的键；若无可执行修改则为 {}\n"
            "- config_overrides: object，可选；仅允许修改温度/结构化输出/反思等系统配置\n"
            "- need_clarification: boolean，是否必须先问清一个关键点才能执行\n"
            "- clarifying_question: string，若 need_clarification=true，给出且只给出一个问题\n"
            "- 若用户要求当前步骤自动写代码完成数据处理任务，优先用 auto_programming_enabled/programming_request/programming_apply_mode 表达\n"
        )
        raw = llm.invoke(prompt).content
        parsed = _safe_json_loads(raw)
        if parsed is None:
            return {
                "intent_summary": "无法解析模型输出，将忽略本次修改并继续。",
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
                    f"【INPUT】HITL 模式：当前步骤可对话（回车=确认执行；skip=跳过本步；exit=退出HITL并继续；或输入自然语言修改当前步骤的方法/内容）\n{preview}"
                )
                user_text = _input_with_log_prefix(self.logger).strip()
                if not user_text:
                    return "continue", {"mode": "accepted_default"}
                user_norm = user_text.strip().lower()
                if user_norm in {"skip", "跳过"}:
                    return "skip", {"mode": "skipped_by_user"}
                if user_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return "continue", {"mode": "disabled_by_user"}
            if node_name == "data_science_expert":
                user_norm2 = user_text.strip().lower()
                forced_model = None
                if "som" in user_norm2 or "self organizing map" in user_norm2 or "self-organizing map" in user_norm2 or "自组织映射" in user_norm2:
                    forced_model = "som"
                if forced_model:
                    applied_step = self._hitl_apply_step_overrides(state, node_name=node_name, overrides={"model_key": forced_model})
                    if applied_step:
                        self.logger.info(f"HITL 步骤调整：{applied_step}")
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
                    self.logger.info("HITL 澄清次数已达上限，将按当前设置继续执行。")
                    return "continue", {"mode": "clarification_limit_reached", "user_text": user_text, "parsed": parsed}
                self.logger.info(f"【INPUT】HITL 需要澄清：{clarifying_question}（skip=跳过；exit=退出HITL）")
                answer = _input_with_log_prefix(self.logger).strip()
                answer_norm = answer.strip().lower()
                if answer_norm in {"skip", "跳过"}:
                    return "skip", {"mode": "skipped_on_clarification", "user_text": user_text, "parsed": parsed}
                if answer_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return "continue", {"mode": "disabled_by_user_on_clarification", "user_text": user_text, "parsed": parsed}
                if not answer:
                    return "continue", {"mode": "cancelled_empty_clarification", "user_text": user_text, "parsed": parsed}
                clarifications += 1
                user_text = f"{user_text}\n补充信息：{answer}"
                continue
            assistant_reply = str(parsed.get("assistant_reply") or "").strip()
            if assistant_reply:
                self.logger.info(f"HITL 回复：{assistant_reply}")
            intent_summary = str(parsed.get("intent_summary") or "").strip()
            if intent_summary:
                self.logger.info(f"HITL 理解：{intent_summary}")
            applied_cfg = self._hitl_apply_config_overrides(state, parsed.get("config_overrides"))
            if applied_cfg:
                self.logger.info(f"HITL 配置调整：{applied_cfg}")
            applied_step = self._hitl_apply_step_overrides(state, node_name=node_name, overrides=parsed.get("step_overrides"))
            if applied_step:
                self.logger.info(f"HITL 步骤调整：{applied_step}")
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
                self.logger.info("【INPUT】HITL 模式：请输入你的指令或疑问（回车=接受系统建议；skip=跳过；exit=退出HITL）")
                user_text = _input_with_log_prefix(self.logger).strip()
                if not user_text:
                    return suggested_decision, {"mode": "accepted_default", "suggested_next_agent": suggested_decision}
                user_norm = user_text.strip().lower()
                if user_norm in {"skip", "跳过"}:
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
                    self.logger.info("HITL 澄清次数已达上限，将不再追问，并按系统建议继续。你可以在下一个决策点再下达更明确的指令。")
                    return suggested_decision, {
                        "mode": "clarification_limit_reached",
                        "suggested_next_agent": suggested_decision,
                        "user_text": user_text,
                        "parsed": parsed,
                        "clarifications": clarifications,
                        "max_clarifications": max_clarifications,
                    }
                self.logger.info(f"【INPUT】HITL 需要澄清：{clarifying_question}（skip=跳过；exit=退出HITL）")
                answer = _input_with_log_prefix(self.logger).strip()
                answer_norm = answer.strip().lower()
                if answer_norm in {"skip", "跳过"}:
                    return suggested_decision, {"mode": "skipped_on_clarification", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed}
                if answer_norm in {"exit", "quit", "q"}:
                    if isinstance(cfg, dict):
                        cfg["interaction_mode"] = "auto"
                    return suggested_decision, {"mode": "disabled_by_user_on_clarification", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed}
                if not answer:
                    return suggested_decision, {"mode": "cancelled_empty_clarification", "suggested_next_agent": suggested_decision, "user_text": user_text, "parsed": parsed}
                clarifications += 1
                user_text = f"{user_text}\n补充信息：{answer}"
                continue
            assistant_reply = str(parsed.get("assistant_reply") or "").strip()
            if assistant_reply:
                self.logger.info(f"HITL 回复：{assistant_reply}")
            intent_summary = str(parsed.get("intent_summary") or "").strip()
            proposed_next_agent_obj = parsed.get("proposed_next_agent", None)
            proposed_next_agent = None
            if isinstance(proposed_next_agent_obj, str):
                proposed_next_agent = proposed_next_agent_obj.strip()
            if proposed_next_agent not in valid_decisions:
                proposed_next_agent = None
            applied_overrides = self._hitl_apply_config_overrides(state, parsed.get("config_overrides"))
            next_agent_preview = proposed_next_agent or suggested_decision
            summary_line = intent_summary if intent_summary else "（未提供意图摘要）"
            self.logger.info(f"HITL 理解：{summary_line}")
            self.logger.info(f"HITL 建议下一步：{next_agent_preview}")
            if applied_overrides:
                self.logger.info(f"HITL 配置调整：{applied_overrides}")
            self.logger.info("【INPUT】请确认：回车/对/y/yes/1=确认执行；skip=本次跳过；exit=退出HITL并接受系统建议；否则请输入纠正说明")
            confirm = _input_with_log_prefix(self.logger).strip()
            confirm_norm = confirm.strip().lower()
            if not confirm_norm or confirm_norm in {"y", "yes", "1", "对", "好", "ok"}:
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
            if confirm_norm in {"skip", "跳过"}:
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
            "initialize": (5, 20, "初始化"),
            "agent_decision": (20, 25, "决策"),
            "data_science_expert": (25, 55, "数据科学专家"),
            "geology_analysis": (55, 75, "地质专家"),
            "result_output": (75, 95, "结果输出"),
            "final_decision": (95, 100, "最终评估"),
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

        action = f"执行步骤：{label}"
        if node_name == "initialize":
            action = "生成全局任务计划并写入 reports/task_plan.md"
        elif node_name == "agent_decision":
            action = "基于当前状态与任务计划，决定下一步调用哪个智能体"
        elif node_name == "data_science_expert":
            geology_ctx_present = False
            try:
                geology_ctx_present = bool(state.get("geology_expert_results")) or bool((isinstance(analysis_results, dict) and analysis_results.get("geology")))
            except Exception:
                geology_ctx_present = bool(isinstance(analysis_results, dict) and analysis_results.get("geology"))
            if not geology_ctx_present:
                action = "执行原始数据质量/分布分析，生成预处理策略，并完成清洗/变换/缩放"
            elif (processed_shape is None and preprocessed_shape is None) and (not has_model and not has_predictions):
                action = "补做快速预处理后，结合地质分析结果构建预测模型并生成预测/评分"
            elif not has_model and not has_predictions:
                action = "结合地质分析结果构建预测模型并生成预测/评分"
            elif has_model and not has_predictions:
                action = "基于现有模型生成预测/评分"
            elif not has_model and has_predictions:
                action = "补充训练模型并对齐已有预测/评分结果"
            else:
                action = "补充特征工程/统计分析或结果诊断，完善可供地质解译的结果"
        elif node_name == "geology_analysis":
            if not has_predictions and not has_model:
                action = "基于预处理/统计结果进行地球化学异常初判与元素组合解译"
            elif not has_geology_results:
                action = "结合建模/预测结果开展综合地质解译，产出 geology 分析结果"
            else:
                action = "在已有 geology 结果上补充空间解释与成矿模式推断"
        elif node_name == "result_output":
            if has_output_results:
                action = "基于已有结果更新报告与可视化文件"
            else:
                action = "汇总已有结果并生成报告与可视化文件"
        elif node_name == "final_decision":
            action = "对流程结果进行最终评估并结束工作流"
        lines = [f"本步将执行：{action}"]
        if data_shape is not None:
            lines.append(f"当前数据：shape={data_shape}")
        if processed_shape is not None or preprocessed_shape is not None:
            lines.append(f"已处理数据：processed={processed_shape}, preprocessed={preprocessed_shape}")
        if has_model or has_predictions:
            lines.append(f"建模状态：model={'yes' if has_model else 'no'}, predictions={'yes' if has_predictions else 'no'}")
        if node_name == "data_science_expert" and ("构建预测模型" in action or "预测/评分" in action or "训练模型" in action):
            lines.append("可选模型：SOM(QE)")
        if analysis_keys:
            lines.append(f"已有结果键：{analysis_keys}")
        if task_plan_head and not task_plan_summary_shown:
            lines.append(f"任务计划摘要：{task_plan_head}")
            if isinstance(state, dict):
                state["_hitl_task_plan_summary_shown"] = True
        return "\n".join(lines)

    def _wrap_node(self, node_name: str, fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        def wrapped(state: dict) -> dict:
            rng = self._node_progress_range(node_name)
            label = node_name
            if rng:
                start, end, label = rng
                self._emit_progress("node_start", node_name, start, f"进入步骤：{label}")
            self._governance_pre(node_name, state)
            if (
                node_name != "final_decision"
                and isinstance(state, dict)
                and state.get("next_agent") == "final_decision"
                and state.get("current_phase") in {"budget_exceeded", "abort"}
            ):
                if rng:
                    _, end, label = rng
                    self._emit_progress("node_end", node_name, end, f"完成步骤：{label}")
                return state
            if self._hitl_enabled(state) and _stdin_is_interactive() and node_name not in {"agent_decision", "initialize", "geology_analysis", "result_output", "final_decision"}:
                preview = self._node_preview_text(node_name, state, label)
                decision, meta = self._maybe_hitl_dialog_for_node(state=state, node_name=node_name, label=label, preview=preview)
                if meta is not None:
                    state.setdefault("processing_history", []).append(f"HITL({node_name})：{meta.get('mode')}")
                if decision == "skip":
                    state.setdefault("processing_history", []).append(f"HITL 跳过步骤：{label}")
                    if node_name not in {"final_decision"}:
                        state["next_agent"] = "agent_decision"
                    if rng:
                        _, end, _ = rng
                        self._emit_progress("node_end", node_name, end, f"完成步骤：{label}")
                    return state
            try:
                result = fn(state)
            except Exception:
                rng2 = self._node_progress_range(node_name)
                if rng2:
                    _, _, label = rng2
                    self._emit_progress("node_error", node_name, self._progress_value, f"步骤异常：{label}")
                raise
            if node_name not in {"agent_decision", "final_decision"}:
                try:
                    self._governance_post(node_name, result)
                except Exception:
                    pass
            rng3 = self._node_progress_range(node_name)
            if rng3:
                _, end, label = rng3
                self._emit_progress("node_end", node_name, end, f"完成步骤：{label}")
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
            self.logger.info('开始工作流初始化...')
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
            cap_blocks: List[str] = ["系统具备以下智能体及其详细能力："]
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
            self.logger.info('使用大模型生成任务计划...')
            target_deposit = state.get('target_deposit_type', '未知矿种')
            task = (
                f"基于地球化学数据分析目标（识别{target_deposit}异常、预测成矿潜力、地质解译），请严格基于以下【现有多智能体系统能力】生成一份【概要协作计划】。\n\n"
                f"{system_capabilities}\n\n"
                "输出要求：\n"
                "1) 只输出 Markdown（不要输出其他解释）。\n"
                "2) 计划必须是“阶段级概要”，控制在 4-6 个阶段。\n"
                "3) 每个阶段只写：负责智能体名称（必须来自系统能力中的智能体）、阶段目标、关键产出（1-3 条）。\n"
                "4) 不要写具体算法/模型名称（例如 LightGBM、XGBoost、神经网络等），除非系统能力里明确要求。\n"
                "5) 不要拆到 1.1/1.2 这类子步骤；不要写代码或参数。\n"
                "6) 顺序应遵循：数据质量与特征准备 → 地质解译 → 结果汇总 → 最终评估。\n"
            )
            self.logger.info('生成协作任务计划...')
            task_plan_final = ""
            try:
                if geology_enabled:
                    task_plan = self.geology_expert_agent.decide(task, config=state.get('config'))
                    if not isinstance(task_plan, str) or not task_plan.strip() or str(task_plan).strip().startswith("Error:"):
                        raise ValueError(f"invalid_task_plan: {str(task_plan)[:120]}")
                    task_plan_final = str(task_plan).strip()
                    self.logger.info('任务计划生成完成')
                else:
                    task_plan_final = _default_task_plan_markdown(
                        str(target_deposit),
                        self._available_agent_names(state),
                        geology_expert_enabled=False,
                    )
                    self.logger.info('地质专家已关闭，使用不含地质解译的固定消融计划')
            except Exception as e:
                if _is_billing_error(e) or _is_auth_error(e):
                    self.logger.error(f"任务计划生成失败（模型服务不可用/欠费/权限问题），将使用内置默认计划继续。错误摘要：{str(e)[:300]}")
                    task_plan_final = _default_task_plan_markdown(str(target_deposit), self._available_agent_names(state), geology_expert_enabled=geology_enabled)
                else:
                    self.logger.warning(f"任务计划生成失败，将使用内置默认计划继续。错误摘要：{str(e)[:300]}")
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
                    state.setdefault("processing_history", []).append(f"HITL 任务计划确认：{hitl_meta.get('mode')}")
            except Exception as e:
                self.logger.warning(f"HITL 任务计划确认失败，将继续使用当前计划。错误摘要：{str(e)[:200]}")
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
            self.logger.info(f'任务计划已保存到: {task_plan_path}')
            state['task_plan'] = [task_plan_final]
            state['processing_history'].append('系统初始化：生成任务计划')
            state['current_phase'] = 'initialization'
            self.logger.info('系统初始化：多智能体协作任务规划完成')
        except Exception as e:
            if _is_auth_error(e):
                raise
            self.logger.exception(f'初始化失败: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'初始化失败: {str(e)}')
            geology_enabled = self._geology_expert_enabled(state)
            state['task_plan'] = [_default_task_plan_markdown(str(state.get("target_deposit_type") or "未知矿种"), self._available_agent_names(state), geology_expert_enabled=geology_enabled)]
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
            self.logger.warning('检测到潜在死循环(history > 50)，路由到 final_decision')
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
                has_result_output = any(('ResultOutputAgent' in item or '结果输出' in item for item in history))
                if has_pred and (not has_out) and (not has_result_output) and phase != "abort":
                    return "result_output"
            except Exception:
                pass
        if not next_agent or next_agent == 'agent_decision':
            return 'agent_decision'
        valid_nodes = list(self.agents.keys()) + ['final_decision', 'agent_decision']
        if next_agent not in valid_nodes:
            self.logger.warning(f"无效 next_agent '{next_agent}'，路由到 agent_decision")
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
                self.logger.warning('工作流步骤过多，强制进入最终评估阶段以防止死循环')
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
            prompt = f"作为地球化学数据分析多智能体系统的决策中心，基于当前状态和任务计划，决定下一步应该调用哪个智能体或结束工作流。\n\n当前状态：{context['current_state']}\n\n可用智能体及其职责：\n- data_science_expert: 负责数据预处理、清洗、标准化、预测建模\n- geology_analysis: 负责地质分析、异常检测、元素组合识别、空间分析\n- result_output: 负责生成报告、可视化、数据导出\n- final_decision: 结束工作流\n\n任务计划：{context['task_plan']}\n\n重要提示：\n1. 如果需要进行数据预处理、数据清洗、建模等，请返回 data_science_expert\n2. 如果需要进行地质分析、异常检测、元素分析等，请返回 geology_analysis\n3. 如果需要生成报告、可视化等，请返回 result_output\n4. 如果所有任务已完成，请返回 final_decision\n\n请严格只返回以下之一（不要返回其他任何内容）：\ndata_science_expert\ngeology_analysis\nresult_output\nfinal_decision"
            llm = self.llm
            raw_decision = llm.invoke(prompt).content.strip()
            agent_name_mapping = {'data_preprocessing': 'data_science_expert', 'data_science': 'data_science_expert', 'data_processing': 'data_science_expert', 'preprocessing': 'data_science_expert', 'modeling': 'data_science_expert', 'prediction': 'data_science_expert', 'data_science_expert': 'data_science_expert', 'geology': 'geology_analysis', 'geology_expert': 'geology_analysis', 'geological_analysis': 'geology_analysis', 'anomaly_detection': 'geology_analysis', 'element_analysis': 'geology_analysis', 'geology_analysis': 'geology_analysis', 'output': 'result_output', 'report': 'result_output', 'visualization': 'result_output', 'result': 'result_output', 'result_output': 'result_output', 'final': 'final_decision', 'end': 'final_decision', 'finish': 'final_decision', 'final_decision': 'final_decision'}
            decision = agent_name_mapping.get(raw_decision.lower(), raw_decision)
            valid_decisions = ['data_science_expert', 'geology_analysis', 'result_output', 'final_decision']
            if decision not in valid_decisions:
                self.logger.warning(f'LLM返回无效决策: {raw_decision}，使用默认决策逻辑')
                decision = self._default_decision(state)
            hitl_decision, hitl_meta = self._maybe_hitl_override_decision(
                state=state,
                context=context,
                llm=llm,
                suggested_decision=decision,
                valid_decisions=valid_decisions,
            )
            decision_to_use = hitl_decision if hitl_decision in valid_decisions else decision
            self.logger.info(f'决策结果: {decision}')
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
            self.logger.exception(f'智能体决策失败: {str(e)}')
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(f'智能体决策失败: {str(e)}')
            state['next_agent'] = self._default_decision(state)
        return state
    def _decide_next_step(self, state: dict) -> str:
        return state.get('next_agent', 'final_decision')
    def _default_decision(self, state: dict) -> str:
        history = state.get('processing_history', [])
        has_data_science = any(('DataScienceExpertAgent' in item or '数据预处理' in item or '数据科学' in item for item in history))
        has_geology = any(('GeologyExpertAgent' in item or '地质分析' in item or 'Geology' in item for item in history))
        has_result_output = any(('ResultOutputAgent' in item or '结果输出' in item for item in history))
        self.logger.debug(f'默认决策检查 - 数据科学: {has_data_science}, 地质分析: {has_geology}, 结果输出: {has_result_output}')
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
            state.setdefault('processing_history', []).append('GeologyExpertAgent: 已按消融配置跳过')
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
            task = '评估以下地球化学数据分析工作流的结果，判断是否成功完成任务，指出可能的改进方向。\n\n结果摘要：\n'
            summary = {'处理历史': state.get('processing_history'), '错误信息': state.get('errors'), '分析结果': {k: v.keys() if isinstance(v, dict) else str(v) for k, v in state.get('analysis_results', {}).items()}}
            llm = self.llm
            evaluation = llm.invoke(task + str(summary)).content
            state['final_evaluation'] = evaluation
            state['processing_history'].append('工作流完成：结果评估完成')
        except Exception as e:
            state['errors'].append(f'最终决策失败: {str(e)}')
            state['final_evaluation'] = '无法评估工作流结果'
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
        self.logger.info(f'开始运行地球化学工作流 - 数据文件: {data_path}, 目标矿种: {target_deposit_type}')
        self._emit_progress("run_start", "run", 1, "开始运行工作流")
        cfg = config or {}
        if study_area_location is None and isinstance(cfg, dict):
            study_area_location = cfg.get("study_area_location")
        if data is None:
            if not data_path:
                raise ValueError('data_path 为空且未提供 data')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f'数据文件不存在: {data_path}')
        if data is None:
            self.logger.info('开始加载数据文件...')
            try:
                self._emit_progress("data_loading", "run", 3, "正在加载数据")
                data = load_data(data_path)
                self.logger.info(f'数据加载完成 - 数据形状: {data.shape}')
                self.logger.debug(f'数据列信息: {list(data.columns)}')
                self.logger.debug(f"数据类型摘要: 共{len(data.dtypes)}列 (int64: {(data.dtypes == 'int64').sum()}, float64: {(data.dtypes == 'float64').sum()})")
                self._emit_progress("data_loaded", "run", 5, "数据加载完成")
            except Exception as e:
                self.logger.exception(f'数据加载失败: {str(e)}')
                raise
        else:
            self.logger.info(f'使用预加载数据 - 数据形状: {data.shape}')
            self._emit_progress("data_loaded", "run", 5, "数据已就绪")
        self.logger.info('初始化工作流状态...')
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
        self.logger.info('工作流状态初始化完成')
        self.logger.info('开始执行工作流...')
        try:
            result = self.workflow.invoke(initial_state, {'recursion_limit': 100})
            self.logger.info('工作流执行完成')
            if 'analysis_results' in result:
                self.logger.info(f"分析结果统计 - 结果数量: {len(result['analysis_results'])}")
                for key, value in result['analysis_results'].items():
                    self.logger.debug(f"分析结果 '{key}': {type(value).__name__}")
            if 'errors' in result and result['errors']:
                self.logger.warning(f"工作流执行过程中产生 {len(result['errors'])} 个错误")
                for idx, err in enumerate(result['errors'], start=1):
                    self.logger.warning(f"错误[{idx}]: {err}")
        except Exception as e:
            if _is_auth_error(e):
                self.logger.error(f'工作流执行失败: {str(e)}')
            else:
                self.logger.exception(f'工作流执行失败: {str(e)}')
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
            self.logger.warning(f'最终结果JSON回写失败: {e}')
        self.logger.info('工作流运行完成，准备返回结果')
        self._emit_progress("run_end", "run", 100, "工作流完成")
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
            logger.info(f'工作流图已保存到: {output_path}')
        except Exception as e:
            logger.exception(f'可视化工作流失败: {str(e)}')
def create_workflow(output_dir: str='./output', progress_hook: Optional[Callable[[str, str, int, str], None]] = None):
    workflow = GeoChemistryWorkflow(output_dir=output_dir, progress_hook=progress_hook)
    logger.info(f'地球化学工作流实例创建成功，输出目录: {output_dir}')
    return workflow
if __name__ == '__main__':
    workflow = create_workflow()
    workflow.visualize_workflow()
