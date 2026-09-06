import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
from datetime import datetime
from typing import Optional, Callable, Any
try:
    from .utils.data_utils import (
        apply_output_language_env as _apply_output_language_env,
        get_bilingual_text as _get_bilingual_text,
        localize_text as _localize_text,
        resolve_output_language as _resolve_output_language,
        setup_matplotlib_output_style as _setup_matplotlib_output_style,
    )
except Exception:
    try:
        from utils.data_utils import (
            apply_output_language_env as _apply_output_language_env,
            get_bilingual_text as _get_bilingual_text,
            localize_text as _localize_text,
            resolve_output_language as _resolve_output_language,
            setup_matplotlib_output_style as _setup_matplotlib_output_style,
        )
    except Exception:
        def _resolve_output_language(config: Optional[dict[str, Any]] = None) -> str:
            return "en"

        def _apply_output_language_env(config: Optional[dict[str, Any]] = None) -> str:
            return "en"

        def _localize_text(text: Any, *, lang: Optional[str] = None) -> str:
            return str(text)

        def _get_bilingual_text(zh_text: str, en_text: str, *, lang: Optional[str] = None) -> str:
            return str(en_text)

        def _setup_matplotlib_output_style(plt_module: Any = None) -> None:
            return
_load_dotenv: Optional[Callable[..., bool]] = None
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    pass
load_dotenv = _load_dotenv
def _prepend_sys_path(path: str) -> None:
    abs_path = os.path.abspath(path)
    abs_norm = os.path.normcase(abs_path)
    for existing in sys.path:
        try:
            if os.path.normcase(os.path.abspath(existing)) == abs_norm:
                return
        except Exception:
            continue
    sys.path.insert(0, abs_path)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
_prepend_sys_path(parent_dir)
_prepend_sys_path(current_dir)
def _get_app_data_dir(app_name: str = "GAI-MAS") -> str:
    base_dir = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    app_dir = os.path.join(base_dir, app_name)
    os.makedirs(app_dir, exist_ok=True)
    return app_dir
_LOGGING_CONFIGURED = False
_REDACT_PATTERNS = [
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?key|secret|token|password)\b\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._\-]+)"),
    re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    re.compile(r"lsv2_[A-Za-z0-9_\-]{10,}"),
]
def _redact_text(text: object) -> str:
    try:
        s = str(text)
    except Exception:
        return "<unprintable>"
    for pattern in _REDACT_PATTERNS:
        if pattern.pattern.lower().startswith("(?i)\\b(bearer)"):
            s = pattern.sub("<REDACTED>", s)
        else:
            s = pattern.sub(lambda m: f"{m.group(1)}=<REDACTED>" if m.lastindex and m.lastindex >= 2 else "<REDACTED>", s)
    return s
class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            record.msg = _redact_text(msg)
            record.args = ()
        except Exception:
            pass
        return True


def _input_with_log_prefix(logger: logging.Logger, level: str = "INFO") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return input(f"{ts} - {logger.name} - {level} - ")
def _attach_run_log_file(output_dir: str) -> Optional[str]:
    if not output_dir or not isinstance(output_dir, str):
        return None
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    run_log_path = os.path.abspath(os.path.join(reports_dir, "run_log.log"))
    root_logger = logging.getLogger()
    run_log_norm = os.path.normcase(run_log_path)
    for handler in root_logger.handlers:
        try:
            base = getattr(handler, "baseFilename", None)
            if base and os.path.normcase(os.path.abspath(str(base))) == run_log_norm:
                return run_log_path
        except Exception:
            continue
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_handler = logging.FileHandler(run_log_path, encoding="utf-8", mode="w")
    run_handler.setFormatter(formatter)
    run_handler.addFilter(_RedactingFilter())
    root_logger.addHandler(run_handler)
    return run_log_path


def _atomic_write_json(path: str, payload: Any) -> str:
    file_path = os.path.abspath(str(path))
    target_dir = os.path.dirname(file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=target_dir or None, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
        return file_path
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


def _atomic_write_text(path: str, content: str, *, encoding: str = "utf-8") -> str:
    file_path = os.path.abspath(str(path))
    target_dir = os.path.dirname(file_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    suffix = os.path.splitext(file_path)[1] or ".txt"
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=target_dir or None, text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(str(content or ""))
            if not str(content or "").endswith("\n"):
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
        return file_path
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise


_EVAL_METRIC_KEY_ZH: dict[str, str] = {
    "generated_at": "生成时间",
    "schema_version": "结构版本",
    "run": "运行",
    "professional_evaluation": "Professional MAS Evaluation",
    "dimensions_0_1": "Dimension Scores (0-1)",
    "indicators_0_1": "Indicator Scores (0-1)",
    "raw_metrics": "Raw Metrics",
    "professional_scoring_config": "Professional Scoring Config",
    "output_dir": "输出目录",
    "data_path": "数据文件路径",
    "study_area_location": "研究区地点",
    "target_deposit_type": "目标矿种",
    "phase": "阶段",
    "runtime_seconds": "耗时秒数",
    "runtime_human": "耗时",
    "governance_run_id": "治理运行ID",
    "config": "运行配置",
    "interaction_mode": "交互模式",
    "learning_mode": "学习方式",
    "reflection_enabled": "反思开关",
    "reflection_max_rounds": "反思最大轮次",
    "structured_output_enabled": "结构化输出开关",
    "outcome": "结果",
    "task_outcome": "Task Outcome",
    "collaboration": "Collaboration",
    "governance_control": "Governance Control",
    "reliability": "Reliability",
    "efficiency": "Efficiency",
    "tsr": "任务成功率",
    "acr": "产出物完成率",
    "task_success": "Task Success",
    "artifact_completeness": "Artifact Completeness",
    "model_quality": "模型质量",
    "overfit_penalty": "过拟合惩罚",
    "label_profile": "标签概况",
    "label_col": "标签列",
    "known_count": "已知标签样本数",
    "missing_count": "缺失标签样本数",
    "pos_count": "正例数",
    "neg_count": "负例数",
    "pos_rate": "正例比例",
    "calibration": "校准指标",
    "brier": "Brier分数",
    "log_loss": "对数损失",
    "ece_10": "ECE(10箱)",
    "mce_10": "MCE(10箱)",
    "ranking": "排序指标",
    "precision_at_1pct": "Top1%精度",
    "recall_at_1pct": "Top1%召回",
    "precision_at_poscount": "Top正例数精度",
    "recall_at_poscount": "Top正例数召回",
    "process": "过程",
    "checks_pass_rate": "检查通过率",
    "verification_pass": "Verification Pass",
    "checks": "检查汇总",
    "error_fail": "错误失败数",
    "error_total": "错误总数",
    "warn_fail": "警告失败数",
    "warn_total": "警告总数",
    "rework_rate": "返工率",
    "budget_utilization": "预算利用率",
    "decision_stability": "决策稳定性",
    "decision_consistency": "Decision Consistency",
    "hitl_intervention_rate": "人工介入率",
    "rework_control": "Rework Control",
    "mechanism": "机制",
    "sofr": "结构化失败率",
    "structured_reliability": "Structured Reliability",
    "repair_resilience": "Repair Resilience",
    "json_repair_success_rate": "JSON修复成功率",
    "reflection_intensity": "反思强度",
    "eval_stats_totals": "统计合计",
    "decide_calls": "决策调用次数",
    "decide_json_calls": "结构化调用次数",
    "reflection_text_rounds": "反思文本轮次",
    "structured_parse_failures": "结构化解析失败数",
    "json_repair_attempts": "JSON修复尝试数",
    "json_repair_successes": "JSON修复成功数",
    "decide_calls_total": "决策调用总数",
    "decide_json_calls_total": "结构化调用总数",
    "reflection_text_rounds_total": "反思文本轮次总数",
    "structured_parse_failures_total": "结构化解析失败总数",
    "json_repair_attempts_total": "JSON修复尝试总数",
    "json_repair_successes_total": "JSON修复成功总数",
    "domain": "领域",
    "high_potential_ratio": "高潜力占比",
    "confidence_level": "置信度等级",
    "geo_pred_consistency": "地质-预测一致性",
    "predictive_quality": "Predictive Quality",
    "calibration_quality": "Calibration Quality",
    "data_profile": "数据概况",
    "rows": "行数",
    "cols": "列数",
    "numeric_cols": "数值列数",
    "missing_cells": "缺失单元格数",
    "duplicate_rows": "重复行数",
    "coord_x": "X坐标列",
    "coord_y": "Y坐标列",
    "agent_eval_stats": "智能体统计",
    "Agent": "智能体",
    "artifacts": "产物",
    "expected": "期望产物",
    "output_paths": "产物路径",
    "comprehensive_report": "综合报告",
    "predictions_data": "预测表",
    "anomaly_analysis": "异常分析",
    "feature_analysis_selection_doc": "特征分析与选择说明",
    "target_element_selection_doc": "目标元素筛选说明",
    "feature_importance": "特征重要性",
    "json_results": "全链路JSON",
    "prediction_map": "预测图",
    "token_usage": "Token使用",
    "agent_coverage": "Agent Coverage",
    "workflow_progression": "Workflow Progression",
    "collaboration_balance": "Collaboration Balance",
    "handoff_cohesion": "Handoff Cohesion",
    "time_budget_ratio": "Time Budget Ratio",
    "step_budget_ratio": "Step Budget Ratio",
    "route_transitions": "Route Transitions",
    "route_revisits": "Route Revisits",
    "active_roles": "Active Roles",
    "Total Tokens": "总Token数",
    "Prompt Tokens": "输入Token数",
    "Completion Tokens": "输出Token数",
    "Total Cost (CNY)": "总成本(CNY)",
    "Successful Requests": "成功请求数",
    "Request History Count": "请求历史条数",
    "score": "评分",
    "overall": "总分(0-100)",
    "subscores_0_1": "子分(0-1)",
    "signals_0_1": "信号得分(0-1)",
    "scoring_config": "评分配置",
    "mechanism_weights": "机制权重",
    "outcome_weights": "结果权重",
    "overall_weights": "总权重",
    "process_weights": "过程权重",
    "targets": "目标阈值",
    "dimension_weights": "Dimension Weights",
    "budget_score": "预算得分",
    "rework_score": "返工得分",
    "structured_output_score": "结构化输出得分",
    "reflection_score": "反思得分",
    "token_efficiency": "Token效率",
    "budget_utilization_bad": "预算利用率差阈值",
    "budget_utilization_good": "预算利用率好阈值",
    "reflection_intensity_good": "反思强度好阈值",
    "rework_rate_good": "返工率好阈值",
    "structured_failure_rate_good": "结构化失败率好阈值",
    "tokens_per_run_good": "单次运行Token目标",
    "ece_good": "ECE Good Threshold",
    "ece_bad": "ECE Bad Threshold",
    "brier_bad": "Brier Bad Threshold",
    "log_loss_bad": "Log Loss Bad Threshold",
}


def _eval_metric_label(key: str, *, include_raw: bool = True) -> str:
    raw = str(key).strip()
    if not raw:
        return raw
    parts = raw.split(".")
    zh = ".".join([_EVAL_METRIC_KEY_ZH.get(p, p) for p in parts])
    if include_raw and zh != raw:
        return f"{zh}({raw})"
    return zh


def _format_eval_metrics_markdown(metrics: dict[str, Any]) -> str:
    def _g(obj: Any, *keys: str, default: Any = None) -> Any:
        cur: Any = obj
        for k in keys:
            if not isinstance(cur, dict):
                return default
            if k not in cur:
                return default
            cur = cur.get(k)
        return cur

    def _fmt(v: Any) -> str:
        if v is None:
            return "NA"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return "NA"
            if abs(v) >= 1000:
                return f"{v:.3g}"
            if abs(v) >= 10:
                return f"{v:.3f}"
            return f"{v:.4f}"
        return str(v)

    def _table(rows: list[tuple[str, Any]]) -> str:
        lines = ["| 项 | 值 |", "|---|---|"]
        for k, v in rows:
            kk = _eval_metric_label(str(k).replace("\n", " ").strip(), include_raw=True)
            vv = _fmt(v).replace("\n", " ").strip()
            lines.append(f"| {kk} | {vv} |")
        return "\n".join(lines) + "\n"

    def _dict_lines(d: Any, *, prefix: str = "") -> list[str]:
        if not isinstance(d, dict) or not d:
            return []
        out: list[str] = []
        for k in sorted([str(x) for x in d.keys()]):
            v = d.get(k)
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            key_display = _eval_metric_label(key, include_raw=True)
            if isinstance(v, dict):
                out.append(f"- {key_display}:")
                out.extend(_dict_lines(v, prefix=key))
            elif isinstance(v, list):
                out.append(f"- {key_display}: [{len(v)}]")
            else:
                out.append(f"- {key_display}: {_fmt(v)}")
        return out

    run_obj = metrics.get("run")
    run = run_obj if isinstance(run_obj, dict) else {}
    outcome_obj = metrics.get("outcome")
    outcome = outcome_obj if isinstance(outcome_obj, dict) else {}
    process_obj = metrics.get("process")
    process = process_obj if isinstance(process_obj, dict) else {}
    mechanism_obj = metrics.get("mechanism")
    mechanism = mechanism_obj if isinstance(mechanism_obj, dict) else {}
    domain_obj = metrics.get("domain")
    domain = domain_obj if isinstance(domain_obj, dict) else {}
    data_profile_obj = metrics.get("data_profile")
    data_profile = data_profile_obj if isinstance(data_profile_obj, dict) else {}
    artifacts_obj = metrics.get("artifacts")
    artifacts = artifacts_obj if isinstance(artifacts_obj, dict) else {}
    agent_stats_obj = metrics.get("agent_eval_stats")
    agent_stats = agent_stats_obj if isinstance(agent_stats_obj, dict) else {}
    token_usage_obj = metrics.get("token_usage")
    token_usage = token_usage_obj if isinstance(token_usage_obj, dict) else {}
    professional_obj = metrics.get("professional_evaluation")
    professional = professional_obj if isinstance(professional_obj, dict) else {}
    score_obj = metrics.get("score")
    score = score_obj if isinstance(score_obj, dict) else {}

    lines: list[str] = []
    lines.append("# Multi-Agent System Evaluation Summary")
    lines.append("")
    lines.append(_table([("generated_at", metrics.get("generated_at")), ("schema_version", metrics.get("schema_version"))]).rstrip())
    lines.append("")

    lines.append("## Run")
    lines.append("")
    lines.append(
        _table(
            [
                ("output_dir", run.get("output_dir")),
                ("data_path", run.get("data_path")),
                ("target_deposit_type", run.get("target_deposit_type")),
                ("phase", run.get("phase")),
                ("runtime_seconds", run.get("runtime_seconds")),
                ("runtime_human", run.get("runtime_human")),
                ("governance_run_id", run.get("governance_run_id")),
            ]
        ).rstrip()
    )
    lines.append("")
    run_config_obj = run.get("config")
    run_config = run_config_obj if isinstance(run_config_obj, dict) else {}
    if run_config:
        lines.append("### Run Config")
        lines.append("")
        lines.extend(_dict_lines(run_config))
        lines.append("")

    if professional:
        lines.append("## Professional MAS Evaluation")
        lines.append("")
        dims_obj = professional.get("dimensions_0_1")
        dims = dims_obj if isinstance(dims_obj, dict) else {}
        lines.append(
            _table(
                [
                    ("overall", professional.get("overall")),
                    ("task_outcome", dims.get("task_outcome")),
                    ("collaboration", dims.get("collaboration")),
                    ("governance_control", dims.get("governance_control")),
                    ("reliability", dims.get("reliability")),
                    ("efficiency", dims.get("efficiency")),
                ]
            ).rstrip()
        )
        lines.append("")
        indicators_obj = professional.get("indicators_0_1")
        indicators = indicators_obj if isinstance(indicators_obj, dict) else {}
        if indicators:
            lines.append("### Indicator Scores")
            lines.append("")
            for k in sorted([str(x) for x in indicators.keys()]):
                lines.append(f"- {_eval_metric_label(k, include_raw=True)}: {_fmt(indicators.get(k))}")
            lines.append("")
        raw_metrics_obj = professional.get("raw_metrics")
        professional_raw = raw_metrics_obj if isinstance(raw_metrics_obj, dict) else {}
        if professional_raw:
            lines.append("### Raw Metrics")
            lines.append("")
            for k in sorted([str(x) for x in professional_raw.keys()]):
                lines.append(f"- {_eval_metric_label(k, include_raw=True)}: {_fmt(professional_raw.get(k))}")
            lines.append("")
        professional_cfg_obj = professional.get("professional_scoring_config")
        professional_cfg = professional_cfg_obj if isinstance(professional_cfg_obj, dict) else {}
        if professional_cfg:
            lines.append("### Professional Scoring Config")
            lines.append("")
            lines.extend(_dict_lines(professional_cfg))
            lines.append("")

    lines.append("## Outcome")
    lines.append("")
    lines.append(_table([("tsr", outcome.get("tsr")), ("acr", outcome.get("acr")), ("model_quality", outcome.get("model_quality")), ("overfit_penalty", outcome.get("overfit_penalty"))]).rstrip())
    lines.append("")
    lines.append("### Label Profile")
    lines.append("")
    lp_obj = outcome.get("label_profile")
    lp = lp_obj if isinstance(lp_obj, dict) else {}
    lines.append(
        _table(
            [
                ("label_col", lp.get("label_col")),
                ("known_count", lp.get("known_count")),
                ("missing_count", lp.get("missing_count")),
                ("pos_count", lp.get("pos_count")),
                ("neg_count", lp.get("neg_count")),
                ("pos_rate", lp.get("pos_rate")),
            ]
        ).rstrip()
    )
    lines.append("")
    lines.append("### Calibration")
    lines.append("")
    cal_obj = outcome.get("calibration")
    cal = cal_obj if isinstance(cal_obj, dict) else {}
    lines.append(_table([("brier", cal.get("brier")), ("log_loss", cal.get("log_loss")), ("ece_10", cal.get("ece_10")), ("mce_10", cal.get("mce_10"))]).rstrip())
    lines.append("")
    lines.append("### Ranking")
    lines.append("")
    rk_obj = outcome.get("ranking")
    rk = rk_obj if isinstance(rk_obj, dict) else {}
    lines.append(
        _table(
            [
                ("precision_at_1pct", rk.get("precision_at_1pct")),
                ("recall_at_1pct", rk.get("recall_at_1pct")),
                ("precision_at_poscount", rk.get("precision_at_poscount")),
                ("recall_at_poscount", rk.get("recall_at_poscount")),
            ]
        ).rstrip()
    )
    lines.append("")

    lines.append("## Process")
    lines.append("")
    checks_obj = process.get("checks")
    checks = checks_obj if isinstance(checks_obj, dict) else {}
    lines.append(_table([("checks_pass_rate", process.get("checks_pass_rate")), ("error_fail", checks.get("error_fail")), ("error_total", checks.get("error_total")), ("warn_fail", checks.get("warn_fail")), ("warn_total", checks.get("warn_total")), ("rework_rate", process.get("rework_rate")), ("budget_utilization", process.get("budget_utilization")), ("decision_stability", process.get("decision_stability")), ("hitl_intervention_rate", process.get("hitl_intervention_rate"))]).rstrip())
    lines.append("")

    lines.append("## Mechanism")
    lines.append("")
    totals_obj = mechanism.get("eval_stats_totals")
    totals = totals_obj if isinstance(totals_obj, dict) else {}
    lines.append(
        _table(
            [
                ("sofr", mechanism.get("sofr")),
                ("json_repair_success_rate", mechanism.get("json_repair_success_rate")),
                ("reflection_intensity", mechanism.get("reflection_intensity")),
                ("decide_calls_total", totals.get("decide_calls")),
                ("decide_json_calls_total", totals.get("decide_json_calls")),
                ("reflection_text_rounds_total", totals.get("reflection_text_rounds")),
                ("structured_parse_failures_total", totals.get("structured_parse_failures")),
                ("json_repair_attempts_total", totals.get("json_repair_attempts")),
                ("json_repair_successes_total", totals.get("json_repair_successes")),
            ]
        ).rstrip()
    )
    lines.append("")

    lines.append("## Domain")
    lines.append("")
    lines.append(_table([("high_potential_ratio", domain.get("high_potential_ratio")), ("confidence_level", domain.get("confidence_level"))]).rstrip())
    lines.append("")

    lines.append("## Data Profile")
    lines.append("")
    lines.append(_table([("rows", data_profile.get("rows")), ("cols", data_profile.get("cols")), ("numeric_cols", data_profile.get("numeric_cols")), ("missing_cells", data_profile.get("missing_cells")), ("duplicate_rows", data_profile.get("duplicate_rows")), ("coord_x", data_profile.get("coord_x")), ("coord_y", data_profile.get("coord_y"))]).rstrip())
    lines.append("")

    lines.append("## Agent Stats")
    lines.append("")
    lines.append(
        "| "
        + " | ".join(
            [
                _eval_metric_label("Agent", include_raw=True),
                _eval_metric_label("decide_calls", include_raw=True),
                _eval_metric_label("decide_json_calls", include_raw=True),
                _eval_metric_label("reflection_text_rounds", include_raw=True),
                _eval_metric_label("structured_parse_failures", include_raw=True),
                _eval_metric_label("json_repair_attempts", include_raw=True),
                _eval_metric_label("json_repair_successes", include_raw=True),
            ]
        )
        + " |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for agent_name in sorted([str(k) for k in agent_stats.keys()]):
        s = agent_stats.get(agent_name)
        if not isinstance(s, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    agent_name,
                    _fmt(s.get("decide_calls")),
                    _fmt(s.get("decide_json_calls")),
                    _fmt(s.get("reflection_text_rounds")),
                    _fmt(s.get("structured_parse_failures")),
                    _fmt(s.get("json_repair_attempts")),
                    _fmt(s.get("json_repair_successes")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    expected_obj = _g(artifacts, "expected", default=None)
    if isinstance(expected_obj, list) and expected_obj:
        lines.append("### Expected")
        lines.append("")
        lines.append(", ".join([str(x) for x in expected_obj if x is not None]))
        lines.append("")
    output_paths = _g(artifacts, "output_paths", default={})
    if isinstance(output_paths, dict) and output_paths:
        for k in sorted([str(x) for x in output_paths.keys()]):
            v = output_paths.get(k)
            if v is None:
                continue
            lines.append(f"- {_eval_metric_label(k, include_raw=True)}: {_fmt(v)}")
    else:
        lines.append("NA")
    lines.append("")

    if score:
        lines.append("## Score")
        lines.append("")
        subs_obj = score.get("subscores_0_1")
        subs = subs_obj if isinstance(subs_obj, dict) else {}
        lines.append(_table([("overall_0_100", score.get("overall_0_100")), ("outcome", subs.get("outcome")), ("process", subs.get("process")), ("mechanism", subs.get("mechanism"))]).rstrip())
        lines.append("")
        signals_obj = score.get("signals_0_1")
        signals = signals_obj if isinstance(signals_obj, dict) else {}
        if signals:
            lines.append("### Signals")
            lines.append("")
            for k in sorted([str(x) for x in signals.keys()]):
                lines.append(f"- {_eval_metric_label(k, include_raw=True)}: {_fmt(signals.get(k))}")
            lines.append("")
        scoring_cfg_obj = score.get("scoring_config")
        scoring_cfg = scoring_cfg_obj if isinstance(scoring_cfg_obj, dict) else {}
        if scoring_cfg:
            lines.append("### Scoring Config")
            lines.append("")
            lines.extend(_dict_lines(scoring_cfg))
            lines.append("")

    if token_usage:
        lines.append("## Token Usage")
        lines.append("")
        lines.append(
            _table(
                [
                    ("Total Tokens", token_usage.get("Total Tokens")),
                    ("Prompt Tokens", token_usage.get("Prompt Tokens")),
                    ("Completion Tokens", token_usage.get("Completion Tokens")),
                    ("Total Cost (CNY)", token_usage.get("Total Cost (CNY)")),
                    ("Successful Requests", token_usage.get("Successful Requests")),
                    ("Request History Count", len(request_history) if isinstance((request_history := token_usage.get("Request History")), list) else 0),
                ]
            ).rstrip()
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _generate_eval_metrics_figures(metrics: dict[str, Any], *, output_dir: str) -> dict[str, str]:
    config_obj = metrics.get("config")
    metrics_config = config_obj if isinstance(config_obj, dict) else {}
    try:
        lang = str(metrics_config.get("output_language") or "en").strip().lower() or "en"
    except Exception:
        lang = "en"

    def _as_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            try:
                x = float(v)
            except Exception:
                return None
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        try:
            s = str(v).strip()
        except Exception:
            return None
        if not s or s.upper() == "NA":
            return None
        try:
            x = float(s)
        except Exception:
            return None
        if math.isnan(x) or math.isinf(x):
            return None
        return x

    fig_dir = os.path.join(str(output_dir), "reports", "eval_metrics_figures")
    try:
        os.makedirs(fig_dir, exist_ok=True)
    except Exception:
        return {}

    try:
        import matplotlib

        try:
            matplotlib.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
    except Exception as e:
        try:
            logger.warning(f"评测指标可视化依赖导入失败: {e}")
        except Exception:
            pass
        return {}

    try:
        import seaborn as sns

        try:
            sns.set_style("whitegrid")
        except Exception:
            pass
    except Exception:
        sns = None

    _setup_matplotlib_output_style(plt)
    try:
        # Ensure per-figure language controls the font stack used by eval charts.
        # The global output language may still be "en" when we intentionally render
        # a localized Chinese figure, which would otherwise fall back to Times New Roman
        # and display Chinese glyphs as boxes.
        if lang == "zh":
            plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "sans-serif"]
            plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans", "sans-serif"]
        else:
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams["font.serif"] = ["Times New Roman"]
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Times New Roman", "Arial", "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    out: dict[str, str] = {}

    def label_zh(k: Any) -> str:
        return _localize_text(_eval_metric_label(str(k), include_raw=False), lang=lang)

    def label_signal(k: Any) -> str:
        raw = str(k).strip()
        if str(lang).lower() == "en":
            return raw.replace("_", " ").title() if raw else raw
        return label_zh(raw)

    def label_agent(k: Any) -> str:
        raw = str(k).strip()
        agent_labels = {
            "DataScienceExpertAgent": ("数据科学专家智能体", "DataScienceExpertAgent"),
            "GeologyExpertAgent": ("地质专家智能体", "GeologyExpertAgent"),
            "ResultOutputAgent": ("综合报告智能体", "ResultOutputAgent"),
            "data_science_expert": ("数据科学专家智能体", "data_science_expert"),
            "geology_analysis": ("地质专家智能体", "geology_analysis"),
            "result_output": ("综合报告智能体", "result_output"),
        }
        if raw in agent_labels:
            zh_text, en_text = agent_labels[raw]
            return _get_bilingual_text(zh_text, en_text, lang=lang)
        return raw

    def label_overview(k: Any) -> str:
        raw = str(k).strip()
        overview_labels = {
            "overall": ("总分", "Overall Score"),
            "task_outcome": ("任务结果", "Task Outcome"),
            "collaboration": ("协作交接", "Collaboration"),
            "governance_control": ("治理控制", "Governance Control"),
            "reliability": ("可靠性", "Reliability"),
            "efficiency": ("效率", "Efficiency"),
        }
        if raw in overview_labels:
            zh_text, en_text = overview_labels[raw]
            return _get_bilingual_text(zh_text, en_text, lang=lang)
        return label_zh(raw)

    def _save(fig, filename: str) -> Optional[str]:
        if not filename:
            return None
        path = os.path.abspath(os.path.join(fig_dir, filename))
        try:
            fig.savefig(path, dpi=220, bbox_inches="tight")
            return path
        except Exception as e:
            try:
                logger.warning(f"{_localize_text('评测指标图保存失败', lang=lang)}({filename}): {e}")
            except Exception:
                pass
            return None
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

    score_obj = metrics.get("score")
    score = score_obj if isinstance(score_obj, dict) else {}
    subs_obj = score.get("subscores_0_1")
    subs = subs_obj if isinstance(subs_obj, dict) else {}
    signals_obj = score.get("signals_0_1")
    signals = signals_obj if isinstance(signals_obj, dict) else {}
    professional_obj = metrics.get("professional_evaluation")
    professional = professional_obj if isinstance(professional_obj, dict) else {}

    try:
        professional_dims_obj = professional.get("dimensions_0_1")
        professional_dims = professional_dims_obj if isinstance(professional_dims_obj, dict) else {}
        professional_items: list[tuple[str, Optional[float]]] = [
            (label_overview("overall"), _as_float(professional.get("overall"))),
            (label_overview("task_outcome"), (_as_float(professional_dims.get("task_outcome")) * 100.0) if _as_float(professional_dims.get("task_outcome")) is not None else None),
            (label_overview("collaboration"), (_as_float(professional_dims.get("collaboration")) * 100.0) if _as_float(professional_dims.get("collaboration")) is not None else None),
            (label_overview("governance_control"), (_as_float(professional_dims.get("governance_control")) * 100.0) if _as_float(professional_dims.get("governance_control")) is not None else None),
            (label_overview("reliability"), (_as_float(professional_dims.get("reliability")) * 100.0) if _as_float(professional_dims.get("reliability")) is not None else None),
            (label_overview("efficiency"), (_as_float(professional_dims.get("efficiency")) * 100.0) if _as_float(professional_dims.get("efficiency")) is not None else None),
        ]
        if any((v is not None for _, v in professional_items)):
            items = professional_items
        else:
            overall = _as_float(score.get("overall_0_100"))
            outcome = _as_float(subs.get("outcome"))
            process = _as_float(subs.get("process"))
            mechanism = _as_float(subs.get("mechanism"))
            items = [
                (_localize_text("总分(0-100)", lang=lang), overall),
                (_localize_text("结果(×100)", lang=lang), (outcome * 100.0) if outcome is not None else None),
                (_localize_text("过程(×100)", lang=lang), (process * 100.0) if process is not None else None),
                (_localize_text("机制(×100)", lang=lang), (mechanism * 100.0) if mechanism is not None else None),
            ]
        labels = [k for k, v in items if v is not None]
        values = [float(v) for _, v in items if v is not None]
        if labels and values:
            fig_w = max(10.5, 1.45 * float(len(labels)))
            fig = plt.figure(figsize=(fig_w, 4.8))
            ax = fig.add_subplot(111)
            palette = ["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#54a24b", "#b279a2", "#ff9da6"]
            ax.bar(labels, values, color=palette[: len(values)])
            ax.set_ylim(0, 100)
            ax.set_ylabel(_localize_text("得分", lang=lang), fontsize=16)
            ax.set_title(_localize_text("评测得分概览", lang=lang), pad=18, fontsize=16)
            ax.tick_params(axis="x", rotation=20, labelsize=16)
            ax.tick_params(axis="y", labelsize=16)
            for i, v in enumerate(values):
                ax.text(i, min(v + 2.0, 99.0), f"{v:.2f}", ha="center", va="bottom", fontsize=14)
            fig.tight_layout()
            p = _save(fig, "score_overview.png")
            if p:
                out["eval_metrics_fig_score_overview"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    try:
        if isinstance(signals, dict) and signals:
            pairs: list[tuple[str, float]] = []
            for k, v in signals.items():
                x = _as_float(v)
                if x is None:
                    continue
                pairs.append((label_signal(k), float(x)))
            pairs.sort(key=lambda kv: kv[1])
            if pairs:
                labels = [k for k, _ in pairs]
                values = [v for _, v in pairs]
                fig_h = max(3.2, 0.33 * float(len(pairs)) + 1.5)
                fig = plt.figure(figsize=(10, fig_h))
                ax = fig.add_subplot(111)
                ax.barh(labels, values, color="#54a24b")
                ax.set_xlim(0, 1.0)
                ax.set_xlabel(_localize_text("得分(0-1)", lang=lang))
                ax.set_title(_localize_text("得分信号(0-1)", lang=lang))
                for i, v in enumerate(values):
                    ax.text(min(v + 0.02, 0.98), i, f"{v:.3f}", va="center", fontsize=9)
                p = _save(fig, "score_signals.png")
                if p:
                    out["eval_metrics_fig_score_signals"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    outcome_obj = metrics.get("outcome")
    outcome_m = outcome_obj if isinstance(outcome_obj, dict) else {}
    try:
        lp_obj = outcome_m.get("label_profile")
        lp = lp_obj if isinstance(lp_obj, dict) else {}
        pos = _as_float(lp.get("pos_count"))
        neg = _as_float(lp.get("neg_count"))
        if pos is not None and neg is not None and (pos + neg) > 0:
            fig = plt.figure(figsize=(8.5, 4.3))
            ax = fig.add_subplot(111)
            ax.bar(
                [
                    _get_bilingual_text("正例", "Positive", lang=lang),
                    _get_bilingual_text("负例", "Negative", lang=lang),
                ],
                [pos, neg],
                color=["#e45756", "#4c78a8"],
            )
            try:
                ax.set_yscale("log")
            except Exception:
                pass
            pos_rate = _as_float(lp.get("pos_rate"))
            title = _localize_text("标签分布", lang=lang)
            if pos_rate is None:
                ratio_name = _get_bilingual_text("正例比例", "Positive Rate", lang=lang)
                title += f" ({ratio_name}={pos / (pos + neg):.4f})"
            else:
                ratio_name = _get_bilingual_text("正例比例", "Positive Rate", lang=lang)
                title += f" ({ratio_name}={pos_rate:.4f})"
            ax.set_title(title)
            ax.set_ylabel(_localize_text("样本数（对数刻度）", lang=lang))
            p = _save(fig, "label_profile.png")
            if p:
                out["eval_metrics_fig_label_profile"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    try:
        cal_obj = outcome_m.get("calibration")
        cal = cal_obj if isinstance(cal_obj, dict) else {}
        rk_obj = outcome_m.get("ranking")
        rk = rk_obj if isinstance(rk_obj, dict) else {}

        cal_items = [
            (label_zh("brier"), _as_float(cal.get("brier"))),
            (label_zh("log_loss"), _as_float(cal.get("log_loss"))),
            (label_zh("ece_10"), _as_float(cal.get("ece_10"))),
            (label_zh("mce_10"), _as_float(cal.get("mce_10"))),
        ]
        rk_items = [
            (label_zh("precision_at_1pct"), _as_float(rk.get("precision_at_1pct"))),
            (label_zh("recall_at_1pct"), _as_float(rk.get("recall_at_1pct"))),
            (label_zh("precision_at_poscount"), _as_float(rk.get("precision_at_poscount"))),
            (label_zh("recall_at_poscount"), _as_float(rk.get("recall_at_poscount"))),
        ]
        cal_labels = [k for k, v in cal_items if v is not None]
        cal_vals = [float(v) for _, v in cal_items if v is not None]
        rk_labels = [k for k, v in rk_items if v is not None]
        rk_vals = [float(v) for _, v in rk_items if v is not None]
        if cal_labels or rk_labels:
            fig = plt.figure(figsize=(12, 4.2))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            if cal_labels:
                ax1.bar(cal_labels, cal_vals, color="#72b7b2")
                ax1.set_title(_localize_text("校准指标", lang=lang))
                ax1.tick_params(axis="x", rotation=25)
            else:
                ax1.axis("off")
            if rk_labels:
                ax2.bar(rk_labels, rk_vals, color="#f58518")
                ax2.set_title(_localize_text("排序指标", lang=lang))
                ax2.tick_params(axis="x", rotation=25)
                try:
                    ax2.set_ylim(0, 1.0)
                except Exception:
                    pass
            else:
                ax2.axis("off")
            p = _save(fig, "calibration_ranking.png")
            if p:
                out["eval_metrics_fig_calibration_ranking"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    process_obj = metrics.get("process")
    process_m = process_obj if isinstance(process_obj, dict) else {}
    mechanism_obj = metrics.get("mechanism")
    mechanism_m = mechanism_obj if isinstance(mechanism_obj, dict) else {}
    try:
        proc_items = [
            (label_zh("checks_pass_rate"), _as_float(process_m.get("checks_pass_rate"))),
            (label_zh("rework_rate"), _as_float(process_m.get("rework_rate"))),
            (label_zh("budget_utilization"), _as_float(process_m.get("budget_utilization"))),
            (label_zh("decision_stability"), _as_float(process_m.get("decision_stability"))),
            (label_zh("hitl_intervention_rate"), _as_float(process_m.get("hitl_intervention_rate"))),
        ]
        mech_items = [
            (label_zh("sofr"), _as_float(mechanism_m.get("sofr"))),
            (label_zh("json_repair_success_rate"), _as_float(mechanism_m.get("json_repair_success_rate"))),
            (label_zh("reflection_intensity"), _as_float(mechanism_m.get("reflection_intensity"))),
        ]
        proc_pairs = [(k, float(v)) for k, v in proc_items if v is not None]
        mech_pairs = [(k, float(v)) for k, v in mech_items if v is not None]
        if proc_pairs or mech_pairs:
            fig = plt.figure(figsize=(12, 4.2))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            if proc_pairs:
                labels = [k for k, _ in proc_pairs]
                values = [v for _, v in proc_pairs]
                ax1.bar(labels, values, color="#4c78a8")
                ax1.set_title(_localize_text("过程指标", lang=lang))
                ax1.tick_params(axis="x", rotation=25)
                try:
                    ax1.set_ylim(0, 1.0)
                except Exception:
                    pass
            else:
                ax1.axis("off")
            if mech_pairs:
                labels = [k for k, _ in mech_pairs]
                values = [v for _, v in mech_pairs]
                ax2.bar(labels, values, color="#e45756")
                ax2.set_title(_localize_text("机制指标", lang=lang))
                ax2.tick_params(axis="x", rotation=25)
                try:
                    ax2.set_ylim(0, 1.0)
                except Exception:
                    pass
            else:
                ax2.axis("off")
            p = _save(fig, "process_mechanism.png")
            if p:
                out["eval_metrics_fig_process_mechanism"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    agent_stats_obj = metrics.get("agent_eval_stats")
    agent_stats = agent_stats_obj if isinstance(agent_stats_obj, dict) else {}
    try:
        agents = [str(k) for k in agent_stats.keys()]
        agents.sort()
        if agents:
            series = [
                ("decide_calls", label_zh("decide_calls"), "#4c78a8"),
                ("decide_json_calls", label_zh("decide_json_calls"), "#72b7b2"),
                ("reflection_text_rounds", label_zh("reflection_text_rounds"), "#f58518"),
            ]
            values_by_series: list[list[int]] = []
            for key, _, _ in series:
                vals: list[int] = []
                for a in agents:
                    s = agent_stats.get(a)
                    if not isinstance(s, dict):
                        vals.append(0)
                        continue
                    raw_val = s.get(key)
                    try:
                        vals.append(int(raw_val) if raw_val is not None else 0)
                    except Exception:
                        vals.append(0)
                values_by_series.append(vals)
            if any((sum(vs) > 0 for vs in values_by_series)):
                fig_h = max(3.6, 0.35 * float(len(agents)) + 1.8)
                fig = plt.figure(figsize=(10.5, fig_h))
                ax = fig.add_subplot(111)
                y = list(range(len(agents)))
                left = [0.0 for _ in agents]
                for (key, display, color), vals in zip(series, values_by_series):
                    ax.barh(y, vals, left=left, label=display, color=color)
                    left = [left_val + float(val) for left_val, val in zip(left, vals)]
                ax.set_yticks(y)
                ax.set_yticklabels([label_agent(a) for a in agents])
                ax.set_xlabel(_localize_text("次数", lang=lang))
                ax.set_title(_localize_text("智能体调用统计", lang=lang))
                ax.legend(loc="lower right")
                p = _save(fig, "agent_stats.png")
                if p:
                    out["eval_metrics_fig_agent_stats"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    token_usage_obj = metrics.get("token_usage")
    token_usage = token_usage_obj if isinstance(token_usage_obj, dict) else {}
    try:
        prompt = _as_float(token_usage.get("Prompt Tokens"))
        completion = _as_float(token_usage.get("Completion Tokens"))
        total = _as_float(token_usage.get("Total Tokens"))
        if prompt is None and completion is None and total is None:
            pass
        else:
            p_val = float(prompt) if prompt is not None else 0.0
            c_val = float(completion) if completion is not None else 0.0
            t_val = float(total) if total is not None else (p_val + c_val)
            fig = plt.figure(figsize=(8.8, 4.2))
            ax = fig.add_subplot(111)
            ax.bar(["Token"], [p_val], label=_localize_text("输入(Prompt)", lang=lang), color="#4c78a8")
            ax.bar(["Token"], [c_val], bottom=[p_val], label=_localize_text("输出(Completion)", lang=lang), color="#f58518")
            ax.set_ylabel(_localize_text("Token数", lang=lang))
            token_usage_title = _localize_text("Token使用情况", lang=lang)
            total_name = _get_bilingual_text("总计", "Total", lang=lang)
            ax.set_title(f"{token_usage_title} ({total_name}={int(t_val) if t_val >= 0 else t_val})")
            ax.legend(loc="upper right")
            p = _save(fig, "token_usage.png")
            if p:
                out["eval_metrics_fig_token_usage"] = p
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    return out


def _compute_eval_metrics(
    *,
    result: dict[str, Any],
    output_dir: str,
    data_path: str,
    target_deposit: Optional[str],
    config: dict[str, Any],
    eval_stats: Optional[dict[str, Any]],
) -> dict[str, Any]:
    def _safe_json_object(text: object) -> Optional[dict]:
        if text is None:
            return None
        try:
            s = str(text).strip()
        except Exception:
            return None
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _clamp01(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        try:
            v = float(x)
        except Exception:
            return None
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v

    def _weighted_mean(pairs: list[tuple[Optional[float], float]]) -> Optional[float]:
        num = 0.0
        den = 0.0
        for val, w in pairs:
            if val is None:
                continue
            try:
                w_f = float(w)
            except Exception:
                continue
            if w_f <= 0:
                continue
            num += float(val) * w_f
            den += w_f
        return (num / den) if den > 0 else None

    output_results_obj = result.get("output_results")
    output_paths = output_results_obj if isinstance(output_results_obj, dict) else {}
    expected_artifacts = [
        "comprehensive_report",
        "feature_analysis_selection_doc",
        "target_element_selection_doc",
        "predictions_data",
        "anomaly_analysis",
        "feature_importance",
        "json_results",
    ]
    exists_count = 0
    for k in expected_artifacts:
        p = output_paths.get(k) if isinstance(output_paths, dict) else None
        if isinstance(p, str) and p.strip() and os.path.exists(p):
            exists_count += 1
    acr = exists_count / len(expected_artifacts) if expected_artifacts else None

    checks_obj = result.get("checks")
    checks = checks_obj if isinstance(checks_obj, list) else []
    err_total = 0
    err_fail = 0
    warn_total = 0
    warn_fail = 0
    for c in checks:
        if not isinstance(c, dict):
            continue
        ok = bool(c.get("ok", False))
        sev = str(c.get("severity") or "").strip().lower()
        if sev == "error":
            err_total += 1
            if not ok:
                err_fail += 1
        elif sev == "warn":
            warn_total += 1
            if not ok:
                warn_fail += 1
    total_weight = float(err_total) + 0.3 * float(warn_total)
    fail_weight = float(err_fail) + 0.3 * float(warn_fail)
    cpr = (1.0 - (fail_weight / total_weight)) if total_weight > 0 else None

    phase = str(result.get("current_phase") or "").strip()
    report_path = output_paths.get("comprehensive_report") if isinstance(output_paths, dict) else None
    report_ok = isinstance(report_path, str) and report_path.strip() and os.path.exists(report_path)
    tsr = bool(report_ok and phase not in {"budget_exceeded", "abort"} and err_fail == 0)

    governance_obj = result.get("governance")
    gov = governance_obj if isinstance(governance_obj, dict) else {}
    budgets_obj = gov.get("budgets") if isinstance(gov, dict) else None
    budgets = budgets_obj if isinstance(budgets_obj, dict) else {}
    counters_obj = gov.get("counters") if isinstance(gov, dict) else None
    counters = counters_obj if isinstance(counters_obj, dict) else {}

    node_steps = counters.get("node_steps") if isinstance(counters, dict) else None
    rework_total = counters.get("rework_total") if isinstance(counters, dict) else None
    try:
        node_steps_int = int(node_steps) if node_steps is not None else None
    except Exception:
        node_steps_int = None
    try:
        rework_total_int = int(rework_total) if rework_total is not None else None
    except Exception:
        rework_total_int = None
    rr = (float(rework_total_int) / float(node_steps_int)) if (node_steps_int is not None and node_steps_int > 0 and rework_total_int is not None) else None

    max_node_steps = budgets.get("max_node_steps") if isinstance(budgets, dict) else None
    max_seconds = budgets.get("max_seconds") if isinstance(budgets, dict) else None
    try:
        max_node_steps_int = int(max_node_steps) if max_node_steps is not None else None
    except Exception:
        max_node_steps_int = None
    try:
        max_seconds_int = int(max_seconds) if max_seconds is not None else None
    except Exception:
        max_seconds_int = None
    runtime_seconds = result.get("workflow_runtime_seconds")
    try:
        runtime_seconds_f = float(runtime_seconds) if runtime_seconds is not None else None
    except Exception:
        runtime_seconds_f = None
    bu_steps = (float(node_steps_int) / float(max_node_steps_int)) if (node_steps_int is not None and max_node_steps_int is not None and max_node_steps_int > 0) else None
    bu_time = (float(runtime_seconds_f) / float(max_seconds_int)) if (runtime_seconds_f is not None and max_seconds_int is not None and max_seconds_int > 0) else None
    if bu_steps is None and bu_time is None:
        bu = None
    else:
        candidates = [x for x in [bu_steps, bu_time] if isinstance(x, (int, float))]
        bu = max(candidates) if candidates else None

    decision_history_obj = result.get("decision_history")
    decision_history = decision_history_obj if isinstance(decision_history_obj, list) else []
    mapped: list[str] = []
    hitl_count = 0
    for d in decision_history:
        if not isinstance(d, dict):
            continue
        m = str(d.get("mapped_decision") or "").strip()
        if m:
            mapped.append(m)
        if str(d.get("raw_decision") or "").strip().lower() == "hitl":
            hitl_count += 1
    switches = 0
    for i in range(1, len(mapped)):
        if mapped[i] != mapped[i - 1]:
            switches += 1
    ds = (1.0 - (float(switches) / float(max(len(mapped) - 1, 1)))) if mapped else None
    hir = (float(hitl_count) / float(len(decision_history))) if decision_history else None

    routed_roles = [x for x in mapped if x in {"data_science_expert", "geology_analysis", "result_output"}]
    dedup_roles: list[str] = []
    for role in routed_roles:
        if not dedup_roles or dedup_roles[-1] != role:
            dedup_roles.append(role)
    active_roles = len(set(routed_roles))
    agent_coverage = (float(active_roles) / 3.0) if routed_roles else 0.0
    canonical_route = ["data_science_expert", "geology_analysis", "result_output"]
    progression_hits = 0
    for role in dedup_roles:
        if progression_hits < len(canonical_route) and role == canonical_route[progression_hits]:
            progression_hits += 1
    workflow_progression = float(progression_hits) / float(len(canonical_route))
    route_transitions = max(len(dedup_roles) - 1, 0)
    route_revisits = max(len(dedup_roles) - len(set(dedup_roles)), 0)
    handoff_cohesion = None
    if dedup_roles:
        handoff_cohesion = 1.0 - (float(route_revisits) / float(max(len(dedup_roles) - 1, 1)))
        handoff_cohesion = _clamp01(handoff_cohesion)
    collaboration_balance = None
    if routed_roles:
        role_counts = [routed_roles.count(role_name) for role_name in canonical_route]
        total_role_calls = sum(role_counts)
        if total_role_calls > 0:
            entropy = 0.0
            for count in role_counts:
                if count <= 0:
                    continue
                p = float(count) / float(total_role_calls)
                entropy -= p * math.log(p)
            collaboration_balance = _clamp01(entropy / math.log(float(len(canonical_route))))

    totals: dict[str, int] = {}
    agent_metrics: dict[str, dict[str, int]] = {}
    if isinstance(eval_stats, dict):
        for agent_name, stats in eval_stats.items():
            if not isinstance(stats, dict):
                continue
            agent_totals = {k: int(v) for k, v in stats.items() if isinstance(k, str) and isinstance(v, int)}
            agent_metrics[str(agent_name)] = agent_totals
            for k, v in agent_totals.items():
                totals[k] = int(totals.get(k, 0) or 0) + int(v)

    decide_calls = int(totals.get("decide_calls", 0) or 0)
    decide_json_calls = int(totals.get("decide_json_calls", 0) or 0)
    reflection_rounds = int(totals.get("reflection_text_rounds", 0) or 0)
    json_repair_attempts = int(totals.get("json_repair_attempts", 0) or 0)
    json_repair_successes = int(totals.get("json_repair_successes", 0) or 0)
    structured_parse_failures = int(totals.get("structured_parse_failures", 0) or 0)

    sofr = (float(structured_parse_failures) / float(decide_json_calls)) if decide_json_calls > 0 else None
    jr = (float(json_repair_successes) / float(json_repair_attempts)) if json_repair_attempts > 0 else None
    ri = (float(reflection_rounds) / float(decide_calls)) if decide_calls > 0 else None
    repair_resilience = None
    if json_repair_attempts > 0:
        repair_resilience = jr
    elif decide_json_calls > 0:
        repair_resilience = 1.0 if structured_parse_failures == 0 else 0.0

    prediction_bundle: dict[str, Any] = {}
    prediction_results_obj = result.get("prediction_results")
    prediction_model_final_obj = result.get("prediction_model_final")
    prediction_model_obj = result.get("prediction_model")
    if isinstance(prediction_results_obj, dict):
        prediction_bundle = prediction_results_obj
    elif isinstance(prediction_model_final_obj, dict):
        prediction_bundle = prediction_model_final_obj
    elif isinstance(prediction_model_obj, dict):
        prediction_bundle = prediction_model_obj

    model_results_obj = prediction_bundle.get("model_results")
    model_results = model_results_obj if isinstance(model_results_obj, dict) else {}
    best_model_name = model_results.get("best_model_name")
    best_model = model_results.get(str(best_model_name)) if best_model_name is not None else None
    best_test_metrics_obj = best_model.get("test_metrics") if isinstance(best_model, dict) else None
    best_test_metrics = best_test_metrics_obj if isinstance(best_test_metrics_obj, dict) else {}
    fit_diag_obj = best_model.get("fit_diagnosis") if isinstance(best_model, dict) else None
    fit_diag = fit_diag_obj if isinstance(fit_diag_obj, dict) else {}
    fit_status = str(fit_diag.get("status") or "").strip().lower()
    auc_gap = fit_diag.get("auc_gap")
    try:
        auc_gap_f = float(auc_gap) if auc_gap is not None else None
    except Exception:
        auc_gap_f = None

    def _metric_float(key: str) -> Optional[float]:
        v = best_test_metrics.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    roc_auc = _metric_float("roc_auc")
    pr_auc = _metric_float("pr_auc")
    balanced_acc = _metric_float("balanced_accuracy")
    f1_pos = _metric_float("f1_pos")
    mq_parts: list[tuple[str, Optional[float], float]] = [
        ("roc_auc", roc_auc, 0.25),
        ("pr_auc", pr_auc, 0.25),
        ("balanced_accuracy", balanced_acc, 0.25),
        ("f1_pos", f1_pos, 0.25),
    ]
    w_sum = 0.0
    s_sum = 0.0
    for _name, metric_value, metric_weight in mq_parts:
        if metric_value is None:
            continue
        w_sum += float(metric_weight)
        s_sum += float(metric_weight) * float(metric_value)
    mq = (s_sum / w_sum) if w_sum > 0 else None

    ofp = None
    if fit_status == "overfit" and auc_gap_f is not None:
        raw = (float(auc_gap_f) - 0.08) / 0.12
        if raw < 0:
            raw = 0.0
        if raw > 1:
            raw = 1.0
        ofp = float(raw)

    predictions_obj = prediction_bundle.get("predictions")
    predictions = predictions_obj if isinstance(predictions_obj, dict) else {}
    high_potential_count = predictions.get("high_potential_count")
    total_samples = predictions.get("total_samples")
    try:
        high_potential_count_i = int(high_potential_count) if high_potential_count is not None else None
    except Exception:
        high_potential_count_i = None
    try:
        total_samples_i = int(total_samples) if total_samples is not None else None
    except Exception:
        total_samples_i = None
    if total_samples_i is None or total_samples_i == 0:
        preprocessing_bundle: dict[str, Any] = {}
        preprocessing_results_obj = result.get("preprocessing_results")
        data_preprocessing_obj = result.get("data_preprocessing")
        if isinstance(preprocessing_results_obj, dict):
            preprocessing_bundle = preprocessing_results_obj
        elif isinstance(data_preprocessing_obj, dict):
            preprocessing_bundle = data_preprocessing_obj

        basic_stats_obj = preprocessing_bundle.get("basic_stats")
        basic_stats = basic_stats_obj if isinstance(basic_stats_obj, dict) else {}
        try:
            total_samples_i = int(basic_stats.get("total_samples", 0) or 0)
        except Exception:
            total_samples_i = 0
    high_potential_ratio = (float(high_potential_count_i) / float(total_samples_i)) if (high_potential_count_i is not None and total_samples_i and total_samples_i > 0) else None

    geology_expert_results_obj = result.get("geology_expert_results")
    geology_results = geology_expert_results_obj if isinstance(geology_expert_results_obj, dict) else {}
    if not geology_results:
        analysis_results_obj = result.get("analysis_results")
        analysis_results = analysis_results_obj if isinstance(analysis_results_obj, dict) else {}
        geology_obj = analysis_results.get("geology") if isinstance(analysis_results, dict) else None
        geology_results = geology_obj if isinstance(geology_obj, dict) else {}
    final_interp_obj = geology_results.get("final_interpretation") if isinstance(geology_results, dict) else None
    final_interp = final_interp_obj if isinstance(final_interp_obj, dict) else {}
    conf_level = str(final_interp.get("confidence_level") or "").strip()

    data_profile = None
    label_profile = None
    calibration = None
    ranking = None
    try:
        if callable(load_data) and isinstance(data_path, str) and data_path.strip() and os.path.exists(data_path):
            import numpy as np
            import pandas as pd

            df = load_data(data_path)
            if isinstance(df, pd.DataFrame) and not df.empty:
                n_rows = int(df.shape[0])
                n_cols = int(df.shape[1])
                missing_cells = int(df.isna().sum().sum())
                dup_rows = int(df.duplicated().sum())
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

                coord_x = None
                coord_y = None
                try:
                    from utils.data_utils import detect_coordinate_columns as _detect_coordinate_columns

                    coord_x, coord_y = _detect_coordinate_columns(df)
                except Exception:
                    coord_x, coord_y = (None, None)

                data_profile = {
                    "rows": n_rows,
                    "cols": n_cols,
                    "numeric_cols": int(len(numeric_cols)),
                    "missing_cells": missing_cells,
                    "duplicate_rows": dup_rows,
                    "coord_x": coord_x,
                    "coord_y": coord_y,
                }

                if "Ore" in df.columns:
                    ore = pd.to_numeric(df["Ore"], errors="coerce")
                    known_mask = ore.notna()
                    known_count = int(known_mask.sum())
                    pos_count = int((ore == 1).sum())
                    neg_count = int(((ore != 1) & known_mask).sum())
                    pos_rate = (float(pos_count) / float(known_count)) if known_count > 0 else None
                    label_profile = {
                        "label_col": "Ore",
                        "known_count": known_count,
                        "missing_count": int(ore.isna().sum()),
                        "pos_count": pos_count,
                        "neg_count": neg_count,
                        "pos_rate": pos_rate,
                    }

                    probs_obj = predictions.get("probabilities")
                    if isinstance(probs_obj, list) and probs_obj:
                        try:
                            p_all = np.asarray(probs_obj, dtype=float)
                        except Exception:
                            p_all = None
                        if p_all is not None and p_all.size:
                            y_all = (ore == 1).astype(int).to_numpy(dtype=int)
                            known_np = known_mask.to_numpy(dtype=bool)
                            n = int(min(p_all.size, y_all.size, known_np.size))
                            p = p_all[:n]
                            y = y_all[:n]
                            known = known_np[:n]
                            p = p[known]
                            y = y[known]
                            p = np.clip(p, 0.0, 1.0)

                            brier = float(np.mean((p - y) ** 2)) if p.size else None
                            eps = 1e-15
                            log_loss = float(-np.mean((y * np.log(np.clip(p, eps, 1.0 - eps))) + ((1 - y) * np.log(np.clip(1.0 - p, eps, 1.0 - eps))))) if p.size else None

                            n_bins = 10
                            if p.size:
                                bin_ids = np.minimum((p * n_bins).astype(int), n_bins - 1)
                                ece = 0.0
                                mce = 0.0
                                for b in range(n_bins):
                                    mask = bin_ids == b
                                    if not np.any(mask):
                                        continue
                                    acc = float(np.mean(y[mask]))
                                    conf = float(np.mean(p[mask]))
                                    gap = abs(acc - conf)
                                    w = float(np.mean(mask))
                                    ece += w * gap
                                    if gap > mce:
                                        mce = gap
                            else:
                                ece, mce = (None, None)

                            calibration = {"brier": brier, "log_loss": log_loss, "ece_10": (float(ece) if ece is not None else None), "mce_10": (float(mce) if mce is not None else None)}

                            if p.size:
                                order = np.argsort(-p)
                                y_sorted = y[order]
                                pos_total = int(np.sum(y_sorted == 1))
                                k_1pct = int(max(1, round(0.01 * float(y_sorted.size))))
                                top_1pct = y_sorted[:k_1pct]
                                prec_1pct = float(np.mean(top_1pct)) if top_1pct.size else None
                                rec_1pct = (float(np.sum(top_1pct)) / float(pos_total)) if pos_total > 0 else None

                                k_pos = int(min(max(pos_total, 1), y_sorted.size))
                                top_kpos = y_sorted[:k_pos]
                                prec_kpos = float(np.mean(top_kpos)) if top_kpos.size else None
                                rec_kpos = (float(np.sum(top_kpos)) / float(pos_total)) if pos_total > 0 else None

                                ranking = {
                                    "precision_at_1pct": prec_1pct,
                                    "recall_at_1pct": rec_1pct,
                                    "precision_at_poscount": prec_kpos,
                                    "recall_at_poscount": rec_kpos,
                                }
    except Exception:
        data_profile = None
        label_profile = None
        calibration = None
        ranking = None

    token_report = None
    if TokenMonitor:
        try:
            token_report = TokenMonitor().get_report()
        except Exception:
            token_report = None

    def _score_lower_better(value: Optional[float], *, good: float, bad: float) -> Optional[float]:
        if value is None:
            return None
        try:
            good_f = float(good)
            bad_f = float(bad)
            value_f = float(value)
        except Exception:
            return None
        if bad_f <= good_f:
            return _clamp01(1.0 - value_f)
        if value_f <= good_f:
            return 1.0
        if value_f >= bad_f:
            return 0.0
        return _clamp01(1.0 - ((value_f - good_f) / (bad_f - good_f)))

    env_scoring = os.environ.get("GEOCHEM_EVAL_SCORING") or os.environ.get("AGENTS_EVAL_SCORING")
    cfg_scoring = config.get("eval_scoring") if isinstance(config, dict) else None
    scoring_overrides = cfg_scoring if isinstance(cfg_scoring, dict) else _safe_json_object(env_scoring)

    default_scoring = {
        "overall_weights": {"outcome": (1.0 / 3.0), "process": (1.0 / 3.0), "mechanism": (1.0 / 3.0)},
        "outcome_weights": {"acr": (1.0 / 3.0), "model_quality": (1.0 / 3.0), "overfit_penalty": (1.0 / 3.0)},
        "process_weights": {
            "checks_pass_rate": 0.25,
            "rework_rate": 0.25,
            "budget_utilization": 0.25,
            "decision_stability": 0.25,
        },
        "mechanism_weights": {
            "sofr": 0.25,
            "json_repair_success_rate": 0.25,
            "reflection_intensity": 0.25,
            "token_efficiency": 0.25,
        },
        "targets": {
            "structured_failure_rate_good": 0.05,
            "rework_rate_good": 0.2,
            "budget_utilization_good": 0.7,
            "budget_utilization_bad": 1.0,
            "reflection_intensity_good": 0.3,
            "tokens_per_run_good": 20000,
        },
    }

    professional_scoring = {
        "overall_weights": {
            "task_outcome": 0.2,
            "collaboration": 0.2,
            "governance_control": 0.2,
            "reliability": 0.2,
            "efficiency": 0.2,
        },
        "dimension_weights": {
            "task_outcome": {"task_success": 0.5, "artifact_completeness": 0.5},
            "collaboration": {
                "handoff_cohesion": 1.0,
            },
            "governance_control": {
                "verification_pass": 0.5,
                "rework_control": 0.5,
            },
            "reliability": {
                "structured_reliability": (1.0 / 3.0),
                "repair_resilience": (1.0 / 3.0),
                "decision_consistency": (1.0 / 3.0),
            },
            "efficiency": {"token_efficiency": 1.0},
        },
        "targets": {
            "budget_utilization_good": 0.7,
            "budget_utilization_bad": 1.0,
            "structured_failure_rate_good": 0.05,
            "ece_good": 0.05,
            "ece_bad": 0.20,
            "brier_bad": 0.25,
            "log_loss_bad": 1.0,
            "tokens_per_run_good": 20000,
        },
    }

    scoring = dict(default_scoring)
    if isinstance(scoring_overrides, dict):
        for k in ["overall_weights", "outcome_weights", "process_weights", "mechanism_weights", "targets"]:
            if isinstance(scoring_overrides.get(k), dict):
                merged = dict(scoring.get(k) or {})
                merged.update(scoring_overrides.get(k) or {})
                scoring[k] = merged

    targets_obj = scoring.get("targets")
    targets = targets_obj if isinstance(targets_obj, dict) else {}
    try:
        sofr_good = float(targets.get("structured_failure_rate_good", 0.05))
    except Exception:
        sofr_good = 0.05
    try:
        rr_good = float(targets.get("rework_rate_good", 0.2))
    except Exception:
        rr_good = 0.2
    try:
        bu_good = float(targets.get("budget_utilization_good", 0.7))
    except Exception:
        bu_good = 0.7
    try:
        bu_bad = float(targets.get("budget_utilization_bad", 1.0))
    except Exception:
        bu_bad = 1.0
    try:
        ri_good = float(targets.get("reflection_intensity_good", 0.3))
    except Exception:
        ri_good = 0.3
    try:
        tokens_good = float(targets.get("tokens_per_run_good", 20000))
    except Exception:
        tokens_good = 20000.0

    acr_s = _clamp01(acr)
    mq_s = _clamp01(mq)
    ofp_s = _clamp01(ofp)
    outcome_weights_obj = scoring.get("outcome_weights")
    outcome_weights = outcome_weights_obj if isinstance(outcome_weights_obj, dict) else {}
    outcome_score = _weighted_mean(
        [
            (acr_s, float(outcome_weights.get("acr", 0.25))),
            (mq_s, float(outcome_weights.get("model_quality", 0.55))),
            ((1.0 - ofp_s) if ofp_s is not None else None, float(outcome_weights.get("overfit_penalty", 0.2))),
        ]
    )

    cpr_s = _clamp01(cpr)
    rr_s = None
    if rr is not None and rr_good > 0:
        rr_s = _clamp01(1.0 - (float(rr) / float(rr_good)))
    bu_s = None
    if bu is not None:
        if bu_bad <= bu_good:
            bu_s = _clamp01(1.0 - float(bu))
        else:
            raw = (float(bu) - float(bu_good)) / float(bu_bad - bu_good)
            bu_s = _clamp01(1.0 - raw)
    ds_s = _clamp01(ds)
    process_weights_obj = scoring.get("process_weights")
    process_weights = process_weights_obj if isinstance(process_weights_obj, dict) else {}
    process_score = _weighted_mean(
        [
            (cpr_s, float(process_weights.get("checks_pass_rate", 0.4))),
            (rr_s, float(process_weights.get("rework_rate", 0.3))),
            (bu_s, float(process_weights.get("budget_utilization", 0.2))),
            (ds_s, float(process_weights.get("decision_stability", 0.1))),
        ]
    )

    sofr_s = None
    if sofr is not None and sofr_good > 0:
        sofr_s = _clamp01(1.0 - (float(sofr) / float(sofr_good)))
    jr_s = _clamp01(jr)
    ri_s = None
    if ri is not None:
        if ri_good <= 0:
            ri_s = _clamp01(1.0 - float(ri))
        else:
            excess = float(ri) - float(ri_good)
            ri_s = _clamp01(1.0 - (excess if excess > 0 else 0.0))
    token_eff_s = None
    if isinstance(token_report, dict):
        total_tokens_obj = token_report.get("Total Tokens")
        try:
            total_tokens_f = float(total_tokens_obj) if total_tokens_obj is not None else None
        except Exception:
            total_tokens_f = None
        if total_tokens_f is not None and total_tokens_f > 0 and tokens_good > 0:
            ratio = math.log1p(total_tokens_f) / math.log1p(tokens_good)
            token_eff_s = _clamp01(1.0 - (ratio - 1.0 if ratio > 1.0 else 0.0))

    time_eff_s = None
    if bu_time is not None:
        time_eff_s = _score_lower_better(bu_time, good=bu_good, bad=1.0)
    step_eff_s = None
    if bu_steps is not None:
        step_eff_s = _score_lower_better(bu_steps, good=bu_good, bad=1.0)

    cal_s = None
    if isinstance(calibration, dict):
        cal_ece_s = _score_lower_better(calibration.get("ece_10"), good=0.05, bad=0.20)
        cal_brier_s = _score_lower_better(calibration.get("brier"), good=0.05, bad=0.25)
        cal_logloss_s = _score_lower_better(calibration.get("log_loss"), good=0.20, bad=1.0)
        cal_s = _weighted_mean([(cal_ece_s, 0.5), (cal_brier_s, 0.25), (cal_logloss_s, 0.25)])

    mechanism_weights_obj = scoring.get("mechanism_weights")
    mechanism_weights = mechanism_weights_obj if isinstance(mechanism_weights_obj, dict) else {}
    mechanism_score = _weighted_mean(
        [
            (sofr_s, float(mechanism_weights.get("sofr", 0.4))),
            (jr_s, float(mechanism_weights.get("json_repair_success_rate", 0.3))),
            (ri_s, float(mechanism_weights.get("reflection_intensity", 0.3))),
            (token_eff_s, float(mechanism_weights.get("token_efficiency", 0.0))),
        ]
    )

    overall_weights_obj = scoring.get("overall_weights")
    overall_weights = overall_weights_obj if isinstance(overall_weights_obj, dict) else {}
    overall_score01 = None
    if tsr:
        overall_score01 = _weighted_mean(
            [
                (outcome_score, float(overall_weights.get("outcome", 0.4))),
                (process_score, float(overall_weights.get("process", 0.3))),
                (mechanism_score, float(overall_weights.get("mechanism", 0.3))),
            ]
        )
    overall_score_100 = (float(overall_score01) * 100.0) if overall_score01 is not None else (0.0 if not tsr else None)

    professional_overall_weights = professional_scoring.get("overall_weights") if isinstance(professional_scoring, dict) else {}
    professional_overall_weights = professional_overall_weights if isinstance(professional_overall_weights, dict) else {}
    professional_dimension_weights = professional_scoring.get("dimension_weights") if isinstance(professional_scoring, dict) else {}
    professional_dimension_weights = professional_dimension_weights if isinstance(professional_dimension_weights, dict) else {}
    task_outcome_weights = professional_dimension_weights.get("task_outcome") if isinstance(professional_dimension_weights.get("task_outcome"), dict) else {}
    collaboration_weights = professional_dimension_weights.get("collaboration") if isinstance(professional_dimension_weights.get("collaboration"), dict) else {}
    governance_weights = professional_dimension_weights.get("governance_control") if isinstance(professional_dimension_weights.get("governance_control"), dict) else {}
    reliability_weights = professional_dimension_weights.get("reliability") if isinstance(professional_dimension_weights.get("reliability"), dict) else {}
    efficiency_weights = professional_dimension_weights.get("efficiency") if isinstance(professional_dimension_weights.get("efficiency"), dict) else {}
    task_outcome_score = _weighted_mean(
        [
            ((1.0 if tsr else 0.0), float(task_outcome_weights.get("task_success", 0.5))),
            (acr_s, float(task_outcome_weights.get("artifact_completeness", 0.5))),
        ]
    )
    collaboration_score = _weighted_mean(
        [
            (handoff_cohesion, float(collaboration_weights.get("handoff_cohesion", 1.0))),
        ]
    )
    governance_control_score = _weighted_mean(
        [
            (cpr_s, float(governance_weights.get("verification_pass", 0.5))),
            (rr_s, float(governance_weights.get("rework_control", 0.5))),
        ]
    )
    structured_reliability = _clamp01((1.0 - sofr) if sofr is not None else None)
    reliability_score = _weighted_mean(
        [
            (structured_reliability, float(reliability_weights.get("structured_reliability", (1.0 / 3.0)))),
            (_clamp01(repair_resilience), float(reliability_weights.get("repair_resilience", (1.0 / 3.0)))),
            (ds_s, float(reliability_weights.get("decision_consistency", (1.0 / 3.0)))),
        ]
    )
    efficiency_score = _weighted_mean(
        [
            (token_eff_s, float(efficiency_weights.get("token_efficiency", 1.0))),
        ]
    )
    professional_overall01 = _weighted_mean(
        [
            (task_outcome_score, float(professional_overall_weights.get("task_outcome", 0.2))),
            (collaboration_score, float(professional_overall_weights.get("collaboration", 0.2))),
            (governance_control_score, float(professional_overall_weights.get("governance_control", 0.2))),
            (reliability_score, float(professional_overall_weights.get("reliability", 0.2))),
            (efficiency_score, float(professional_overall_weights.get("efficiency", 0.2))),
        ]
    )
    professional_overall_100 = (float(professional_overall01) * 100.0) if professional_overall01 is not None else None

    metrics = {
        "schema_version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "run": {
            "output_dir": os.path.abspath(str(output_dir)),
            "data_path": os.path.abspath(str(data_path)),
            "target_deposit_type": target_deposit,
            "phase": phase or None,
            "runtime_seconds": runtime_seconds_f,
            "runtime_human": result.get("workflow_runtime_human"),
            "governance_run_id": gov.get("run_id") if isinstance(gov, dict) else None,
            "config": config,
        },
        "outcome": {"tsr": tsr, "acr": acr, "model_quality": mq, "overfit_penalty": ofp, "label_profile": label_profile, "calibration": calibration, "ranking": ranking},
        "process": {
            "checks_pass_rate": cpr,
            "checks": {"error_fail": err_fail, "error_total": err_total, "warn_fail": warn_fail, "warn_total": warn_total},
            "rework_rate": rr,
            "budget_utilization": bu,
            "decision_stability": ds,
            "hitl_intervention_rate": hir,
        },
        "mechanism": {"sofr": sofr, "json_repair_success_rate": jr, "reflection_intensity": ri, "eval_stats_totals": totals},
        "domain": {"high_potential_ratio": high_potential_ratio, "confidence_level": conf_level or None},
        "data_profile": data_profile,
        "artifacts": {"expected": expected_artifacts, "output_paths": output_paths},
        "agent_eval_stats": agent_metrics,
        "token_usage": token_report,
        "professional_evaluation": {
            "overall": professional_overall_100,
            "dimensions_0_1": {
                "task_outcome": task_outcome_score,
                "collaboration": collaboration_score,
                "governance_control": governance_control_score,
                "reliability": reliability_score,
                "efficiency": efficiency_score,
            },
            "indicators_0_1": {
                "task_success": (1.0 if tsr else 0.0),
                "artifact_completeness": acr_s,
                "handoff_cohesion": handoff_cohesion,
                "verification_pass": cpr_s,
                "rework_control": rr_s,
                "structured_reliability": structured_reliability,
                "repair_resilience": _clamp01(repair_resilience),
                "decision_consistency": ds_s,
                "token_efficiency": token_eff_s,
                "predictive_quality": mq_s,
                "calibration_quality": cal_s,
            },
            "raw_metrics": {
                "active_roles": active_roles,
                "route_transitions": route_transitions,
                "route_revisits": route_revisits,
                "time_budget_ratio": bu_time,
                "step_budget_ratio": bu_steps,
            },
            "professional_scoring_config": professional_scoring,
        },
        "score": {
            "overall_0_100": overall_score_100,
            "subscores_0_1": {"outcome": outcome_score, "process": process_score, "mechanism": mechanism_score},
            "signals_0_1": {
                "acr": acr_s,
                "model_quality": mq_s,
                "overfit_penalty": ofp_s,
                "checks_pass_rate": cpr_s,
                "rework_score": rr_s,
                "budget_score": bu_s,
                "decision_stability": ds_s,
                "structured_output_score": sofr_s,
                "json_repair_success_rate": jr_s,
                "reflection_score": ri_s,
                "token_efficiency": token_eff_s,
            },
            "scoring_config": scoring,
        },
    }
    return metrics
def _ensure_logging_configured(verbose: bool = False) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        return
    logs_dir = os.path.join(_get_app_data_dir(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        file_handler = logging.FileHandler(os.path.join(logs_dir, "agent_system.log"), encoding="utf-8")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        root_logger.setLevel(logging.INFO)
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    redacting_filter = _RedactingFilter()
    for handler in root_logger.handlers:
        handler.addFilter(redacting_filter)
    _LOGGING_CONFIGURED = True
logger = logging.getLogger(__name__)
def _resolve_workflow_imports():
    errors = []
    try:
        from .geo_workflow import GeoChemistryWorkflow, create_workflow
        return create_workflow, GeoChemistryWorkflow
    except Exception as e:
        errors.append(e)
    try:
        from agents.geo_workflow import GeoChemistryWorkflow, create_workflow
        return create_workflow, GeoChemistryWorkflow
    except Exception as e:
        errors.append(e)
    try:
        from geo_workflow import GeoChemistryWorkflow, create_workflow
        return create_workflow, GeoChemistryWorkflow
    except Exception as e:
        errors.append(e)
    last = errors[-1] if errors else ImportError("unknown import error")
    raise ImportError(f"无法导入工作流模块: {last}") from last
_TokenMonitor: Optional[type[Any]] = None
try:
    from utils.token_counter import TokenMonitor as _TokenMonitor
except ImportError:
    pass
TokenMonitor: Optional[type[Any]] = _TokenMonitor
HAS_LANGSMITH = False
try:
    import langsmith
    HAS_LANGSMITH = True
except ImportError:
    pass
_load_data: Optional[Callable[[str], Any]] = None
try:
    from utils.data_utils import load_data as _load_data
except ImportError:
    try:
        from agents.utils.data_utils import load_data as _load_data_alt
        _load_data = _load_data_alt
    except ImportError:
        pass
load_data = _load_data
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="地球化学多智能体成矿潜力预测软件")
    default_data_path = os.path.join(os.path.dirname(__file__), "data", "data.csv")
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    env_flag = os.environ.get("GEOCHEM_STRUCTURED_OUTPUT") or os.environ.get("AGENTS_STRUCTURED_OUTPUT")
    if env_flag is None:
        env_enabled = True
    else:
        env_norm = str(env_flag).strip().lower()
        if env_norm in truthy:
            env_enabled = True
        elif env_norm in falsy:
            env_enabled = False
        else:
            env_enabled = True

    reflection_flag = os.environ.get("GEOCHEM_REFLECTION") or os.environ.get("AGENTS_REFLECTION")
    if reflection_flag is None:
        reflection_enabled = True
    else:
        reflection_norm = str(reflection_flag).strip().lower()
        if reflection_norm in truthy:
            reflection_enabled = True
        elif reflection_norm in falsy:
            reflection_enabled = False
        else:
            reflection_enabled = True

    reflection_rounds_flag = os.environ.get("GEOCHEM_REFLECTION_MAX_ROUNDS") or os.environ.get("AGENTS_REFLECTION_MAX_ROUNDS")
    if reflection_rounds_flag is None:
        reflection_rounds_default = 1
    else:
        try:
            reflection_rounds_default = int(str(reflection_rounds_flag).strip())
        except Exception:
            reflection_rounds_default = 1
    if reflection_rounds_default < 0:
        reflection_rounds_default = 0
    if reflection_rounds_default > 3:
        reflection_rounds_default = 3
    parser.add_argument("-d", "--data", type=str, default=default_data_path, help=f"数据文件路径 (默认: {default_data_path})")
    parser.add_argument("-o", "--output", type=str, required=False, default="./output", help="输出目录路径 (默认: ./output)")
    parser.add_argument("-v", "--verbose", action="store_true", help="启用详细输出")
    parser.add_argument("--structured-output", action="store_true", default=env_enabled, help="启用结构化输出解析")
    parser.add_argument("--no-structured-output", action="store_false", dest="structured_output", help="禁用结构化输出解析")
    parser.add_argument("--reflection", action="store_true", default=reflection_enabled, help="启用反思与自修正机制")
    parser.add_argument("--no-reflection", action="store_false", dest="reflection", help="禁用反思与自修正机制")
    parser.add_argument("--reflection-max-rounds", type=int, default=reflection_rounds_default, help="反思与自修正最大轮数 (0-3)")
    parser.add_argument(
        "--learning-mode",
        choices=["unsupervised", "self_supervised", "supervised"],
        default="unsupervised",
        help="SOM学习方式（默认：unsupervised）",
    )
    parser.add_argument("--geology-four-stage-cot", action="store_true", default=False, help="开启地质解译四阶段CoT（默认关闭，关闭时仍输出最终地质解释）")
    parser.add_argument("--som-all-elements", action="store_true", default=False, help="开启SOM全元素实验（默认关闭，仅运行筛选元素实验）")
    parser.add_argument("--no-som-all-elements", action="store_false", dest="som_all_elements", help="关闭SOM全元素实验")
    parser.add_argument("--auto-programming", action="store_true", default=False, help="开启数据科学智能体的自动编程能力")
    parser.add_argument(
        "--som-use-raw-data",
        "--som-use-raw",
        action="store_true",
        default=False,
        dest="som_use_raw_data",
        help="SOM使用原始数据作为输入（开启后使用原始数据；默认使用预处理后数据）",
    )
    parser.add_argument(
        "--no-som-use-raw-data",
        "--no-som-use-raw",
        action="store_false",
        dest="som_use_raw_data",
        help="SOM不使用原始数据输入（使用预处理后数据）",
    )
    parser.add_argument(
        "-a",
        "--study-area",
        type=str,
        required=False,
        default="南岭成矿带",
        help="研究区地点/区域信息（默认并固定为：南岭成矿带）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")),
        help="LangSmith API密钥，如未提供将尝试从环境变量获取",
    )
    parser.add_argument("-t", "--target", type=str, required=False, default=None, help="Target deposit type (e.g., Tungsten, Copper, Gold)")
    parser.add_argument(
        "--output-language",
        type=str,
        choices=["en", "zh"],
        default=_resolve_output_language(),
        help="输出语言：en=英文(默认)，zh=中文",
    )
    args, unknown = parser.parse_known_args()
    try:
        rounds = int(getattr(args, "reflection_max_rounds", 1))
    except Exception:
        rounds = 1
    if rounds < 0:
        rounds = 0
    if rounds > 3:
        rounds = 3
    setattr(args, "reflection_max_rounds", rounds)
    if unknown and getattr(args, "verbose", False):
        _ensure_logging_configured(verbose=True)
        logger.warning(f"忽略未识别的参数: {unknown}")
    return args
def _disable_langsmith_tracing(verbose: bool = False) -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    for k in ["LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"]:
        try:
            os.environ.pop(k, None)
        except Exception:
            pass
    if verbose:
        _ensure_logging_configured(verbose=True)
        logger.info("LangSmith tracing 已禁用")
def setup_langsmith(api_key: Optional[str] = None, project_name: str = "default", verbose: bool = False) -> bool:
    if not HAS_LANGSMITH:
        return False
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
    current_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if current_key and str(current_key).startswith("lsv2_"):
        os.environ["LANGSMITH_API_KEY"] = current_key
        os.environ["LANGCHAIN_API_KEY"] = current_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        if verbose:
            _ensure_logging_configured(verbose=True)
            logger.info(f"成功导入langsmith模块，版本: {langsmith.__version__}")
            logger.info(f"LangSmith跟踪已启用 (项目: {project_name})")
        return True
    return False
def check_requirements() -> bool:
    required_packages = [
        ("langgraph", "langgraph"),
        ("langchain", "langchain"),
        ("langchain_core", "langchain_core"),
        ("requests", "requests"),
        ("markdown", "markdown"),
        ("yaml", "PyYAML"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scipy", "scipy"),
    ]
    missing_packages = []
    for import_name, pip_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    if missing_packages:
        _ensure_logging_configured()
        logger.error("缺少以下必要的Python包：")
        for package in missing_packages:
            logger.error(f"- {package}")
        logger.info("请运行以下命令安装所有依赖：pip install -r requirements.txt")
        return False
    return True
def create_app():
    _, GeoChemistryWorkflow = _resolve_workflow_imports()
    workflow = GeoChemistryWorkflow()
    _ensure_logging_configured()
    logger.info(f"工作流图已创建，类型: {type(workflow.workflow).__name__}")
    return workflow.workflow
def main() -> None:
    if load_dotenv is not None:
        env_path = os.path.join(current_dir, ".env")
        try:
            if os.path.exists(env_path):
                load_dotenv(env_path)
        except Exception:
            pass
    _ensure_logging_configured()
    logger.info("=" * 60)
    logger.info("欢迎使用地球化学多智能体成矿潜力预测软件")
    logger.info("基于LangGraph的多智能体地球化学数据分析平台")
    logger.info("=" * 60)
    if not check_requirements():
        logger.error("程序将在安装所有依赖后继续执行。")
        sys.exit(1)
    args = parse_arguments()
    _ensure_logging_configured(verbose=bool(getattr(args, "verbose", False)))
    try:
        create_workflow, _ = _resolve_workflow_imports()
    except Exception as e:
        logger.error(f"无法导入工作流模块: {e}")
        logger.info("请确保以项目根目录运行，或已正确配置PYTHONPATH。")
        sys.exit(1)
    _disable_langsmith_tracing(verbose=bool(getattr(args, "verbose", False)))
    data_path = args.data
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.info("请指定正确的数据文件路径: python main.py -d /path/to/your/data.csv")
        sys.exit(1)
    study_area_location = "南岭成矿带"
    target_deposit = "钨矿"
    interaction_mode = "auto"
    is_interactive = False
    try:
        is_interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    except Exception:
        is_interactive = False
    if is_interactive and not study_area_location:
        logger.warning("未指定研究区地点 (Study Area)")
        logger.info("为了更精确地检索目标矿种信息，请先提供研究区地点/区域（如省市/矿集区/成矿带等）。")
        while not study_area_location:
            logger.info("【INPUT】请输入研究区地点/区域:")
            user_input = _input_with_log_prefix(logger).strip()
            if user_input:
                study_area_location = user_input
            else:
                logger.error("研究区地点不能为空，请重新输入。")
    preview_data = None
    if load_data:
        try:
            logger.info(f"正在读取数据文件以进行预览: {data_path}")
            preview_data = load_data(data_path)
            logger.info("数据预览:")
            logger.info(f"- 数据形状: {preview_data.shape}")
            logger.info(f"- 数据列名: {', '.join(list(preview_data.columns)[:10])}...")
            logger.info(f"- 数据类型: {preview_data.dtypes.value_counts().to_dict()}")
        except Exception as e:
            logger.warning(f"数据预览失败: {e}")
            logger.info("将在工作流执行阶段尝试重新加载并验证数据...")
            preview_data = None
    if not target_deposit:
        if not is_interactive:
            logger.error("未指定目标矿种，且当前为非交互模式。")
            logger.info("请使用 -t/--target 指定目标矿种类型，或设置 TARGET_DEPOSIT_TYPE 环境变量。")
            sys.exit(2)
        logger.warning("未指定目标矿种 (Target Deposit)")
        logger.info("为了进行针对性的成矿预测，系统需要知道您关注的目标矿种。")
        while not target_deposit:
            logger.info("【INPUT】请输入目标矿种类型:")
            user_input = _input_with_log_prefix(logger).strip()
            if user_input:
                target_deposit = user_input
            else:
                logger.error("目标矿种不能为空，请重新输入。")
        logger.info(f"已设置目标矿种: {target_deposit}")
        while True:
            logger.info("【INPUT】请选择运行模式：1 默认智能化流程；2 开启人机交互（HITL）")
            mode_input = _input_with_log_prefix(logger).strip()
            if not mode_input or mode_input == "1":
                interaction_mode = "auto"
                break
            if mode_input == "2":
                interaction_mode = "hitl"
                break
            logger.error("输入无效，请输入 1 或 2。")
    output_dir = args.output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.rstrip(os.sep)
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    run_log_path = _attach_run_log_file(output_dir)
    if run_log_path:
        logger.info(f"本次运行日志: {run_log_path}")
    try:
        workflow = create_workflow(output_dir=output_dir)
        logger.info(f"正在处理数据文件: {data_path}")
        config: dict[str, Any] = {
            "structured_output_enabled": bool(getattr(args, "structured_output", False)),
            "reflection_enabled": bool(getattr(args, "reflection", False)),
            "reflection_max_rounds": int(getattr(args, "reflection_max_rounds", 1)),
            "learning_mode": str(getattr(args, "learning_mode", "unsupervised")).strip().lower(),
            "som_qe_calibration_enabled": True,
            "geology_four_stage_cot_enabled": bool(getattr(args, "geology_four_stage_cot", False)),
            "som_all_elements_enabled": bool(getattr(args, "som_all_elements", False)),
            "som_use_raw_data_enabled": bool(getattr(args, "som_use_raw_data", False)),
            "interaction_mode": interaction_mode,
            "study_area_location": study_area_location,
            "output_language": str(getattr(args, "output_language", "en")).strip().lower(),
            "auto_programming_enabled": bool(getattr(args, "auto_programming", False)),
            "auto_programming_max_retries": 1,
        }
        output_lang = _apply_output_language_env(config=config)
        truthy = {"1", "true", "yes", "on"}
        falsy = {"0", "false", "no", "off"}

        def _env_flag(names: tuple[str, ...]) -> Optional[str]:
            for name in names:
                val = os.environ.get(name)
                if val is not None:
                    return val
            return None

        def _flag_source(argv_on: str, argv_off: str, env_names: tuple[str, ...], default_when_unset: bool) -> tuple[str, Optional[str], bool]:
            argv = [str(a) for a in sys.argv[1:]]
            if argv_on in argv:
                return "命令行", None, True
            if argv_off in argv:
                return "命令行", None, False
            raw = _env_flag(env_names)
            if raw is None:
                return "默认", None, default_when_unset
            norm = str(raw).strip().lower()
            if norm in truthy:
                return "环境变量", raw, True
            if norm in falsy:
                return "环境变量", raw, False
            return "环境变量(无效值)", raw, default_when_unset

        structured_source, structured_raw, _ = _flag_source(
            argv_on="--structured-output",
            argv_off="--no-structured-output",
            env_names=("GEOCHEM_STRUCTURED_OUTPUT", "AGENTS_STRUCTURED_OUTPUT"),
            default_when_unset=True,
        )
        reflection_source, reflection_raw, _ = _flag_source(
            argv_on="--reflection",
            argv_off="--no-reflection",
            env_names=("GEOCHEM_REFLECTION", "AGENTS_REFLECTION"),
            default_when_unset=True,
        )

        logger.info("功能开关状态：")
        logger.info(f"- 地质解译四阶段CoT: {'开启（受全局CoT开关约束）' if config['geology_four_stage_cot_enabled'] else '关闭，仅输出最终解释'}")
        logger.info(
            f"- SOM学习方式: {config['learning_mode']}；"
            f"QE概率校准: {'开启' if config['som_qe_calibration_enabled'] else '关闭'}"
        )
        logger.info(
            f"- 反思与自修正机制: {'开启' if config['reflection_enabled'] else '关闭'}"
            f" (来源: {reflection_source}"
            f"{'' if reflection_raw is None else f', 值: {reflection_raw}'}; max_rounds={config['reflection_max_rounds']})"
        )
        logger.info(
            f"- 结构化输出与工具标准化: 结构化输出{'开启' if config['structured_output_enabled'] else '关闭'}"
            f" (来源: {structured_source}{'' if structured_raw is None else f', 值: {structured_raw}'}); 工具标准化开启(固定)"
        )
        logger.info(f"- SOM全元素实验: {'开启' if config['som_all_elements_enabled'] else '关闭'} (默认关闭，可用 --som-all-elements 开启)")
        logger.info(f"- SOM输入使用原始数据: {'开启' if config['som_use_raw_data_enabled'] else '关闭'} (默认关闭，关闭时使用预处理后数据)")
        logger.info(f"- 输出语言: {output_lang}")
        result = workflow.run(data_path, target_deposit_type=target_deposit, study_area_location=study_area_location, config=config, data=preview_data)
        eval_stats = None
        get_eval_stats = None
        try:
            from base_agent import get_eval_stats as _get_eval_stats0

            get_eval_stats = _get_eval_stats0
        except Exception:
            try:
                from agents.base_agent import get_eval_stats as _get_eval_stats1

                get_eval_stats = _get_eval_stats1
            except Exception:
                try:
                    from .base_agent import get_eval_stats as _get_eval_stats2

                    get_eval_stats = _get_eval_stats2
                except Exception:
                    get_eval_stats = None

        if get_eval_stats is not None:
            try:
                eval_stats = get_eval_stats(reset=True)
            except Exception:
                eval_stats = None
            if isinstance(eval_stats, dict) and eval_stats:
                totals: dict[str, int] = {}
                for agent_stats in eval_stats.values():
                    if not isinstance(agent_stats, dict):
                        continue
                    for k, val in agent_stats.items():
                        if isinstance(k, str) and isinstance(val, int):
                            totals[k] = totals.get(k, 0) + val
                logger.info("评估统计（反思/结构化输出相关）：")
                logger.info(
                    f"- 合计: decide={totals.get('decide_calls', 0)}, decide_json={totals.get('decide_json_calls', 0)}, "
                    f"reflection_rounds={totals.get('reflection_text_rounds', 0)}, "
                    f"json_repair={totals.get('json_repair_successes', 0)}/{totals.get('json_repair_attempts', 0)}, "
                    f"structured_parse_failures={totals.get('structured_parse_failures', 0)}"
                )
                for agent_name in sorted(eval_stats.keys()):
                    agent_stats = eval_stats.get(agent_name, {})
                    if not isinstance(agent_stats, dict):
                        continue
                    decide_calls = agent_stats.get("decide_calls", 0)
                    decide_json_calls = agent_stats.get("decide_json_calls", 0)
                    reflection_rounds = agent_stats.get("reflection_text_rounds", 0)
                    json_repair_attempts = agent_stats.get("json_repair_attempts", 0)
                    json_repair_successes = agent_stats.get("json_repair_successes", 0)
                    structured_parse_failures = agent_stats.get("structured_parse_failures", 0)
                    logger.info(
                        f"- {agent_name}: decide={decide_calls}, decide_json={decide_json_calls}, "
                        f"reflection_rounds={reflection_rounds}, json_repair={json_repair_successes}/{json_repair_attempts}, "
                        f"structured_parse_failures={structured_parse_failures}"
                    )
        logger.info("=" * 60)
        logger.info("地球化学多智能体成矿潜力预测完成")
        logger.info("=" * 60)
        output_paths = None
        if isinstance(result.get("output_results"), dict):
            output_paths = result.get("output_results")
        elif isinstance(result.get("result_output"), dict):
            output_paths = result.get("result_output")
        try:
            metrics = _compute_eval_metrics(
                result=result if isinstance(result, dict) else {},
                output_dir=output_dir,
                data_path=data_path,
                target_deposit=target_deposit,
                config=config,
                eval_stats=eval_stats if isinstance(eval_stats, dict) else None,
            )
            figures = _generate_eval_metrics_figures(metrics, output_dir=output_dir)
            if figures:
                artifacts_obj = metrics.get("artifacts")
                artifacts = artifacts_obj if isinstance(artifacts_obj, dict) else {}
                output_paths_obj = artifacts.get("output_paths")
                if isinstance(output_paths_obj, dict):
                    metrics_paths = output_paths_obj
                else:
                    metrics_paths = {}
                    artifacts["output_paths"] = metrics_paths
                    metrics["artifacts"] = artifacts
                for k, fig_path in figures.items():
                    if isinstance(fig_path, str) and fig_path.strip():
                        metrics_paths[k] = fig_path
                if not isinstance(output_paths, dict):
                    output_paths = metrics_paths
            metrics_md_path = _atomic_write_text(
                os.path.join(output_dir, "reports", "eval_metrics.md"), _format_eval_metrics_markdown(metrics), encoding="utf-8"
            )
            if isinstance(output_paths, dict):
                output_paths["eval_metrics_md"] = metrics_md_path
            logger.info(f"评测指标摘要已保存: {metrics_md_path}")
        except Exception as e:
            logger.warning(f"评测指标生成失败: {e}")
        if output_paths:
            logger.info("生成的报告和数据文件：")
            for file_type, file_path in output_paths.items():
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024
                    if file_size < 1024:
                        size_str = f"{file_size:.2f} KB"
                    else:
                        size_str = f"{file_size / 1024:.2f} MB"
                    logger.info(f"- {file_type.replace('_', ' ').title()}: {file_path} ({size_str})")
        logger.info("预测结果摘要：")
        if "prediction_model_final" in result:
            prediction = result["prediction_model_final"].get("predictions", {})
        elif "prediction_model" in result:
            prediction = result["prediction_model"].get("predictions", {})
        else:
            prediction = {}
        if prediction:
            high_potential_count = prediction.get("high_potential_count", 0)
            total_samples = prediction.get("total_samples", 0)
            if total_samples == 0:
                if "data_preprocessing" in result and "basic_stats" in result["data_preprocessing"]:
                    total_samples = result["data_preprocessing"]["basic_stats"].get("total_samples", 0)
                elif "geology_expert_primary" in result:
                    total_samples = result["geology_expert_primary"].get("total_samples", 0)
            logger.info(f"- 高潜力区域: {high_potential_count} 个")
            logger.info(f"- 总分析样本: {total_samples} 个")
            if total_samples > 0:
                percentage = (high_potential_count / total_samples) * 100
                logger.info(f"- 高潜力占比: {percentage:.2f}%")
        if "geology_expert_final" in result:
            final_geology = result["geology_expert_final"]
            if final_geology:
                mineralization_types = final_geology.get("mineralization_types", "未明确识别")
                logger.info("地质分析结果：")
                logger.info(f"- 推断矿化类型: {mineralization_types}")
        if "prediction_model" in result:
            mineral_info = result["prediction_model"].get("mineralization_info", {})
            if mineral_info:
                primary_type = mineral_info.get("primary_type", "未明确识别")
                logger.info(f"推断矿化类型: {primary_type}")
        logger.info("提示：请查看输出目录中的综合报告以获取详细分析结果。")
        if run_log_path:
            logger.info(f"完整运行日志已保存: {run_log_path}")
        if TokenMonitor:
            TokenMonitor().print_report()
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"程序执行出错：{str(e)}")
        if bool(getattr(args, "verbose", False)):
            logger.exception("异常堆栈")
        if TokenMonitor:
            TokenMonitor().print_report()
        logger.info("请检查以下几点：")
        logger.info("1. 数据文件格式是否正确")
        logger.info("2. 是否安装了所有依赖包")
        logger.info("3. Python版本是否兼容（建议Python 3.8+）")
        sys.exit(1)
if __name__ == "__main__":
    main()
