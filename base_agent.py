import os
import logging
import re
import importlib.util
import subprocess
import sys
import shlex
import inspect
from typing import Dict, Any, List, Optional, Type, Callable, Tuple
from langchain_core.prompts import ChatPromptTemplate
_JsonOutputParser: Optional[Type[Any]] = None
try:
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser as _JsonOutputParser
except Exception:
    from langchain_core.output_parsers import StrOutputParser
JsonOutputParser = _JsonOutputParser
try:
    from .utils.llm_utils import get_llm
except Exception:
    from utils.llm_utils import get_llm
_SkillContext: Optional[type[Any]] = None
_SkillRegistry: Optional[type[Any]] = None
_SkillSpecT: Optional[type[Any]] = None
_load_skills_from_env: Optional[Callable[..., Any]] = None
_load_skill_docs_index: Optional[Callable[..., Any]] = None
_load_skills_catalog: Optional[Callable[..., Any]] = None
try:
    from .skills.core import SkillContext as _SkillContextT, SkillRegistry as _SkillRegistryT, SkillSpec as _SkillSpec0
    from .skills.loader import load_skill_docs_index as _load_skill_docs_index_fn, load_skills_from_env as _load_skills_from_env_fn, load_skills_catalog as _load_skills_catalog_fn
    _SkillContext = _SkillContextT
    _SkillRegistry = _SkillRegistryT
    _SkillSpecT = _SkillSpec0
    _load_skills_from_env = _load_skills_from_env_fn
    _load_skill_docs_index = _load_skill_docs_index_fn
    _load_skills_catalog = _load_skills_catalog_fn
except Exception:
    try:
        from skills.core import SkillContext as _SkillContextT2, SkillRegistry as _SkillRegistryT2, SkillSpec as _SkillSpec1
        from skills.loader import load_skill_docs_index as _load_skill_docs_index_fn2, load_skills_from_env as _load_skills_from_env_fn2, load_skills_catalog as _load_skills_catalog_fn2
        _SkillContext = _SkillContextT2
        _SkillRegistry = _SkillRegistryT2
        _SkillSpecT = _SkillSpec1
        _load_skills_from_env = _load_skills_from_env_fn2
        _load_skill_docs_index = _load_skill_docs_index_fn2
        _load_skills_catalog = _load_skills_catalog_fn2
    except Exception:
        _SkillContext = None
        _SkillRegistry = None
        _SkillSpecT = None
        _load_skills_from_env = None
        _load_skill_docs_index = None
        _load_skills_catalog = None
_LOGGING_CONFIGURED = False
_REDACT_PATTERNS = [re.compile('(?i)\\b(api[_-]?key|access[_-]?key|secret|token|password)\\b\\s*[:=]\\s*([^\\s,;]+)'), re.compile('(?i)\\b(bearer)\\s+([A-Za-z0-9._\\-]+)'), re.compile('\\bsk-[A-Za-z0-9]{10,}\\b'), re.compile('lsv2_[A-Za-z0-9_\\-]{10,}')]
_EVAL_STATS: Dict[str, Dict[str, int]] = {}


def get_eval_stats(reset: bool = False) -> Dict[str, Dict[str, int]]:
    snapshot = {k: dict(v) for k, v in _EVAL_STATS.items()}
    if reset:
        _EVAL_STATS.clear()
    return snapshot
def _redact_text(text: str) -> str:
    try:
        s = str(text)
    except Exception:
        return '<unprintable>'
    for pattern in _REDACT_PATTERNS:
        if pattern.pattern.lower().startswith('(?i)\\b(bearer)'):
            s = pattern.sub('<REDACTED>', s)
        else:
            s = pattern.sub(lambda m: f'{m.group(1)}=<REDACTED>' if m.lastindex and m.lastindex >= 2 else '<REDACTED>', s)
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
def _ensure_logging_configured() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    try:
        base_dir = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    except Exception:
        base_dir = os.path.expanduser("~")
    logs_dir = os.path.join(base_dir, "GAI-MAS", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        file_handler = logging.FileHandler(os.path.join(logs_dir, 'agent_system.log'), encoding='utf-8')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        root_logger.setLevel(logging.INFO)
    redacting_filter = _RedactingFilter()
    for handler in root_logger.handlers:
        handler.addFilter(redacting_filter)
    _LOGGING_CONFIGURED = True
class BaseAgent:
    def __init__(self, agent_name: str, role_description: str, llm=None):
        self.agent_name = agent_name
        self.role_description = role_description
        _ensure_logging_configured()
        self.llm = llm or get_llm()
        self.memory: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.agent_name)
        self.logger.info(f'{self.agent_name} initialized')
        self.skills = _SkillRegistry(owner=self.agent_name) if _SkillRegistry is not None else None
        self.skill_docs = {}
        self.skills_catalog: List[Dict[str, str]] = []
        self.system_prompt = ChatPromptTemplate.from_template('\n            你是{agent_name}，{role_description}\n            \n            可用 Skills（name | description）：\n            {skills}\n            \n            当前对话历史：\n            {chat_history}\n            \n            当前任务：\n            {task}\n            \n            请根据你的角色和任务，生成下一步响应。如果需要调用工具，请使用指定格式。\n            ')
        if _load_skill_docs_index is not None:
            try:
                self.skill_docs = _load_skill_docs_index()
            except Exception:
                self.skill_docs = {}
        if _load_skills_catalog is not None:
            try:
                self.skills_catalog = _load_skills_catalog()
            except Exception:
                self.skills_catalog = []
        self._register_skills_from_skill_docs()
        self._skill_doc_cache: Dict[str, str] = {}
        self._skill_resource_cache: Dict[str, str] = {}
        self._skill_py_module_cache: Dict[str, Tuple[float, Any]] = {}
        if self.agent_name not in _EVAL_STATS:
            _EVAL_STATS[self.agent_name] = {
                'decide_calls': 0,
                'decide_json_calls': 0,
                'reflection_text_rounds': 0,
                'structured_parse_failures': 0,
                'json_repair_attempts': 0,
                'json_repair_successes': 0,
            }
    def add_memory(self, message: Dict[str, Any]):
        self.memory.append(message)

    def build_skill_context(self, state: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
        if _SkillContext is None:
            return None
        out_dir = output_dir
        if not out_dir:
            try:
                out_dir = getattr(self, "output_dir", None)
            except Exception:
                out_dir = None
        cfg = config if isinstance(config, dict) else None
        return _SkillContext(state=state or {}, output_dir=out_dir, config=cfg, llm=self.llm, logger=self.logger)

    def describe_skills_text(self) -> str:
        try:
            docs = getattr(self, "skill_docs", None)
            if not isinstance(docs, dict):
                docs = {}
            reg = getattr(self, "skills", None)
            lines: List[str] = []
            if reg is not None:
                for spec in reg.list_specs():
                    doc = docs.get(getattr(spec, "id", "")) if isinstance(docs, dict) else None
                    name = getattr(spec, "name", "")
                    description = getattr(spec, "description", "")
                    if isinstance(doc, dict):
                        dn = doc.get("name")
                        dd = doc.get("description")
                        if dn:
                            name = dn
                        if dd:
                            description = dd
                    lines.append(f"- {spec.id}: {name} | {description}")
            if not lines and isinstance(docs, dict) and docs:
                for sid in sorted(docs.keys()):
                    meta = docs.get(sid)
                    if not isinstance(meta, dict):
                        continue
                    name = str(meta.get("name") or "").strip()
                    description = str(meta.get("description") or "").strip()
                    lines.append(f"- {sid}: {name} | {description}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _skills_metadata_text(self) -> str:
        try:
            catalog = getattr(self, "skills_catalog", None)
            if isinstance(catalog, list) and catalog:
                lines: List[str] = []
                for item in catalog:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    description = str(item.get("description") or "").strip()
                    if not name or not description:
                        continue
                    lines.append(f"- {name} | {description}")
                if lines:
                    return "\n".join(lines)
        except Exception:
            pass
        try:
            s = self.describe_skills_text()
        except Exception:
            s = ""
        return s.strip() if isinstance(s, str) and s.strip() else "（无）"

    def _skills_router_choices_text(self) -> str:
        docs = getattr(self, "skill_docs", None)
        if not isinstance(docs, dict) or not docs:
            return ""
        lines: List[str] = []
        for sid in sorted(docs.keys()):
            meta = docs.get(sid)
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name") or "").strip()
            description = str(meta.get("description") or "").strip()
            lines.append(f"- {sid}: {name} | {description}")
        return "\n".join(lines).strip()

    def _infer_method_name_for_skill_id(self, skill_id: str) -> str:
        sid = str(skill_id or "").strip()
        if not sid:
            return ""
        parts = [p for p in sid.split(".") if p]
        if not parts:
            return ""
        if parts[0] == "data" and len(parts) == 2:
            return f"{parts[1]}_data"
        return parts[-1]

    def _register_skills_from_skill_docs(self) -> None:
        reg = getattr(self, "skills", None)
        if reg is None or _SkillSpecT is None:
            return
        docs = getattr(self, "skill_docs", None)
        if not isinstance(docs, dict) or not docs:
            return

        for sid, meta in docs.items():
            try:
                if reg.has(sid):
                    continue
            except Exception:
                continue
            method_name = self._infer_method_name_for_skill_id(sid)
            if not method_name:
                continue
            method = getattr(self, method_name, None)
            if not callable(method):
                continue

            name = str(meta.get("name") or sid).strip() if isinstance(meta, dict) else sid
            description = str(meta.get("description") or "").strip() if isinstance(meta, dict) else ""
            try:
                spec = _SkillSpecT(id=str(sid), name=name, description=description, inputs={}, outputs={}, tags=())
            except Exception:
                continue
            sig = None
            try:
                sig = inspect.signature(method)
            except Exception:
                sig = None

            def _make_runner(meth, msig):
                def _runner(*, ctx, **kwargs):
                    call_kwargs = dict(kwargs)
                    if msig is not None:
                        try:
                            if "config" in msig.parameters and "config" not in call_kwargs:
                                call_kwargs["config"] = getattr(ctx, "config", None)
                        except Exception:
                            pass
                    return meth(**call_kwargs)

                return _runner

            try:
                reg.register(spec, _make_runner(method, sig))
            except Exception:
                continue

    def _select_skill_ids_for_task(self, task: str, config: Optional[Dict[str, Any]] = None, max_ids: int = 3) -> List[str]:
        docs = getattr(self, "skill_docs", None)
        if not isinstance(docs, dict) or not docs:
            return []
        if JsonOutputParser is None or not self._structured_output_enabled(config):
            return []
        skills_text = self._skills_router_choices_text()
        if not skills_text:
            return []
        parser = JsonOutputParser()
        router_prompt = ChatPromptTemplate.from_template(
            "你是一个技能路由器。根据【任务】从【可用 Skills】选择最相关的 0-3 个 Skill ID。\n"
            "只能从列表中选择；如果不需要任何技能，返回空列表。\n"
            "输出要求：只输出 JSON，不要输出解释文字。\n\n"
            "【可用 Skills】\n{skills}\n\n"
            "【任务】\n{task}\n\n"
            "{format_instructions}\n"
        )
        chain = router_prompt | self.llm | parser
        try:
            result = chain.invoke({"skills": skills_text, "task": str(task or ""), "format_instructions": parser.get_format_instructions()})
        except Exception:
            return []
        if not isinstance(result, dict):
            return []
        raw = result.get("skill_ids", [])
        if not isinstance(raw, list):
            return []
        picked: List[str] = []
        for item in raw:
            sid = str(item or "").strip()
            if not sid:
                continue
            if sid in docs and sid not in picked:
                picked.append(sid)
            if len(picked) >= int(max_ids):
                break
        return picked

    def describe_capabilities_text(self, include_skills: bool = True) -> str:
        try:
            role = str(getattr(self, "role_description", "") or "").strip()
        except Exception:
            role = ""
        caps = None
        try:
            caps = getattr(self, "CAPABILITIES", None)
        except Exception:
            caps = None
        if caps is None:
            try:
                caps = getattr(type(self), "CAPABILITIES", None)
            except Exception:
                caps = None
        lines: List[str] = []
        header = str(getattr(self, "agent_name", "") or type(self).__name__).strip()
        if role:
            lines.append(f"{header}：{role}")
        else:
            lines.append(f"{header}：")
        if isinstance(caps, list):
            for it in caps:
                s = str(it or "").strip()
                if s:
                    lines.append(f"- {s}")
        elif isinstance(caps, dict):
            for k, v in caps.items():
                key = str(k or "").strip()
                if key:
                    lines.append(f"- {key}：")
                if isinstance(v, list):
                    for it in v:
                        s = str(it or "").strip()
                        if s:
                            lines.append(f"  - {s}")
                else:
                    s = str(v or "").strip()
                    if s:
                        lines.append(f"  - {s}")
        elif isinstance(caps, str) and caps.strip():
            for row in str(caps).splitlines():
                s = str(row or "").strip()
                if s:
                    lines.append(f"- {s}")
        if include_skills:
            try:
                skills_text = self.describe_skills_text()
            except Exception:
                skills_text = ""
            if skills_text:
                lines.append("- Skills：")
                for row in str(skills_text).splitlines():
                    s = str(row or "").rstrip()
                    if s:
                        lines.append(f"  {s}")
        return "\n".join(lines).strip()

    def _find_referenced_skill_ids(self, text: str) -> List[str]:
        s = str(text or "")
        if not s:
            return []
        docs = getattr(self, "skill_docs", None)
        if not isinstance(docs, dict) or not docs:
            return []
        found: List[str] = []
        for sid in docs.keys():
            if not sid or "." not in sid:
                continue
            if sid in s:
                found.append(str(sid))
        return sorted(set(found))

    def _load_skill_markdown(self, skill_id: str) -> Optional[str]:
        sid = str(skill_id or "").strip()
        if not sid:
            return None
        cached = self._skill_doc_cache.get(sid)
        if cached is not None:
            return cached
        docs = getattr(self, "skill_docs", None)
        if not isinstance(docs, dict):
            return None
        meta = docs.get(sid)
        if not isinstance(meta, dict):
            return None
        path = meta.get("skill_md")
        if not path:
            return None
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            try:
                with open(str(path), "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except Exception:
                return None
        content_str = str(content or "")
        self._skill_doc_cache[sid] = content_str
        return content_str

    def _extract_markdown_link_targets(self, markdown: str) -> List[str]:
        text = str(markdown or "")
        if not text.strip():
            return []
        targets: List[str] = []
        for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            raw = str(m.group(1) or "").strip()
            if not raw:
                continue
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1].strip()
            first = raw.split()[0].strip().strip('"').strip("'")
            if not first:
                continue
            if first.startswith("#"):
                continue
            low = first.lower()
            if low.startswith("http://") or low.startswith("https://") or low.startswith("mailto:"):
                continue
            targets.append(first)
        for line in text.splitlines():
            ref = re.match(r"^\s*\[[^\]]+\]\s*:\s*(\S+)\s*$", str(line))
            if not ref:
                continue
            raw = str(ref.group(1) or "").strip()
            if not raw:
                continue
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1].strip()
            first = raw.split()[0].strip().strip('"').strip("'")
            if not first:
                continue
            if first.startswith("#"):
                continue
            low = first.lower()
            if low.startswith("http://") or low.startswith("https://") or low.startswith("mailto:"):
                continue
            targets.append(first)
        return sorted(set(targets))

    def _resolve_skill_resource_path(self, skill_id: str, rel_path: str) -> Optional[Tuple[str, str]]:
        sid = str(skill_id or "").strip()
        if not sid:
            return None
        docs = getattr(self, "skill_docs", None)
        if not isinstance(docs, dict):
            return None
        meta = docs.get(sid)
        if not isinstance(meta, dict):
            return None
        base_dir = meta.get("dir")
        skill_md = meta.get("skill_md")
        if not base_dir and skill_md:
            base_dir = os.path.dirname(str(skill_md))
        if not base_dir:
            return None
        root = os.path.abspath(str(base_dir))
        rel = str(rel_path or "").strip().replace("/", os.sep).replace("\\", os.sep)
        if not rel or os.path.isabs(rel):
            return None
        abs_path = os.path.abspath(os.path.normpath(os.path.join(root, rel)))
        try:
            common = os.path.commonpath([root, abs_path])
        except Exception:
            return None
        if os.path.normcase(common) != os.path.normcase(root):
            return None
        if not os.path.isfile(abs_path):
            return None
        _, ext = os.path.splitext(abs_path)
        allow = {".md", ".txt", ".json", ".csv", ".tsv", ".py", ".yaml", ".yml"}
        if ext.lower() not in allow:
            return None
        display_rel = os.path.relpath(abs_path, start=root).replace("\\", "/")
        return display_rel, abs_path

    def _should_load_skill_resource(self, request_text: str, resource_rel: str) -> bool:
        base = str(request_text or "")
        if not base.strip():
            return False
        rel = str(resource_rel or "")
        name = os.path.basename(rel)
        if rel and rel in base:
            return True
        if name and name in base:
            return True
        low = base.lower()
        triggers = ["参考", "详见", "更多", "参考资料", "reference", "forms", "脚本", "示例", "example", "demo", "run", "运行"]
        return any((t.lower() in low for t in triggers))

    def _load_text_file(self, abs_path: str) -> Optional[str]:
        p = os.path.abspath(str(abs_path or ""))
        if not p or not os.path.isfile(p):
            return None
        cached = self._skill_resource_cache.get(p)
        if cached is not None:
            return cached
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            try:
                with open(p, "r", encoding="utf-8-sig") as f:
                    content = f.read()
            except Exception:
                return None
        content_str = str(content or "")
        self._skill_resource_cache[p] = content_str
        return content_str

    def _should_view_skill_script_code(self, request_text: str, script_rel: str) -> bool:
        base = str(request_text or "")
        if not base.strip():
            return False
        rel = str(script_rel or "")
        name = os.path.basename(rel)
        if not rel and not name:
            return False
        if rel and rel not in base and name and name not in base:
            return False
        low = base.lower()
        triggers = ["查看", "打开", "源码", "代码", "内容", "source", "code"]
        return any((t.lower() in low for t in triggers))

    def _should_run_skill_script(self, request_text: str, script_rel: str) -> bool:
        base = str(request_text or "")
        if not base.strip():
            return False
        rel = str(script_rel or "")
        name = os.path.basename(rel)
        if not rel and not name:
            return False
        if rel and rel not in base and name and name not in base:
            return False
        low = base.lower()
        triggers = ["运行", "执行", "run", "execute"]
        return any((t.lower() in low for t in triggers))

    def _parse_script_args_from_request(self, request_text: str, script_rel: str) -> List[str]:
        text = str(request_text or "")
        rel = str(script_rel or "")
        name = os.path.basename(rel)
        candidates = [c for c in (rel, name) if c]
        if not candidates:
            return []
        hit = None
        pos = -1
        for c in candidates:
            p = text.find(c)
            if p >= 0 and (pos < 0 or p < pos):
                pos = p
                hit = c
        if hit is None:
            return []
        tail = text[pos + len(hit) :]
        first_line = tail.splitlines()[0] if tail else ""
        s = str(first_line).strip()
        if not s:
            return []
        try:
            args = shlex.split(s, posix=False)
        except Exception:
            args = s.split()
        raw = [str(a) for a in args if str(a).strip()]
        picked: List[str] = []
        expect_value = False
        for tok in raw:
            t = str(tok)
            if expect_value:
                picked.append(t)
                expect_value = False
                continue
            if t.startswith("-"):
                picked.append(t)
                expect_value = True
                continue
            break
        if picked and picked[-1].startswith("-"):
            picked = picked[:-1]
        return picked

    def _run_python_script(self, abs_path: str, args: List[str], timeout_seconds: int = 20) -> Dict[str, str]:
        p = os.path.abspath(str(abs_path or ""))
        if not p or not os.path.isfile(p):
            return {"ok": "false", "error": "script_not_found"}
        cmd = [sys.executable, p] + [str(a) for a in (args or [])]
        try:
            proc = subprocess.run(
                cmd,
                cwd=None,
                capture_output=True,
                text=True,
                timeout=int(timeout_seconds),
                check=False,
            )
        except Exception as e:
            return {"ok": "false", "error": str(e)}
        out = _redact_text(str(proc.stdout or ""))
        err = _redact_text(str(proc.stderr or ""))
        payload: Dict[str, str] = {"ok": "true" if proc.returncode == 0 else "false", "returncode": str(proc.returncode), "stdout": out}
        if err.strip():
            payload["stderr"] = err
        return payload

    def _load_python_module_from_path(self, abs_path: str) -> Optional[Any]:
        p = os.path.abspath(str(abs_path or ""))
        if not p or not os.path.isfile(p):
            return None
        try:
            mtime = float(os.path.getmtime(p))
        except Exception:
            mtime = -1.0
        cached = self._skill_py_module_cache.get(p)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        module_name = f"_skill_mod_{abs(hash(p))}_{int(mtime) if mtime >= 0 else 0}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, p)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._skill_py_module_cache[p] = (mtime, mod)
            return mod
        except Exception:
            return None

    def _get_skill_tool_callable(self, skill_id: str, rel_py_path: str, attr_name: str) -> Optional[Callable[..., Any]]:
        resolved = self._resolve_skill_resource_path(skill_id, rel_py_path)
        if resolved is None:
            return None
        _, abs_path = resolved
        if not str(abs_path).lower().endswith(".py"):
            return None
        mod = self._load_python_module_from_path(abs_path)
        if mod is None:
            return None
        fn = getattr(mod, str(attr_name or ""), None)
        if callable(fn):
            return fn
        return None

    def _fence_lang_for_path(self, path: str) -> str:
        _, ext = os.path.splitext(str(path or ""))
        e = ext.lower()
        if e == ".py":
            return "python"
        if e in {".yml", ".yaml"}:
            return "yaml"
        if e == ".json":
            return "json"
        if e == ".md":
            return "markdown"
        return "text"

    def _inject_skill_docs_into_task(self, full_task: str, max_total_chars: int = 18000, max_per_skill_chars: int = 8000) -> Tuple[str, List[str]]:
        base = str(full_task or "")
        referenced = self._find_referenced_skill_ids(base)
        if not referenced:
            return base, []
        parts: List[str] = []
        used: List[str] = []
        total = 0
        for sid in referenced:
            md = self._load_skill_markdown(sid)
            if not md:
                continue
            md_full = str(md)
            md_str = md_full
            if len(md_str) > int(max_per_skill_chars):
                md_str = md_str[: int(max_per_skill_chars)] + "\n...(truncated)"
            chunk_parts: List[str] = [f"### {sid}", md_str]

            link_targets = self._extract_markdown_link_targets(md_full)
            loaded_resources: List[str] = []
            resource_blocks: List[str] = []
            resource_total_chars = 0
            max_resources_per_skill = 3
            auto_resources_per_skill = 0
            max_resources_chars_per_skill = 9000
            prioritized_targets = sorted(
                link_targets,
                key=lambda t: (
                    0
                    if (t and (t in base or os.path.basename(t) in base))
                    else 1,
                    len(t or ""),
                ),
            )
            for target in prioritized_targets:
                resolved = self._resolve_skill_resource_path(sid, target)
                if resolved is None:
                    continue
                rel_display, abs_path = resolved
                mentioned = bool(rel_display and (rel_display in base or os.path.basename(rel_display) in base))
                if (
                    not mentioned
                    and len(loaded_resources) >= auto_resources_per_skill
                    and not self._should_load_skill_resource(base, rel_display)
                ):
                    continue
                _, ext = os.path.splitext(rel_display)
                ext_low = ext.lower()
                block = None
                if ext_low == ".py":
                    if self._should_run_skill_script(base, rel_display):
                        args = self._parse_script_args_from_request(base, rel_display)
                        if args:
                            res = self._run_python_script(abs_path, args=args)
                            out = str(res.get("stdout") or "")
                            if len(out) > 6000:
                                out = out[:6000] + "\n...(truncated)"
                            stderr = str(res.get("stderr") or "")
                            if len(stderr) > 3000:
                                stderr = stderr[:3000] + "\n...(truncated)"
                            lines = [f"#### {rel_display}（执行输出）", "```text", out.strip(), "```"]
                            if stderr.strip():
                                lines.extend(["```text", stderr.strip(), "```"])
                            block = "\n".join(lines)
                    elif self._should_view_skill_script_code(base, rel_display):
                        content = self._load_text_file(abs_path)
                        if content is None:
                            continue
                        content_str = str(content)
                        if len(content_str) > 6000:
                            content_str = content_str[:6000] + "\n...(truncated)"
                        block = "\n".join([f"#### {rel_display}", "```python", content_str, "```"])
                else:
                    content = self._load_text_file(abs_path)
                    if content is None:
                        continue
                    content_str = str(content)
                    if len(content_str) > 6000:
                        content_str = content_str[:6000] + "\n...(truncated)"
                    lang = self._fence_lang_for_path(rel_display)
                    block = "\n".join([f"#### {rel_display}", f"```{lang}", content_str, "```"])
                if not block:
                    continue
                if resource_total_chars + len(block) > max_resources_chars_per_skill:
                    continue
                resource_blocks.append(block)
                loaded_resources.append(rel_display)
                resource_total_chars += len(block)
                if len(loaded_resources) >= max_resources_per_skill:
                    break

            if loaded_resources and resource_blocks:
                chunk_parts.append("#### 附加资源（按需加载）")
                chunk_parts.append("\n\n".join(resource_blocks))

            chunk = "\n".join(chunk_parts)
            if total + len(chunk) > int(max_total_chars):
                break
            parts.append(chunk)
            used.append(sid)
            total += len(chunk)
        if not parts:
            return base, []
        injected = base + "\n\n相关技能指南（按需加载）：\n" + "\n\n".join(parts)
        return injected, used
    def _stat_inc(self, key: str, amount: int = 1) -> None:
        try:
            s = _EVAL_STATS.get(self.agent_name)
            if not isinstance(s, dict):
                _EVAL_STATS[self.agent_name] = {}
                s = _EVAL_STATS[self.agent_name]
            current = s.get(key, 0)
            if not isinstance(current, int):
                current = 0
            s[key] = current + int(amount)
        except Exception:
            pass
    def _reflection_enabled(self, config: Optional[Dict[str, Any]]=None) -> bool:
        flag = None
        if isinstance(config, dict):
            flag = config.get('reflection_enabled')
        if flag is not None:
            return bool(flag)
        env_flag = os.environ.get('GEOCHEM_REFLECTION') or os.environ.get('AGENTS_REFLECTION')
        if env_flag is None:
            return True
        env_norm = str(env_flag).strip().lower()
        if env_norm in {'1', 'true', 'yes', 'on'}:
            return True
        if env_norm in {'0', 'false', 'no', 'off'}:
            return False
        return True
    def _reflection_max_rounds(self, config: Optional[Dict[str, Any]]=None) -> int:
        rounds = None
        if isinstance(config, dict):
            rounds = config.get('reflection_max_rounds')
        if rounds is None:
            env_rounds = os.environ.get('GEOCHEM_REFLECTION_MAX_ROUNDS') or os.environ.get('AGENTS_REFLECTION_MAX_ROUNDS')
            rounds = env_rounds
        if isinstance(rounds, int):
            rounds_int = rounds
        elif isinstance(rounds, str):
            try:
                rounds_int = int(rounds.strip())
            except Exception:
                rounds_int = 1
        else:
            rounds_int = 1
        if rounds_int < 0:
            rounds_int = 0
        if rounds_int > 3:
            rounds_int = 3
        return rounds_int
    def _reflect_and_revise_text(self, task: str, draft: str, context: Optional[Dict[str, Any]]=None, llm: Any=None) -> str:
        draft_str = str(draft or '').strip()
        if not draft_str:
            return draft_str
        task_str = str(task or '')
        if len(task_str) > 12000:
            task_str = task_str[:12000] + '...(truncated)'
        if len(draft_str) > 12000:
            draft_str = draft_str[:12000] + '...(truncated)'
        context_str = ''
        if context:
            try:
                context_str = str(context)
            except Exception:
                context_str = ''
            if len(context_str) > 8000:
                context_str = context_str[:8000] + '...(truncated)'
        reflection_prompt = ChatPromptTemplate.from_template(
            '你是一个严格的审稿人（Reflection）。你的任务是检查并修正草稿回答，使其更准确、更完整、更符合要求。\n'
            '要求：\n'
            '1) 只输出最终修正后的回答，不要输出分析过程或反思过程。\n'
            '2) 不要编造不存在的事实；如无法确定，明确说明不确定并给出稳妥的替代建议。\n'
            '3) 保持原有语言风格（中文）。\n\n'
            '【任务】\n{task}\n\n'
            '【上下文（如有）】\n{context}\n\n'
            '【草稿回答】\n{draft}\n'
        )
        chain = reflection_prompt | (llm or self.llm) | StrOutputParser()
        try:
            revised = chain.invoke({'task': task_str, 'context': context_str, 'draft': draft_str})
            revised_str = str(revised or '').strip()
            return revised_str if revised_str else draft_str
        except Exception as e:
            self.logger.warning(f'Reflection revise failed: {e}')
            return draft_str
    def decide(self, task: str, context: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]]=None) -> str:
        self._stat_inc('decide_calls', 1)
        recent_memory = self.memory[-10:] if len(self.memory) > 10 else self.memory
        chat_history = ''
        for msg in recent_memory:
            chat_history += f"{msg['sender']}: {msg['content']}\n"
        full_task = self._build_full_task(task, context, config=config)
        llm = self.llm
        chain = self.system_prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({'agent_name': self.agent_name, 'role_description': self.role_description, 'skills': self._skills_metadata_text(), 'chat_history': chat_history, 'task': full_task})
            final = result
            if self._reflection_enabled(config):
                max_rounds = self._reflection_max_rounds(config)
                if max_rounds > 0:
                    draft = str(final)
                    for _ in range(max_rounds):
                        self._stat_inc('reflection_text_rounds', 1)
                        revised = self._reflect_and_revise_text(task=task, draft=draft, context=context, llm=llm)
                        revised_str = str(revised)
                        if revised_str.strip() == draft.strip():
                            break
                        draft = revised_str
                    final = draft
            return final
        except Exception as e:
            self.logger.error(f'Error in decide: {str(e)}')
            if 'context_length_exceeded' in str(e) or 'string too long' in str(e):
                self.logger.warning('Token overflow detected, trying with minimal context...')
                try:
                    minimal_task = task[:1000]
                    result = chain.invoke({'agent_name': self.agent_name, 'role_description': self.role_description, 'skills': self._skills_metadata_text(), 'chat_history': 'History truncated due to length limit.', 'task': minimal_task + '\n(Context truncated)'})
                    final = result
                    if self._reflection_enabled(config):
                        max_rounds = self._reflection_max_rounds(config)
                        if max_rounds > 0:
                            draft = str(final)
                            for _ in range(max_rounds):
                                self._stat_inc('reflection_text_rounds', 1)
                                revised = self._reflect_and_revise_text(task=task, draft=draft, context=context, llm=llm)
                                revised_str = str(revised)
                                if revised_str.strip() == draft.strip():
                                    break
                                draft = revised_str
                            final = draft
                    return final
                except Exception as e2:
                    self.logger.error(f'Retry failed: {str(e2)}')
                    return f'Error: Unable to process request due to length limits. Original error: {str(e)}'
            raise e
    def _structured_output_enabled(self, config: Optional[Dict[str, Any]]=None) -> bool:
        flag = None
        if isinstance(config, dict):
            flag = config.get('structured_output_enabled')
        if flag is not None:
            return bool(flag)
        env_flag = os.environ.get('GEOCHEM_STRUCTURED_OUTPUT') or os.environ.get('AGENTS_STRUCTURED_OUTPUT')
        if env_flag is None:
            return True
        return str(env_flag).strip().lower() in {'1', 'true', 'yes', 'on'}
    def _build_full_task(self, task: str, context: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]] = None) -> str:
        full_task = task
        if context:
            context_str = ''
            for k, v in context.items():
                v_str = str(v)
                if len(v_str) > 1000:
                    v_str = v_str[:1000] + '...(truncated)'
                context_str += f'{k}: {v_str}\n'
            full_task += f'\n\n相关上下文信息：\n{context_str}'
        referenced = self._find_referenced_skill_ids(full_task)
        if not referenced:
            skill_ids = self._select_skill_ids_for_task(full_task, config=config)
            if skill_ids:
                full_task += "\n\n可用技能建议（自动选择）：\n" + "\n".join([f"- {sid}" for sid in skill_ids])
        full_task, _ = self._inject_skill_docs_into_task(full_task)
        if len(full_task) > 50000:
            self.logger.warning(f'Task content is too long ({len(full_task)} chars), truncating...')
            full_task = full_task[:50000] + '...(truncated)'
        return full_task
    def _decide_json_legacy(self, task: str, default: Dict[str, Any], context: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        import json
        response = self.decide(task, context, config=config)
        try:
            json_match = re.search('\\{.*\\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(default, dict) and isinstance(data, dict):
                    merged = dict(default)
                    merged.update(data)
                    return merged
                return data
        except Exception as e:
            self.logger.warning(f'Legacy JSON parse failed: {e}')
        return default
    def _json_looks_compatible(self, candidate: Any, default: Dict[str, Any]) -> bool:
        if not isinstance(candidate, dict):
            return False
        if not isinstance(default, dict):
            return True
        for k, dv in default.items():
            if k not in candidate:
                return False
            cv = candidate.get(k)
            if dv is None:
                continue
            if isinstance(dv, bool):
                if not isinstance(cv, bool):
                    return False
            elif isinstance(dv, (int, float)):
                if not isinstance(cv, (int, float)) and cv is not None:
                    return False
            elif isinstance(dv, str):
                if not isinstance(cv, str) and cv is not None:
                    return False
            elif isinstance(dv, list):
                if not isinstance(cv, list) and cv is not None:
                    return False
            elif isinstance(dv, dict):
                if not isinstance(cv, dict) and cv is not None:
                    return False
        return True
    def _repair_json_with_reflection(self, task: str, context: Optional[Dict[str, Any]], default: Dict[str, Any], current: Any) -> Dict[str, Any]:
        if JsonOutputParser is None:
            return default
        parser = JsonOutputParser()
        format_instructions = parser.get_format_instructions()
        context_str = ''
        if context:
            try:
                context_str = str(context)
            except Exception:
                context_str = ''
            if len(context_str) > 8000:
                context_str = context_str[:8000] + '...(truncated)'
        repair_prompt = ChatPromptTemplate.from_template(
            '你是一个严格的JSON修复器（Reflection）。请基于原任务与已有输出，生成一个可解析且字段完整的JSON。\n'
            '要求：\n'
            '1) 只输出JSON，不要输出解释文字。\n'
            '2) 必须包含所有必需字段，且字段类型必须合理。\n\n'
            '【原任务】\n{task}\n\n'
            '【上下文（如有）】\n{context}\n\n'
            '【当前输出】\n{current}\n\n'
            '【默认结构（必须至少包含这些字段）】\n{default}\n\n'
            '{format_instructions}\n'
        )
        chain = repair_prompt | self.llm | parser
        try:
            repaired = chain.invoke(
                {
                    'task': str(task or ''),
                    'context': context_str,
                    'current': str(current),
                    'default': str(default),
                    'format_instructions': format_instructions,
                }
            )
            if isinstance(repaired, dict):
                merged = dict(default) if isinstance(default, dict) else {}
                if isinstance(default, dict):
                    merged.update(repaired)
                return merged if merged else repaired
        except Exception as e:
            self.logger.warning(f'Reflection JSON repair failed: {e}')
        return default
    def decide_json(self, task: str, default: Dict[str, Any], context: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        self._stat_inc('decide_json_calls', 1)
        if not self._structured_output_enabled(config) or JsonOutputParser is None:
            return self._decide_json_legacy(task, default, context, config=config)
        parser = JsonOutputParser()
        format_instructions = parser.get_format_instructions()
        full_task = self._build_full_task(task, context, config=config) + '\n\n' + format_instructions
        if len(full_task) > 50000:
            self.logger.warning(f'Task content is too long ({len(full_task)} chars), truncating...')
            full_task = full_task[:50000] + '...(truncated)'
        recent_memory = self.memory[-10:] if len(self.memory) > 10 else self.memory
        chat_history = ''
        for msg in recent_memory:
            chat_history += f"{msg['sender']}: {msg['content']}\n"
        chain = self.system_prompt | self.llm | parser
        try:
            result = chain.invoke({'agent_name': self.agent_name, 'role_description': self.role_description, 'skills': self._skills_metadata_text(), 'chat_history': chat_history, 'task': full_task})
            merged = None
            if isinstance(default, dict) and isinstance(result, dict):
                merged = dict(default)
                merged.update(result)
            candidate = merged if merged is not None else (result if result is not None else default)
            if isinstance(default, dict) and self._reflection_enabled(config):
                max_rounds = self._reflection_max_rounds(config)
                if max_rounds > 0 and not self._json_looks_compatible(candidate, default):
                    self.logger.info('Reflection attempting to repair JSON output')
                    current: Any = candidate
                    last_repaired: Any = candidate
                    for _ in range(max_rounds):
                        self._stat_inc('json_repair_attempts', 1)
                        repaired = self._repair_json_with_reflection(task=task, context=context, default=default, current=current)
                        last_repaired = repaired
                        if self._json_looks_compatible(repaired, default):
                            self._stat_inc('json_repair_successes', 1)
                            return repaired
                        current = repaired
                    if isinstance(last_repaired, dict):
                        return last_repaired
            return candidate
        except Exception as e:
            self.logger.warning(f'Structured output parse failed: {e}')
            self._stat_inc('structured_parse_failures', 1)
            return self._decide_json_legacy(task, default, context, config=config)

    def _cot_enabled(self, config: Optional[Dict[str, Any]] = None) -> bool:
        flag = None
        if isinstance(config, dict):
            flag = config.get("cot_enabled")
        if flag is not None:
            return bool(flag)
        env_flag = os.environ.get("GEOCHEM_COT_ENABLED") or os.environ.get("AGENTS_COT_ENABLED")
        if env_flag is None:
            return True
        return str(env_flag).strip().lower() in {"1", "true", "yes", "on"}

    def decide_json_cot(
        self,
        task: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        max_steps: int = 12,
        default_final: str = "",
    ) -> Dict[str, Any]:
        if max_steps < 1:
            max_steps = 1
        if max_steps > 25:
            max_steps = 25
        default = {"final": str(default_final or ""), "cot_steps": []}
        if not self._cot_enabled(config):
            final_text = self.decide(task, context=context, config=config)
            return {"final": str(final_text or "").strip(), "cot_steps": []}
        cot_task = (
            str(task or "").rstrip()
            + "\n\n"
            + "你必须只输出 JSON（不要输出其他任何文本，不要使用 Markdown 代码块）。\n"
            + "字段要求：\n"
            + "- final: string，给用户/下游使用的最终输出\n"
            + f"- cot_steps: string[]，逐步推理链条，每步一句，最多 {int(max_steps)} 步\n"
            + "硬约束：cot_steps 只描述推理步骤，不要杜撰数据；无法确定就写“不确定”。"
        )
        out = self.decide_json(cot_task, default, context=context, config=config)
        final = out.get("final") if isinstance(out, dict) else ""
        cot_steps = out.get("cot_steps") if isinstance(out, dict) else []
        if not isinstance(final, str):
            final = str(final or "")
        if not isinstance(cot_steps, list):
            cot_steps = []
        cleaned_steps: List[str] = []
        for s in cot_steps[: int(max_steps)]:
            if not isinstance(s, str):
                continue
            ss = s.strip()
            if ss:
                cleaned_steps.append(ss)
        return {"final": str(final or "").strip(), "cot_steps": cleaned_steps}
    def run(self, task: str, context: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        raise NotImplementedError('子类必须实现run方法')
