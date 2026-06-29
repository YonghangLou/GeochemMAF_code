from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class SkillSpec:
    id: str
    name: str
    description: str
    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()


@dataclass
class SkillArtifact:
    kind: str
    path: Optional[str] = None
    description: Optional[str] = None
    payload: Optional[Any] = None


@dataclass
class SkillResult:
    ok: bool
    output: Any = None
    artifacts: List[SkillArtifact] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SkillContext:
    state: Dict[str, Any]
    output_dir: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    llm: Any = None
    logger: Any = None


SkillFn = Callable[..., Any]


@dataclass(frozen=True)
class _RegisteredSkill:
    spec: SkillSpec
    fn: SkillFn


class SkillRegistry:
    def __init__(self, owner: str = "") -> None:
        self._owner = str(owner or "")
        self._skills: Dict[str, _RegisteredSkill] = {}

    @property
    def owner(self) -> str:
        return self._owner

    def register(self, spec: SkillSpec, fn: SkillFn) -> None:
        sid = str(spec.id or "").strip()
        if not sid:
            raise ValueError("SkillSpec.id cannot be empty")
        if sid in self._skills:
            return
        self._skills[sid] = _RegisteredSkill(spec=spec, fn=fn)

    def has(self, skill_id: str) -> bool:
        return str(skill_id or "").strip() in self._skills

    def get(self, skill_id: str) -> SkillSpec:
        sid = str(skill_id or "").strip()
        if sid not in self._skills:
            raise KeyError(f"Skill not found: {sid}")
        return self._skills[sid].spec

    def list_specs(self) -> List[SkillSpec]:
        return [v.spec for _, v in sorted(self._skills.items(), key=lambda kv: kv[0])]

    def describe_text(self) -> str:
        lines: List[str] = []
        for spec in self.list_specs():
            lines.append(f"- {spec.id}: {spec.name} | {spec.description}")
        return "\n".join(lines)

    def run(self, skill_id: str, ctx: SkillContext, **kwargs: Any) -> SkillResult:
        sid = str(skill_id or "").strip()
        if sid not in self._skills:
            return SkillResult(ok=False, error=f"Skill not found: {sid}")
        reg = self._skills[sid]
        start = time.perf_counter()
        try:
            out = reg.fn(ctx=ctx, **kwargs)
            elapsed = float(time.perf_counter() - start)
            metrics = {"elapsed_seconds": elapsed, "skill_id": sid, "owner": self._owner}
            return SkillResult(ok=True, output=out, metrics=metrics)
        except Exception as e:
            elapsed = float(time.perf_counter() - start)
            if getattr(ctx, "logger", None) is not None:
                try:
                    ctx.logger.exception(f"Skill execution failed: {sid} ({self._owner}) | {e}")
                except Exception:
                    pass
            metrics = {"elapsed_seconds": elapsed, "skill_id": sid, "owner": self._owner}
            return SkillResult(ok=False, error=str(e), metrics=metrics)

    def run_into_state(self, skill_id: str, ctx: SkillContext, state_key: str, **kwargs: Any) -> SkillResult:
        res = self.run(skill_id, ctx, **kwargs)
        if res.ok:
            try:
                ctx.state[state_key] = res.output
            except Exception:
                pass
        return res

    def merge(self, other: "SkillRegistry") -> None:
        for spec in other.list_specs():
            sid = spec.id
            if sid in self._skills:
                continue
            fn = other._skills[sid].fn
            self._skills[sid] = _RegisteredSkill(spec=spec, fn=fn)

    def __len__(self) -> int:
        return len(self._skills)
