from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .core import SkillContext, SkillRegistry


def _split_modules(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    parts = []
    for item in str(raw).split(","):
        name = str(item).strip()
        if name:
            parts.append(name)
    return parts


def load_skills_from_modules(registry: SkillRegistry, module_names: Iterable[str], ctx: Optional[SkillContext] = None) -> Tuple[int, List[str]]:
    loaded = 0
    errors: List[str] = []
    for module_name in module_names:
        name = str(module_name or "").strip()
        if not name:
            continue
        try:
            mod = importlib.import_module(name)
        except Exception as e:
            errors.append(f"{name}: import_failed: {e}")
            continue
        try:
            register = getattr(mod, "register_skills", None)
            if callable(register):
                register(registry, ctx)
                loaded += 1
                continue
            exports = getattr(mod, "SKILL_EXPORTS", None)
            if isinstance(exports, SkillRegistry):
                registry.merge(exports)
                loaded += 1
                continue
            errors.append(f"{name}: no_register_skills")
        except Exception as e:
            errors.append(f"{name}: register_failed: {e}")
            continue
    return loaded, errors


def load_skills_from_env(registry: SkillRegistry, ctx: Optional[SkillContext] = None, env_name: str = "GEOCHEM_SKILL_MODULES") -> Tuple[int, List[str]]:
    return load_skills_from_modules(registry, _split_modules(os.environ.get(env_name)), ctx=ctx)


def _find_skill_md_in_dir(dir_path: str) -> Optional[str]:
    try:
        entries = os.listdir(dir_path)
    except Exception:
        return None
    for name in entries:
        if str(name).lower() == "skill.md":
            return os.path.join(dir_path, name)
    return None


def _parse_yaml_frontmatter(lines: List[str]) -> Tuple[Dict[str, Any], int]:
    i = 0
    while i < len(lines) and not str(lines[i]).strip():
        i += 1
    if i >= len(lines) or str(lines[i]).strip() != "---":
        return {}, 0
    i += 1
    yaml_lines: List[str] = []
    while i < len(lines):
        raw = str(lines[i]).rstrip("\n")
        if raw.strip() == "---":
            try:
                import yaml  # type: ignore

                parsed = yaml.safe_load("\n".join(yaml_lines)) or {}
                return parsed if isinstance(parsed, dict) else {}, i + 1
            except Exception:
                return {}, i + 1
        yaml_lines.append(raw)
        i += 1
    return {}, 0


def _skill_ids_from_meta(meta: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("id", "skill_id"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip() and "." in val:
            out.append(val.strip())
    for key in ("ids", "skill_ids", "skills"):
        val = meta.get(key)
        if isinstance(val, str):
            raw = val.strip()
            if raw and "." in raw:
                out.append(raw)
        elif isinstance(val, list):
            for item in val:
                s = str(item or "").strip()
                if s and "." in s:
                    out.append(s)
    seen: List[str] = []
    for sid in out:
        if any(ch.isspace() for ch in sid):
            continue
        if sid not in seen:
            seen.append(sid)
    return seen


def _extract_skill_ids(lines: List[str], start_index: int) -> List[str]:
    ids: List[str] = []
    for raw in lines[start_index:]:
        s = str(raw).strip()
        if not s.startswith("-"):
            continue
        item = s.lstrip("-").strip()
        if not item or "." not in item:
            continue
        if any(ch.isspace() for ch in item):
            continue
        ids.append(item)
    return ids


def load_skill_docs_index(root_dir: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    base_dir = root_dir or os.path.dirname(__file__)
    index: Dict[str, Dict[str, str]] = {}
    try:
        entries = os.listdir(base_dir)
    except Exception:
        return index
    for entry in entries:
        dir_path = os.path.join(base_dir, entry)
        if not os.path.isdir(dir_path):
            continue
        if str(entry).startswith("__") or str(entry).startswith("."):
            continue
        skill_md = _find_skill_md_in_dir(dir_path)
        if not skill_md:
            continue
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except Exception:
            continue
        meta, body_start = _parse_yaml_frontmatter(lines)
        if body_start <= 0:
            continue
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        if not name or not description:
            continue
        skill_ids = _skill_ids_from_meta(meta) or _extract_skill_ids(lines, body_start)
        if not skill_ids:
            continue
        for sid in skill_ids:
            if sid in index:
                continue
            index[sid] = {
                "name": name,
                "description": description,
                "dir": os.path.abspath(dir_path),
                "skill_md": os.path.abspath(skill_md),
            }
    return index


def load_skills_catalog(root_dir: Optional[str] = None) -> List[Dict[str, str]]:
    base_dir = root_dir or os.path.dirname(__file__)
    out: List[Dict[str, str]] = []
    try:
        entries = os.listdir(base_dir)
    except Exception:
        return out
    for entry in entries:
        dir_path = os.path.join(base_dir, entry)
        if not os.path.isdir(dir_path):
            continue
        if str(entry).startswith("__") or str(entry).startswith("."):
            continue
        skill_md = _find_skill_md_in_dir(dir_path)
        if not skill_md:
            continue
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except Exception:
            continue
        meta, body_start = _parse_yaml_frontmatter(lines)
        if body_start <= 0:
            continue
        name = str(meta.get("name") or "").strip()
        description = str(meta.get("description") or "").strip()
        if not name or not description:
            continue
        out.append(
            {
                "name": name,
                "description": description,
                "dir": os.path.abspath(dir_path),
                "skill_md": os.path.abspath(skill_md),
            }
        )
    out.sort(key=lambda item: (str(item.get("name") or ""), str(item.get("dir") or "")))
    return out
