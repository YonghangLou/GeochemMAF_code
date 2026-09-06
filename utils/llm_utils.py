import os
import logging
import sys
import getpass
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from .token_counter import TokenMonitor
logger = logging.getLogger(__name__)


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            value = str(value).strip()
            if value:
                return value
    return None


def _is_interactive() -> bool:
    try:
        return bool(getattr(sys.stdin, "isatty", lambda: False)())
    except Exception:
        return False


def _prompt_secret(prompt: str) -> str:
    try:
        return getpass.getpass(prompt)
    except Exception:
        try:
            return input(prompt)
        except Exception:
            return ""


def _normalize_provider(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"deepseek", "ds"}:
        return "deepseek"
    if raw in {"qwen", "dashscope"}:
        return "qwen"
    return ""


def _resolve_provider_from_model(model_name: Optional[str]) -> str:
    model = str(model_name or "").strip().lower()
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("qwen"):
        return "qwen"
    if not model:
        return ""
    return "deepseek"


def get_llm(model_name: Optional[str] = None) -> ChatOpenAI:
    token_monitor = TokenMonitor()
    callbacks = [token_monitor.get_callback()]
    configured_provider = _normalize_provider(_first_env("GEOCHEM_LLM_PROVIDER", "AGENTS_LLM_PROVIDER", "LLM_PROVIDER"))
    configured_model = _first_env("GEOCHEM_LLM_MODEL", "AGENTS_LLM_MODEL", "LLM_MODEL")
    if not model_name and configured_model:
        model_name = configured_model
    if not model_name:
        if configured_provider == "deepseek":
            model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            provider = "deepseek"
        else:
            model_name = os.getenv("QWEN_MODEL", "qwen3-max")
            provider = "qwen"
    else:
        provider = _resolve_provider_from_model(model_name)
    if not provider:
        provider = "deepseek"
    pricing_mode = os.getenv("QWEN_PRICING_MODE") if provider == "qwen" else None
    token_monitor.set_pricing_context(provider=provider, model_name=model_name, pricing_mode=pricing_mode)
    if provider == "deepseek":
        api_key = _first_env("DEEPSEEK_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "未设置 DEEPSEEK_API_KEY 环境变量。\n"
                "PowerShell 临时设置示例：$env:DEEPSEEK_API_KEY=\"你的Key\"；\n"
                "或在 .env 中添加：DEEPSEEK_API_KEY=你的Key\n"
                "也可使用 OPENAI_API_KEY 作为 DEEPSEEK_API_KEY 的别名。"
            )
        os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
        base_url = "https://api.deepseek.com"
        logger.info(f"Initializing DeepSeek LLM: {model_name}")
        api_key_secret = SecretStr(api_key)
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key_secret,
            base_url=base_url,
            temperature=0,
            callbacks=callbacks,
        )
    else:
        api_key = _first_env("QWEN_API_KEY", "DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未设置 QWEN_API_KEY 环境变量。\n"
                "PowerShell 临时设置示例：$env:QWEN_API_KEY=\"你的Key\"；\n"
                "或在 .env 中添加：QWEN_API_KEY=你的Key\n"
                "也可使用 DASHSCOPE_API_KEY 作为 QWEN_API_KEY 的别名。"
            )
        os.environ.setdefault("QWEN_API_KEY", api_key)
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        logger.info(f"Initializing Qwen LLM: {model_name}")
        api_key_secret = SecretStr(api_key)
        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key_secret,
            base_url=base_url,
            temperature=0,
            callbacks=callbacks,
        )
    return llm
