"""OpenRouter LLM client with cost tracking, tier routing, and retry.

Public API:
    call_llm(prompt, tier="fast") -> dict | None

Tier routing:
    "fast"  -> config.llm.fast_model   (80% of calls, cheap)
    "deep"  -> config.llm.deep_model   (triggered analysis only)

Cost tracking:
    Daily spend is persisted to state/llm_spend.json.
    When spend reaches downgrade_threshold_pct of daily_cap_usd,
    both tiers are downgraded to the fast model.
    When spend reaches daily_cap_usd, all calls return None.

Retry:
    tenacity exponential backoff on HTTP 429 / 500 / 503 and network errors.
    Up to 3 attempts, wait 2s→4s→8s.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_SPEND_FILE = PROJECT_ROOT / "state" / "llm_spend.json"
_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Spend tracking
# ---------------------------------------------------------------------------

def _load_spend() -> dict[str, Any]:
    if _SPEND_FILE.exists():
        try:
            return json.loads(_SPEND_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_spend(data: dict[str, Any]) -> None:
    _SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SPEND_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_SPEND_FILE)


def _get_today_spend() -> float:
    today = str(date.today())
    return _load_spend().get(today, 0.0)


def _add_spend(amount: float) -> None:
    today = str(date.today())
    data = _load_spend()
    data[today] = round(data.get(today, 0.0) + amount, 6)
    _save_spend(data)
    cap = settings.llm.daily_cap_usd
    threshold = cap * settings.llm.downgrade_threshold_pct
    if data[today] >= cap:
        logger.warning("LLM daily cap reached ($%.2f). Calls will be skipped.", cap)
    elif data[today] >= threshold:
        logger.warning(
            "LLM spend $%.4f is at %.0f%% of daily cap — downgrading to fast model.",
            data[today],
            settings.llm.downgrade_threshold_pct * 100,
        )


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _choose_model(tier: str) -> str:
    """Return the model to use, applying downgrade logic if near cap."""
    spend = _get_today_spend()
    cap = settings.llm.daily_cap_usd
    threshold = cap * settings.llm.downgrade_threshold_pct

    if spend >= threshold:
        # Downgrade both tiers to fast model to preserve budget
        return settings.llm.fast_model

    if tier == "deep":
        return settings.llm.deep_model
    return settings.llm.fast_model


# ---------------------------------------------------------------------------
# HTTP retry logic
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 503)
    return isinstance(exc, httpx.RequestError)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_openrouter(
    model: str,
    prompt: str,
    response_schema: dict | None = None,
) -> httpx.Response:
    """Make a single HTTP call to OpenRouter. Retried by tenacity."""
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/HolsteredSoul/Oracle_py",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Structured output mode: enforce JSON Schema compliance
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "oracle_response",
                "strict": True,
                "schema": response_schema,
            },
        }

    with httpx.Client(timeout=_TIMEOUT) as client:
        response = client.post(_OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    tier: str = "fast",
    response_schema: dict | None = None,
) -> dict | None:
    """Call OpenRouter and return parsed JSON dict.

    Args:
        prompt: Full prompt string (system instruction embedded).
        tier: "fast" or "deep". Controls model selection and cost.
        response_schema: Optional JSON Schema dict. When provided and
            use_structured_output is enabled, OpenRouter enforces schema
            compliance. Falls back to prompt-based JSON on 400 errors.

    Returns:
        Parsed dict if the LLM returns valid JSON, else None.
        Returns None immediately if no API key is configured or daily cap hit.
    """
    if not settings.openrouter_api_key:
        logger.debug("No OpenRouter API key configured; skipping LLM call.")
        return None

    spend = _get_today_spend()
    if spend >= settings.llm.daily_cap_usd:
        logger.warning("LLM daily cap hit ($%.2f); skipping call.", settings.llm.daily_cap_usd)
        return None

    model = _choose_model(tier)

    # Only use structured output when enabled in config
    schema = response_schema if settings.llm.use_structured_output else None

    logger.debug(
        "LLM call | tier=%s model=%s spend_today=$%.4f structured=%s",
        tier, model, spend, schema is not None,
    )

    try:
        response = _call_openrouter(model, prompt, response_schema=schema)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 400 and schema is not None:
            # Model may not support structured output — retry without schema
            logger.warning(
                "Structured output rejected by %s (400) — retrying without schema.",
                model,
            )
            try:
                response = _call_openrouter(model, prompt, response_schema=None)
            except (httpx.HTTPStatusError, httpx.RequestError) as retry_exc:
                logger.error("LLM retry without schema also failed: %s", retry_exc)
                return None
        else:
            logger.error("LLM HTTP error %d after retries: %s", exc.response.status_code, exc)
            return None
    except httpx.RequestError as exc:
        logger.error("LLM network error after retries: %s", exc)
        return None

    data = response.json()

    # Track cost (OpenRouter returns "cost" in usage)
    cost: float = (data.get("usage") or {}).get("cost") or 0.0
    if cost:
        _add_spend(cost)
        logger.debug("LLM call cost: $%.6f | daily total: $%.4f", cost, _get_today_spend())

    # Extract content
    try:
        content: str = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected OpenRouter response structure: %s | raw: %s", exc, str(data)[:300])
        return None

    # Parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON content: %s", content[:300])
        return None
