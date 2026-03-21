"""Pydantic models for validating LLM JSON responses.

All three response types share sentiment_delta and uncertainty_penalty.
Parse failures return None — callers must handle None and skip the trade.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LightScanResponse(BaseModel):
    """Response from the light batch scan prompt."""

    sentiment_delta: float = Field(ge=-1.0, le=1.0)
    uncertainty_penalty: float = Field(ge=0.0, le=1.0)
    rationale: str


class DeepTriggerResponse(BaseModel):
    """Response from the deep trigger analysis prompt."""

    sentiment_delta: float = Field(ge=-1.0, le=1.0)
    uncertainty_penalty: float = Field(ge=0.0, le=1.0)
    key_factors: list[str] = Field(default_factory=list)
    rationale: str


class ExitResponse(BaseModel):
    """Response from the exit decision prompt."""

    decision: Literal["HOLD", "TRAIL-SELL", "FULL-EXIT"]
    trailing_stop_pct: float = Field(ge=0.0, le=1.0)
    target_price: float = Field(gt=0.0)
    rationale: str


# ---------------------------------------------------------------------------
# JSON Schema generators for OpenRouter structured output mode
# ---------------------------------------------------------------------------

def light_scan_schema() -> dict:
    """JSON Schema for LightScanResponse (OpenRouter response_format)."""
    return LightScanResponse.model_json_schema()


def deep_trigger_schema() -> dict:
    """JSON Schema for DeepTriggerResponse (OpenRouter response_format)."""
    return DeepTriggerResponse.model_json_schema()
