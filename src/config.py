"""Configuration loader for Oracle Python Agent.

Loads settings from config.toml and .env, validates all values at startup,
and fails fast with clear errors if required keys are missing or invalid.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

# Project root is two levels up from this file (src/config.py -> oracle-py/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMConfig(BaseModel):
    """LLM routing and cost configuration."""

    fast_model: str
    deep_model: str
    daily_cap_usd: float = Field(gt=0)
    downgrade_threshold_pct: float = Field(gt=0, le=1.0)


class TriggersConfig(BaseModel):
    """Event-driven trigger thresholds."""

    news_sentiment_delta: float = Field(gt=0, le=1.0)
    x_momentum: float = Field(gt=0, le=1.0)
    volatility_z: float = Field(gt=0)
    margin_min_paper: float = Field(gt=0, le=1.0)
    margin_min_live: float = Field(gt=0, le=1.0)


class RiskConfig(BaseModel):
    """Risk management parameters."""

    max_exposure: float = Field(gt=0, le=1.0)
    drawdown_throttle_pct: float = Field(ge=0, le=1.0)
    drawdown_throttle_factor: float = Field(gt=0, le=1.0)
    confidence_floor: int = Field(ge=0, le=100)
    commission_pct: float = Field(ge=0, le=1.0)
    beta: float = Field(ge=0)
    liquidity_safety_factor: float = Field(gt=0, le=1.0)
    kelly_base_fraction: float = Field(gt=0, le=1.0)


class ScannerConfig(BaseModel):
    """Market scanner settings."""

    poll_interval_sec: int = Field(gt=0)
    manifold_min_volume: int = Field(ge=0)
    manifold_min_prob_range: Tuple[float, float]

    @field_validator("manifold_min_prob_range")
    @classmethod
    def validate_prob_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        low, high = v
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                f"manifold_min_prob_range must satisfy 0 <= low < high <= 1, got ({low}, {high})"
            )
        return v


class BetfairConfig(BaseModel):
    """Betfair exchange settings (Phase 6 only)."""

    api_key: str = ""
    session_token: str = ""
    min_bet_aud: float = Field(ge=0, default=2.0)
    synthetic_tif_sec: int = Field(gt=0, default=300)


class Settings(BaseSettings):
    """Top-level application settings.

    Loads structured config from config.toml and secrets from environment
    variables (via .env file).
    """

    # Sections loaded from config.toml
    llm: LLMConfig
    triggers: TriggersConfig
    risk: RiskConfig
    scanner: ScannerConfig
    betfair: BetfairConfig

    # Secrets loaded from environment variables
    openrouter_api_key: str = ""
    newsdata_api_key: str = ""
    x_bearer_token: str = ""
    betfair_username: str = ""
    betfair_password: str = ""
    betfair_app_key: str = ""
    betfair_certs_path: str = ""

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}


def load_settings(config_path: Path | None = None) -> Settings:
    """Load and validate all settings from config.toml and .env.

    Args:
        config_path: Path to config.toml. Defaults to PROJECT_ROOT/config.toml.

    Returns:
        Fully validated Settings instance.

    Raises:
        SystemExit: If config.toml is missing or contains invalid values.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.toml"

    if not config_path.exists():
        print(f"FATAL: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load .env for environment variables
    load_dotenv(PROJECT_ROOT / ".env")

    try:
        raw = toml.load(config_path)
    except toml.TomlDecodeError as e:
        print(f"FATAL: Failed to parse {config_path}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        return Settings(**raw)
    except Exception as e:
        print(f"FATAL: Config validation failed: {e}", file=sys.stderr)
        sys.exit(1)


# Module-level singleton — import `settings` from anywhere
settings = load_settings()
