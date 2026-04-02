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
    perplexity_model: str = "perplexity/sonar"
    daily_cap_usd: float = Field(gt=0)
    downgrade_threshold_pct: float = Field(gt=0, le=1.0)
    use_structured_output: bool = False


class TriggersConfig(BaseModel):
    """Event-driven trigger thresholds."""

    news_sentiment_delta: float = Field(gt=0, le=1.0)
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
    kelly_hard_cap: float = Field(default=0.10, gt=0, le=1.0)
    lay_max_pnl_pct: float = Field(default=0.15, gt=0, le=1.0)
    paper_max_cost_aud: float = Field(default=100.0, gt=0)
    # Realism parameters
    min_market_liquidity_aud: float = Field(default=50.0, ge=0)
    min_matched_volume_aud: float = Field(default=500.0, ge=0)
    max_lay_probability: float = Field(default=0.90, gt=0, le=1.0)
    slippage_model: str = Field(default="linear")
    slippage_factor: float = Field(default=0.10, ge=0)
    allow_in_play: bool = False
    reject_crossed_book: bool = True
    max_new_positions_per_cycle: int = Field(default=3, ge=1)


class ScannerConfig(BaseModel):
    """Market scanner settings."""

    poll_interval_sec: int = Field(gt=0)
    prob_range: Tuple[float, float]
    betfair_event_types: list[int] = Field(default_factory=list)
    betfair_country_codes: list[str] = Field(default_factory=lambda: ["AU"])
    betfair_hours_ahead: int = Field(default=6, gt=0)
    betfair_max_market_age_hours: int = Field(default=168, ge=0)
    # Adaptive scheduling
    max_markets_per_cycle: int = Field(default=25, gt=0)
    min_interval_min: int = Field(default=10, gt=0)
    max_interval_min: int = Field(default=45, gt=0)
    rejection_cache_ttl_min: int = Field(default=120, ge=0)
    # Don't trade markets more than N hours from kickoff (scan only)
    betfair_min_hours_before_start: float = Field(default=0, ge=0)
    # Competition whitelist per sport (Betfair competition IDs)
    competition_ids_football: list[int] = Field(default_factory=list)
    competition_ids_baseball: list[int] = Field(default_factory=list)
    competition_ids_basketball: list[int] = Field(default_factory=list)
    competition_ids_hockey: list[int] = Field(default_factory=list)
    competition_ids_afl: list[int] = Field(default_factory=list)
    competition_ids_rugby_union: list[int] = Field(default_factory=list)
    competition_ids_rugby_league: list[int] = Field(default_factory=list)

    @field_validator("prob_range")
    @classmethod
    def validate_prob_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        low, high = v
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                f"prob_range must satisfy 0 <= low < high <= 1, got ({low}, {high})"
            )
        return v


class StatsConfig(BaseModel):
    """Statistical data source configuration."""

    football_api: str = "football-data"
    football_api_base: str = "https://api.football-data.org/v4"
    afl_api_base: str = "https://api.squiggle.com.au"
    basketball_api_base: str = "https://v1.basketball.api-sports.io"
    mlb_api_base: str = "https://statsapi.mlb.com/api/v1"
    rugby_api_base: str = "https://v1.rugby.api-sports.io"
    nhl_api_base: str = "https://api-web.nhle.com/v1"
    cricket_api_base: str = "https://api.cricapi.com/v1"
    nrl_api_base: str = "https://www.thesportsdb.com/api/v1/json/3"
    cache_ttl_hours: int = Field(default=6, gt=0)
    min_data_completeness: float = Field(default=0.5, ge=0, le=1.0)


class PaperConfig(BaseModel):
    """Paper-trading realism simulation parameters."""

    queue_position_model: str = Field(default="probabilistic")
    queue_factor: float = Field(default=0.50, ge=0, le=2.0)
    adverse_drift_enabled: bool = False
    adverse_drift_base_sigma: float = Field(default=0.003, ge=0)
    track_fill_rates: bool = True

    @field_validator("queue_position_model")
    @classmethod
    def validate_queue_model(cls, v: str) -> str:
        allowed = {"none", "linear", "probabilistic"}
        if v not in allowed:
            raise ValueError(f"queue_position_model must be one of {allowed}, got {v!r}")
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
    stats: StatsConfig = StatsConfig()
    paper: PaperConfig = PaperConfig()
    betfair: BetfairConfig

    # Secrets loaded from environment variables
    openrouter_api_key: str = ""
    newsdata_api_key: str = ""
    football_data_api_key: str = ""
    basketball_api_key: str = ""
    cricket_api_key: str = ""
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
