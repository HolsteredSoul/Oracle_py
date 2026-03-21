"""Prompt template builders for all three LLM call types.

Each function returns a complete prompt string with embedded JSON-only instruction.
The system message is prepended inline so callers pass a single string to call_llm().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.enrichment.stats import MatchStats


_SYSTEM_LIGHT = (
    "You are an expert prediction-market trader. "
    "Output JSON only — no prose, no markdown, no code fences."
)

_SYSTEM_DEEP = (
    "You are an expert prediction-market trader. Think step-by-step before answering. "
    "Output JSON only — no prose, no markdown, no code fences."
)

_SYSTEM_EXIT = (
    "You are an expert prediction-market trader managing an open position. "
    "Output JSON only — no prose, no markdown, no code fences."
)


def build_light_scan_prompt(
    question: str,
    mid_price: float,
    news_summary: str,
    runner_name: str = "",
    market_type: str = "",
    stats_context: str = "",
    model_probability: float | None = None,
) -> str:
    """Build a light batch scan prompt.

    Args:
        question: The prediction market question text.
        mid_price: Current market mid-price (0–1 probability).
        news_summary: Recent headlines/descriptions as a single string.
        runner_name: Specific selection being priced (e.g. "Sydney Swans").
        market_type: Betfair market type (e.g. "MATCH_ODDS", "WINNER").
        stats_context: Pre-formatted statistical context string (form, H2H).
        model_probability: Statistical model's estimated probability for this selection.

    Returns:
        Full prompt string to pass to call_llm(tier="fast").
    """
    selection_ctx = ""
    if runner_name:
        selection_ctx = f"\nSelection being priced: {runner_name}"
    if market_type:
        selection_ctx += f"\nMarket type: {market_type}"

    # Reference probability: use model estimate if available, else mid_price
    ref_price = model_probability if model_probability is not None else mid_price

    selection_label = runner_name or "YES"
    direction_instruction = (
        f"IMPORTANT — delta sign convention for THIS selection:\n"
        f"  sentiment_delta > 0  means '{selection_label}' is MORE likely than {ref_price:.3f} implies.\n"
        f"  sentiment_delta < 0  means '{selection_label}' is LESS likely than {ref_price:.3f} implies.\n"
        f"  sentiment_delta = 0  means the estimate is approximately fair.\n"
        f"Example: if the market shows Under 7.5 Goals at 9.5% but you think it should be ~95%, "
        f"return a POSITIVE delta (the Under selection is more likely than priced)."
    )

    # Build stats block if available
    stats_block = ""
    if model_probability is not None:
        stats_block = f"\nStatistical model estimate: {model_probability:.3f}"
        if stats_context:
            stats_block += f"\n{stats_context}"
        stats_block += (
            "\n\nYour role: identify qualitative factors (injuries, team news, travel, "
            "weather, morale) that the statistical model CANNOT capture. Only return a "
            "non-zero sentiment_delta if you have specific information that shifts the "
            "probability AWAY from the statistical estimate."
        )

    # When no stats are available, fall back to original reasoning instructions
    reasoning_block = ""
    if model_probability is None:
        reasoning_block = f"""

Reason about whether the current mid-price looks fair or mispriced. Consider:
- Your knowledge of this sport, team, or event
- Known market biases (e.g. longshot bias in racing: punters overbid longshots, underbid favourites)
- Whether {mid_price:.3f} is plausible given the number of competitors and typical outcomes
- Any relevant context from the news above
Only return sentiment_delta=0 if you genuinely have no view at all."""

    return f"""{_SYSTEM_LIGHT}

Market question: {question}{selection_ctx}
Current mid-price (implied probability): {mid_price:.3f}{stats_block}
Recent news: {news_summary or "No recent news available."}

{direction_instruction}{reasoning_block}

Output this JSON structure exactly:
{{
  "sentiment_delta": <float -1.0 to 1.0>,
  "uncertainty_penalty": <float 0.0 to 1.0>,
  "rationale": "<one sentence>"
}}"""


def build_deep_trigger_prompt(
    question: str,
    mid_price: float,
    news_summary: str,
    x_summary: str,
    runner_name: str = "",
    market_type: str = "",
    stats_context: str = "",
    model_probability: float | None = None,
) -> str:
    """Build a deep trigger analysis prompt.

    Args:
        question: The prediction market question text.
        mid_price: Current market mid-price (0–1 probability).
        news_summary: Recent news headlines as a single string.
        x_summary: Recent X/Twitter posts as a single string.
        runner_name: Specific selection being priced (e.g. "Sydney Swans").
        market_type: Betfair market type (e.g. "MATCH_ODDS", "WINNER").
        stats_context: Pre-formatted statistical context string.
        model_probability: Statistical model's estimated probability.

    Returns:
        Full prompt string to pass to call_llm(tier="deep").
    """
    selection_ctx = ""
    if runner_name:
        selection_ctx = f"\nSelection being priced: {runner_name}"
    if market_type:
        selection_ctx += f"\nMarket type: {market_type}"

    ref_price = model_probability if model_probability is not None else mid_price

    selection_label = runner_name or "YES"
    direction_instruction = (
        f"IMPORTANT — delta sign convention for THIS selection:\n"
        f"  sentiment_delta > 0  means '{selection_label}' is MORE likely than {ref_price:.3f} implies.\n"
        f"  sentiment_delta < 0  means '{selection_label}' is LESS likely than {ref_price:.3f} implies.\n"
        f"  sentiment_delta = 0  means the estimate is approximately fair.\n"
        f"Example: if the market shows Under 7.5 Goals at 9.5% but you think it should be ~95%, "
        f"return a POSITIVE delta (the Under selection is more likely than priced)."
    )

    stats_block = ""
    if model_probability is not None:
        stats_block = f"\nStatistical model estimate: {model_probability:.3f}"
        if stats_context:
            stats_block += f"\n{stats_context}"

    return f"""{_SYSTEM_DEEP}

Market question: {question}{selection_ctx}
Current mid-price (implied probability): {mid_price:.3f}{stats_block}
Recent news: {news_summary or "No recent news available."}
X/Twitter sentiment: {x_summary or "No X data available."}

{direction_instruction}

Step-by-step:
1. Synthesize all news and X/Twitter signals.
2. Consider your prior knowledge of this sport, team, or competition.
3. Estimate whether {ref_price:.3f} is fair, too high, or too low.
4. Estimate the directional impact and its magnitude.
5. Identify the key uncertainties.

Output this JSON structure exactly:
{{
  "sentiment_delta": <float -1.0 to 1.0>,
  "uncertainty_penalty": <float 0.0 to 1.0>,
  "key_factors": ["<factor1>", "<factor2>"],
  "rationale": "<two to three sentences>"
}}"""


def build_exit_prompt(
    position: dict,
    updated_news: str,
    current_price: float,
) -> str:
    """Build an exit decision prompt.

    Args:
        position: Dict with keys: market_id, question, direction, entry_price,
                  current_pnl_pct, holding_hours.
        updated_news: Latest news summary as a single string.
        current_price: Current market mid-price (0–1 probability).

    Returns:
        Full prompt string to pass to call_llm(tier="deep").
    """
    question = position.get("question", "Unknown market")
    direction = position.get("direction", "unknown")
    entry_price = position.get("entry_price", 0.0)
    pnl_pct = position.get("current_pnl_pct", 0.0)
    hours = position.get("holding_hours", 0)

    return f"""{_SYSTEM_EXIT}

Market question: {question}
Position: {direction.upper()} entered at {entry_price:.3f}
Current price: {current_price:.3f}
Unrealised P&L: {pnl_pct:+.1f}%
Holding period: {hours:.1f} hours
Updated news: {updated_news or "No new information."}

Decide whether to hold or exit this position.

Output this JSON structure exactly:
{{
  "decision": "HOLD" | "TRAIL-SELL" | "FULL-EXIT",
  "trailing_stop_pct": <float 0.0 to 1.0>,
  "target_price": <float 0.0 to 1.0>,
  "rationale": "<one to two sentences>"
}}"""


def format_stats_context(stats: MatchStats) -> str:
    """Format MatchStats into a human-readable context block for LLM prompts."""
    lines = ["Statistical context:"]

    if stats.home_form_pts_per_game is not None:
        lines.append(f"  Home form: {stats.home_form_pts_per_game:.1f} pts/game (last 5)")
    if stats.away_form_pts_per_game is not None:
        lines.append(f"  Away form: {stats.away_form_pts_per_game:.1f} pts/game (last 5)")
    if stats.home_goals_scored_avg is not None:
        lines.append(f"  Home goals scored avg: {stats.home_goals_scored_avg:.2f}")
    if stats.home_goals_conceded_avg is not None:
        lines.append(f"  Home goals conceded avg: {stats.home_goals_conceded_avg:.2f}")
    if stats.away_goals_scored_avg is not None:
        lines.append(f"  Away goals scored avg: {stats.away_goals_scored_avg:.2f}")
    if stats.away_goals_conceded_avg is not None:
        lines.append(f"  Away goals conceded avg: {stats.away_goals_conceded_avg:.2f}")
    if stats.home_league_position is not None:
        lines.append(f"  Home league position: {stats.home_league_position}")
    if stats.away_league_position is not None:
        lines.append(f"  Away league position: {stats.away_league_position}")

    h2h_total = stats.h2h_home_wins + stats.h2h_draws + stats.h2h_away_wins
    if h2h_total > 0:
        lines.append(
            f"  H2H record: {stats.h2h_home_wins}W-{stats.h2h_draws}D-{stats.h2h_away_wins}L "
            f"({h2h_total} matches)"
        )

    return "\n".join(lines)
