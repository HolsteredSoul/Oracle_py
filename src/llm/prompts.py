"""Prompt template builders for all three LLM call types.

Each function returns a complete prompt string with embedded JSON-only instruction.
The system message is prepended inline so callers pass a single string to call_llm().
"""

from __future__ import annotations


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
) -> str:
    """Build a light batch scan prompt.

    Args:
        question: The prediction market question text.
        mid_price: Current market mid-price (0–1 probability).
        news_summary: Recent headlines/descriptions as a single string.

    Returns:
        Full prompt string to pass to call_llm(tier="fast").
    """
    return f"""{_SYSTEM_LIGHT}

Market question: {question}
Current mid-price (probability): {mid_price:.3f}
Recent news: {news_summary or "No recent news available."}

Even when no news is available, reason about whether the current mid-price looks \
fair or mispriced using your prior knowledge of the topic, known market biases \
(e.g. longshot bias in racing: punters systematically overbid longshots and \
underbid favourites), or any other relevant factors. Only return sentiment_delta=0 \
if you have genuinely no view at all.

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
) -> str:
    """Build a deep trigger analysis prompt.

    Args:
        question: The prediction market question text.
        mid_price: Current market mid-price (0–1 probability).
        news_summary: Recent news headlines as a single string.
        x_summary: Recent X/Twitter posts as a single string.

    Returns:
        Full prompt string to pass to call_llm(tier="deep").
    """
    return f"""{_SYSTEM_DEEP}

Market question: {question}
Current mid-price (probability): {mid_price:.3f}
Recent news: {news_summary or "No recent news available."}
X/Twitter sentiment: {x_summary or "No X data available."}

Step-by-step:
1. Synthesize all news and X/Twitter signals.
2. Estimate the directional impact and its magnitude.
3. Identify the key uncertainties.

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
