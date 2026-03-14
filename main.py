"""Oracle Python Agent — Entry point."""

from __future__ import annotations

import logging
import signal
import time

from apscheduler.schedulers.background import BackgroundScheduler

from src.enrichment.news import get_news_summary
from src.enrichment.trigger import should_trigger_deep
from src.enrichment.x_sentiment import get_x_summary
from src.llm.client import call_llm
from src.llm.models import DeepTriggerResponse, LightScanResponse
from src.llm.prompts import build_deep_trigger_prompt, build_light_scan_prompt
from src.logging_setup import configure_logging
from src.scanner.manifold import get_markets

configure_logging()
logger = logging.getLogger(__name__)

# Maximum markets to run through the intelligence layer per cycle.
# Keeps LLM costs predictable during development.
_MAX_MARKETS_PER_CYCLE = 5


def _analyse_market(market: dict) -> None:
    """Run the intelligence pipeline on a single market."""
    question = market["question"]
    mid_price = market["probability"]

    # Derive a short search query from the question (first 80 chars)
    search_query = question[:80]

    # --- Enrichment ---
    news = get_news_summary(search_query)
    x_data = get_x_summary(search_query.split()[:5])

    # --- Light scan ---
    light_prompt = build_light_scan_prompt(question, mid_price, news)
    raw = call_llm(light_prompt, tier="fast")

    if raw is None:
        logger.debug("Light scan returned None for market %s", market["id"])
        return

    try:
        light = LightScanResponse(**raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Light scan parse error for %s: %s | raw: %s", market["id"], exc, raw)
        return

    logger.info(
        "Light scan | market=%s delta=%.3f uncertainty=%.3f | %s",
        market["id"],
        light.sentiment_delta,
        light.uncertainty_penalty,
        light.rationale[:80],
    )

    # --- Deep trigger ---
    # volatility_z is not yet computed (Phase 3); pass 0.0 as placeholder
    if should_trigger_deep(light.sentiment_delta, volatility_z=0.0, x_momentum=0.0):
        logger.info("Deep trigger fired for market %s", market["id"])
        deep_prompt = build_deep_trigger_prompt(question, mid_price, news, x_data)
        deep_raw = call_llm(deep_prompt, tier="deep")

        if deep_raw is None:
            logger.debug("Deep analysis returned None for market %s", market["id"])
            return

        try:
            deep = DeepTriggerResponse(**deep_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deep scan parse error for %s: %s | raw: %s", market["id"], exc, deep_raw)
            return

        logger.info(
            "Deep analysis | market=%s delta=%.3f uncertainty=%.3f factors=%s | %s",
            market["id"],
            deep.sentiment_delta,
            deep.uncertainty_penalty,
            deep.key_factors,
            deep.rationale[:120],
        )


def scan_cycle() -> None:
    """Run one full scan cycle: fetch markets, enrich, and analyse via LLM."""
    try:
        markets = get_markets()
        logger.info("Scan cycle started, found %d markets", len(markets))

        for market in markets[:_MAX_MARKETS_PER_CYCLE]:
            try:
                _analyse_market(market)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error analysing market %s: %s", market.get("id"), exc)

    except Exception as exc:  # noqa: BLE001
        logger.error("Scan cycle failed: %s", exc)


def main() -> None:
    from src.config import settings

    interval_min = settings.scanner.poll_interval_sec // 60
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scan_cycle,
        "interval",
        minutes=interval_min,
        id="scan",
        next_run_time=__import__("datetime").datetime.now(),
    )
    scheduler.start()
    logger.info(
        "Oracle agent started. Scan interval: %d minutes. Press Ctrl+C to stop.",
        interval_min,
    )

    stop = False

    def _shutdown(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while not stop:
        time.sleep(1)

    logger.info("Shutting down scheduler…")
    scheduler.shutdown(wait=False)
    logger.info("Oracle agent stopped.")


if __name__ == "__main__":
    main()
