"""Oracle Python Agent — Entry point.

Usage:
    python main.py           # paper trading on Betfair AU data (default)
    python main.py --live    # raises NotImplementedError (Phase 6 only)

Kill switch:
    touch state/kill_switch.txt   — agent skips execution, scan only
    rm state/kill_switch.txt      — trading resumes next cycle
"""

from __future__ import annotations

import argparse
import logging
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import settings
from src.enrichment.news import get_news_summary, rewrite_query
from src.enrichment.stats import get_match_stats
from src.enrichment.trigger import should_trigger_deep
from src.enrichment.x_sentiment import get_x_summary
from src.execution.paper import PaperBroker
from src.llm.client import call_llm, call_perplexity
from src.llm.models import DeepTriggerResponse, LightScanResponse, light_scan_schema, deep_trigger_schema
from src.llm.prompts import (
    build_deep_trigger_prompt,
    build_light_scan_prompt,
    build_perplexity_query,
    format_stats_context,
)
from src.logging_setup import configure_logging
from src.risk.manager import check_risk_gates
from src.scanner import betfair_scanner
from src.storage.state_manager import OracleState, StateManager
from src.strategy.bayesian import update_probability
from src.strategy.edge import executable_edge
from src.strategy.kelly import apply_oracle_sizing, commission_aware_kelly
from src.strategy.statistical_model import predict_match_odds, select_runner_prob

configure_logging()
logger = logging.getLogger(__name__)

# Maximum markets to run through the intelligence layer per cycle.
# Keeps LLM costs predictable during development.
_MAX_MARKETS_PER_CYCLE = 15

# Betfair event type IDs for sport detection
_EVENT_TYPE_SOCCER = "1"
_EVENT_TYPE_AFL = "61"
# Event types that have team-vs-team stats available
_STATS_ELIGIBLE_EVENT_TYPES = {_EVENT_TYPE_SOCCER, _EVENT_TYPE_AFL}

# Halts all new trade execution when this file exists.
_KILL_SWITCH_PATH = Path("state/kill_switch.txt")
# Dashboard can drop this file to trigger an immediate scan cycle.
_TRIGGER_SCAN_PATH = Path("state/trigger_scan")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle prediction market agent")
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable live Betfair execution (Phase 6 only — currently disabled).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-market pipeline
# ---------------------------------------------------------------------------

def _analyse_and_trade(
    market: dict,
    state: OracleState,
    state_manager: StateManager,
    broker: PaperBroker,
    exposure: float,
    drawdown: float,
) -> None:
    """Run the full intelligence + strategy + execution pipeline for one market."""
    market_id = market["id"]
    question = market["question"]
    mid_price = market["probability"]
    runner_name = market.get("runner_name", "")
    market_type = market.get("market_type", "")
    home_team = market.get("home_team", "")
    away_team = market.get("away_team", "")
    event_type_id = market.get("event_type_id", "")
    market_start_time = market.get("market_start_time")
    selection_id = market.get("selection_id")

    # --- Statistical model (Phase 5A.2) ---
    p_model: float | None = None
    stats_context = ""
    match_stats = None

    if home_team and away_team and event_type_id in _STATS_ELIGIBLE_EVENT_TYPES:
        sport = "afl" if event_type_id == _EVENT_TYPE_AFL else "football"
        match_stats = get_match_stats(home_team, away_team, sport)
        if match_stats is not None:
            model_probs = predict_match_odds(match_stats)
            if model_probs is not None:
                p_model = select_runner_prob(
                    model_probs, runner_name, market_type, home_team, away_team,
                )
                if p_model is not None:
                    logger.info(
                        "Statistical model: p_model=%.3f for %s (market=%s)",
                        p_model, runner_name, market_id,
                    )
            stats_context = format_stats_context(match_stats)

    # --- Enrichment ---
    search_query = rewrite_query(question, runner_name=runner_name, market_type=market_type)
    news = get_news_summary(search_query)
    x_data = get_x_summary(search_query.split()[:5])

    # --- Light scan ---
    light_prompt = build_light_scan_prompt(
        question, mid_price, news,
        runner_name=runner_name, market_type=market_type,
        stats_context=stats_context, model_probability=p_model,
    )
    raw = call_llm(light_prompt, tier="fast", response_schema=light_scan_schema())

    if raw is None:
        logger.debug("Light scan returned None for market %s", market_id)
        return

    try:
        light = LightScanResponse(**raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Light scan parse error for %s: %s | raw: %s", market_id, exc, raw)
        return

    logger.info(
        "Light scan | market=%s delta=%.3f uncertainty=%.3f | %s",
        market_id,
        light.sentiment_delta,
        light.uncertainty_penalty,
        light.rationale[:80],
    )

    # --- Deep trigger decision ---
    response = light
    if should_trigger_deep(light.sentiment_delta, volatility_z=0.0, x_momentum=0.0):
        logger.info("Deep trigger fired for market %s", market_id)
        deep_prompt = build_deep_trigger_prompt(
            question, mid_price, news, x_data,
            runner_name=runner_name, market_type=market_type,
            stats_context=stats_context, model_probability=p_model,
        )
        deep_raw = call_llm(deep_prompt, tier="deep", response_schema=deep_trigger_schema())

        if deep_raw is not None:
            try:
                response = DeepTriggerResponse(**deep_raw)
                logger.info(
                    "Deep analysis | market=%s delta=%.3f uncertainty=%.3f factors=%s | %s",
                    market_id,
                    response.sentiment_delta,
                    response.uncertainty_penalty,
                    response.key_factors,
                    response.rationale[:120],
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Deep scan parse error for %s: %s | raw: %s", market_id, exc, deep_raw
                )
                # Fall through — use light scan result

    sentiment_delta = response.sentiment_delta
    uncertainty_penalty = response.uncertainty_penalty

    # --- Bayesian update (chain from prior if available) ---
    # Use statistical model probability as default prior when available;
    # otherwise fall back to market mid-price.
    default_prior = p_model if p_model is not None else mid_price
    prior_p = state.priors.get(market_id, default_prior)
    p_fair = update_probability(prior_p, sentiment_delta, settings.risk.beta)

    # Sanity gate: extreme divergence from mid_price likely means sign confusion.
    # (e.g. LLM returns negative delta for "Under X Goals" at low implied prob.)
    divergence = abs(p_fair - mid_price)
    if divergence > 0.40:
        logger.warning(
            "Extreme divergence | market=%s p_fair=%.3f mid=%.3f delta=%.3f — skipping (likely sign confusion).",
            market_id, p_fair, mid_price, sentiment_delta,
        )
        return

    # --- Market detail for accurate spread + liquidity ---
    try:
        detail = betfair_scanner.get_market_detail(market_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch market detail for %s: %s", market_id, exc)
        return

    if detail.get("isResolved"):
        logger.debug("Market %s already resolved — skipping.", market_id)
        return

    # Phase 5A.1: update last_seen_price for CLV tracking on open positions
    if market_id in state.positions:
        current_prob = detail.get("probability", mid_price)
        state.positions[market_id].last_seen_price = current_prob

    # Use real Betfair back/lay prices where available; fall back to synthetic spread.
    p_ask = detail.get("p_back") or PaperBroker.derive_spread(detail["probability"])[0]
    p_bid = detail.get("p_lay") or PaperBroker.derive_spread(detail["probability"])[1]
    available_liquidity = detail.get("totalLiquidity", state.bankroll * 0.10)

    # --- Direction selection (pick best positive edge ≥ margin_min_paper) ---
    back_edge = executable_edge(p_fair, p_ask, p_bid, "back")
    lay_edge = executable_edge(p_fair, p_ask, p_bid, "lay")
    margin_min = settings.triggers.margin_min_paper

    if back_edge >= margin_min and back_edge >= lay_edge:
        direction = "back"
        edge = back_edge
        q_market = p_ask
    elif lay_edge >= margin_min:
        direction = "lay"
        edge = lay_edge
        q_market = p_bid
    else:
        logger.info(
            "No edge | market=%s back=%.3f lay=%.3f min=%.3f",
            market_id, back_edge, lay_edge, margin_min,
        )
        return

    # --- Tier 2: Perplexity-enriched deep re-scan for edge candidates ---
    perplexity_query = build_perplexity_query(
        question, runner_name=runner_name, market_type=market_type,
        home_team=home_team, away_team=away_team,
    )
    perplexity_news = call_perplexity(perplexity_query)
    if perplexity_news:
        # Re-run deep LLM with grounded web search context
        enriched_news = f"{news}\n\n--- Perplexity web search ---\n{perplexity_news}"
        deep_prompt = build_deep_trigger_prompt(
            question, mid_price, enriched_news, x_data,
            runner_name=runner_name, market_type=market_type,
            stats_context=stats_context, model_probability=p_model,
        )
        deep_raw = call_llm(deep_prompt, tier="deep", response_schema=deep_trigger_schema())
        if deep_raw is not None:
            try:
                deep_resp = DeepTriggerResponse(**deep_raw)
                logger.info(
                    "Perplexity-enriched deep scan | market=%s delta=%.3f "
                    "uncertainty=%.3f factors=%s",
                    market_id, deep_resp.sentiment_delta,
                    deep_resp.uncertainty_penalty, deep_resp.key_factors,
                )
                # Override sentiment with the better-informed deep analysis
                sentiment_delta = deep_resp.sentiment_delta
                uncertainty_penalty = deep_resp.uncertainty_penalty

                # Re-compute Bayesian update and edge with new delta
                prior_p = state.priors.get(market_id, default_prior)
                p_fair = update_probability(prior_p, sentiment_delta, settings.risk.beta)

                divergence = abs(p_fair - mid_price)
                if divergence > 0.40:
                    logger.warning(
                        "Extreme divergence after Perplexity enrichment | "
                        "market=%s p_fair=%.3f mid=%.3f — skipping.",
                        market_id, p_fair, mid_price,
                    )
                    return

                # Re-fetch prices — market may have moved during Perplexity call
                try:
                    fresh = betfair_scanner.get_market_detail(market_id)
                    p_ask = fresh.get("p_back") or PaperBroker.derive_spread(fresh["probability"])[0]
                    p_bid = fresh.get("p_lay") or PaperBroker.derive_spread(fresh["probability"])[1]
                except Exception:  # noqa: BLE001
                    pass  # keep existing p_ask/p_bid if re-fetch fails

                back_edge = executable_edge(p_fair, p_ask, p_bid, "back")
                lay_edge = executable_edge(p_fair, p_ask, p_bid, "lay")

                if back_edge >= margin_min and back_edge >= lay_edge:
                    direction = "back"
                    edge = back_edge
                    q_market = p_ask
                elif lay_edge >= margin_min:
                    direction = "lay"
                    edge = lay_edge
                    q_market = p_bid
                else:
                    logger.info(
                        "Edge lost after Perplexity re-scan | market=%s back=%.3f lay=%.3f",
                        market_id, back_edge, lay_edge,
                    )
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Perplexity deep parse error for %s: %s", market_id, exc,
                )

    # --- Kelly sizing ---
    conf_score = PaperBroker.derive_conf_score(uncertainty_penalty)
    f_star = commission_aware_kelly(
        p_fair, q_market, settings.risk.commission_pct, direction
    )
    if f_star <= 0:
        logger.info("Negative Kelly for %s — no edge after commission, skipping.", market_id)
        return

    f_final = apply_oracle_sizing(f_star, conf_score, drawdown, settings)

    # --- Risk gates ---
    approved, reason = check_risk_gates(f_final, conf_score, exposure, drawdown, settings)
    if not approved:
        logger.info("Risk gate blocked %s: %s", market_id, reason)
        return

    # --- Execution (paper) ---
    fill_price = p_ask if direction == "back" else p_bid
    mst_iso = market_start_time.isoformat() if market_start_time else None
    state, trade = broker.execute(
        state=state,
        market_id=market_id,
        question=question[:120],
        direction=direction,
        f_final=f_final,
        fill_price=fill_price,
        edge=edge,
        p_fair=p_fair,
        kelly_f_star=f_star,
        kelly_f_final=f_final,
        conf_score=conf_score,
        uncertainty_penalty=uncertainty_penalty,
        available_liquidity=available_liquidity,
        market_start_time=mst_iso,
        selection_id=selection_id,
    )
    if trade is not None:
        state.priors[market_id] = p_fair
        state_manager.save(state)
        logger.info(
            "Trade logged | market=%s dir=%s filled=%.4f price=%.3f edge=%.3f",
            market_id, direction, trade.filled_size, trade.fill_price, edge,
        )


# ---------------------------------------------------------------------------
# Scan cycle
# ---------------------------------------------------------------------------

def _settle_betfair_positions(
    state: OracleState,
    broker: PaperBroker,
    state_manager: StateManager,
) -> OracleState:
    """Settle open positions using Betfair market data.

    Auto-cancels positions whose market_start_time has aged beyond the
    configured max_market_age_hours (catches season-long outrights).
    """
    max_age_h = settings.scanner.betfair_max_market_age_hours
    now = datetime.now(timezone.utc)

    for market_id in list(state.positions.keys()):
        pos = state.positions[market_id]

        # Auto-cancel positions in markets that have aged past the cutoff.
        if max_age_h > 0 and pos.market_start_time:
            try:
                mst_dt = datetime.fromisoformat(pos.market_start_time)
                age_hours = (now - mst_dt).total_seconds() / 3600
                if age_hours > max_age_h:
                    logger.info(
                        "Auto-cancelling aged position %s (age=%.0fh, max=%dh)",
                        market_id, age_hours, max_age_h,
                    )
                    try:
                        state, _ = broker.cancel_position(
                            state, market_id,
                            reason=f"market aged out ({age_hours:.0f}h > {max_age_h}h)",
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Auto-cancel failed for %s: %s", market_id, exc)
                    continue
            except (ValueError, TypeError):
                pass  # unparseable market_start_time — fall through to normal check

        try:
            detail = betfair_scanner.get_market_detail(market_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to fetch Betfair detail for open position %s: %s — skipping.",
                market_id,
                exc,
            )
            continue

        # Phase 5A.1: always update last_seen_price while market is still open.
        # This becomes the closing_price approximation at settlement.
        current_prob = detail.get("probability")
        if current_prob is not None and not detail.get("isResolved"):
            state.positions[market_id].last_seen_price = current_prob

        # Backfill market_start_time for positions opened before this field existed
        mst = detail.get("market_start_time")
        if mst and not state.positions[market_id].market_start_time:
            state.positions[market_id].market_start_time = mst.isoformat()

        # Backfill selection_id for positions opened before this field existed
        sid = detail.get("selection_id")
        if sid is not None and state.positions[market_id].selection_id is None:
            state.positions[market_id].selection_id = sid

        if detail.get("isResolved"):
            resolution = detail.get("resolution", "MKT")
            res_prob = detail.get("probability", 0.5)
            closing_price = state.positions[market_id].last_seen_price
            try:
                state, _ = broker.settle_position(
                    state, market_id, resolution, res_prob,
                    closing_price=closing_price,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Settlement failed for %s: %s", market_id, exc)

    # Persist updated last_seen_price values even when no settlements occurred
    state_manager.save(state)
    return state


def scan_cycle(
    state_manager: StateManager,
    broker: PaperBroker,
) -> None:
    """Run one full scan cycle with settlement checks and paper execution."""

    try:
        # Kill switch — scan only, no new trades
        if _KILL_SWITCH_PATH.exists():
            logger.warning("Kill switch active — skipping execution this cycle.")
            return

        # Load state and settle any resolved positions
        state = state_manager.load()
        state = _settle_betfair_positions(state, broker, state_manager)

        # Compute risk metrics once at cycle start
        exposure = state_manager.current_exposure(state)
        drawdown = state_manager.drawdown_pct(state)

        markets = betfair_scanner.get_markets(
            country_codes=settings.scanner.betfair_country_codes,
            hours_ahead=settings.scanner.betfair_hours_ahead,
        )
        logger.info("Scan cycle started, found %d markets", len(markets))

        # Randomize order so we don't re-scan the same subset every cycle.
        random.shuffle(markets)

        for market in markets[:_MAX_MARKETS_PER_CYCLE]:
            # Skip markets where we already hold a position
            if market["id"] in state.positions:
                logger.debug("Already hold position in %s — skipping.", market["id"])
                continue

            try:
                _analyse_and_trade(
                    market=market,
                    state=state,
                    state_manager=state_manager,
                    broker=broker,
                    exposure=exposure,
                    drawdown=drawdown,
                )
                # Reload after potential trade to get updated exposure/drawdown
                state = state_manager.load()
                exposure = state_manager.current_exposure(state)
                drawdown = state_manager.drawdown_pct(state)

            except Exception as exc:  # noqa: BLE001
                logger.error("Error analysing market %s: %s", market.get("id"), exc)

        # Always persist state so the dashboard shows current bankroll/positions
        # even during cycles where no edge was found and no trade was placed.
        state_manager.save(state)

        # Per-cycle summary
        open_count = len(state.positions)
        settled_count = sum(1 for t in state.trade_history if t.status == "settled")
        total_pnl = sum(t.pnl for t in state.trade_history if t.status == "settled" and t.pnl is not None)
        logger.info(
            "Cycle summary | markets=%d positions=%d settled=%d "
            "bankroll=%.2f pnl=%.2f exposure=%.2f drawdown=%.2f",
            len(markets), open_count, settled_count,
            state.bankroll, total_pnl, exposure, drawdown,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Scan cycle failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.live:
        raise NotImplementedError(
            "--live mode is reserved for Phase 6. "
            "Remove --live to run paper trading."
        )

    state_manager = StateManager()
    broker = PaperBroker(state_manager, settings)

    interval_min = settings.scanner.poll_interval_sec // 60
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scan_cycle,
        "interval",
        minutes=interval_min,
        id="scan",
        max_instances=1,    # prevents overlapping cycles during slow scans
        coalesce=True,      # if a cycle was missed, run once not multiple times
        kwargs={"state_manager": state_manager, "broker": broker},
        next_run_time=datetime.now(),
    )
    scheduler.start()
    logger.info(
        "Oracle agent started | data=Betfair AU (paper) interval=%d min | Ctrl+C to stop.",
        interval_min,
    )

    stop = False

    def _shutdown(signum, frame):  # noqa: ANN001
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while not stop:
        # Check for a manual trigger from the dashboard
        if _TRIGGER_SCAN_PATH.exists():
            _TRIGGER_SCAN_PATH.unlink(missing_ok=True)
            logger.info("Manual scan triggered via dashboard.")
            scan_cycle(state_manager=state_manager, broker=broker)
        time.sleep(1)

    logger.info("Shutting down scheduler…")
    scheduler.shutdown(wait=False)
    logger.info("Oracle agent stopped.")


if __name__ == "__main__":
    main()
