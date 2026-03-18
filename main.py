"""Oracle Python Agent — Entry point.

Usage:
    python main.py           # paper trading (default)
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
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import settings
from src.enrichment.news import get_news_summary, rewrite_query
from src.enrichment.trigger import should_trigger_deep
from src.enrichment.x_sentiment import get_x_summary
from src.execution.paper import PaperBroker
from src.llm.client import call_llm
from src.llm.models import DeepTriggerResponse, LightScanResponse
from src.llm.prompts import build_deep_trigger_prompt, build_light_scan_prompt
from src.logging_setup import configure_logging
from src.risk.manager import check_risk_gates
from src.scanner import betfair_scanner
from src.scanner.manifold import get_market_detail, get_markets
from src.storage.state_manager import OracleState, StateManager
from src.strategy.bayesian import update_probability
from src.strategy.edge import executable_edge
from src.strategy.kelly import apply_oracle_sizing, commission_aware_kelly

configure_logging()
logger = logging.getLogger(__name__)

# Maximum markets to run through the intelligence layer per cycle.
# Keeps LLM costs predictable during development.
_MAX_MARKETS_PER_CYCLE = 15

# Halts all new trade execution when this file exists.
_KILL_SWITCH_PATH = Path("state/kill_switch.txt")


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
    parser.add_argument(
        "--betfair-paper",
        action="store_true",
        default=False,
        help=(
            "Paper trading using real Betfair AU market data. "
            "No bets are placed — uses PaperBroker with Betfair prices."
        ),
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
    mode: str,
    get_detail_fn=get_market_detail,
) -> None:
    """Run the full intelligence + strategy + execution pipeline for one market."""
    market_id = market["id"]
    question = market["question"]
    mid_price = market["probability"]
    runner_name = market.get("runner_name", "")
    market_type = market.get("market_type", "")

    # --- Enrichment ---
    search_query = rewrite_query(question, runner_name=runner_name, market_type=market_type)
    news = get_news_summary(search_query)
    x_data = get_x_summary(search_query.split()[:5])

    # --- Light scan ---
    light_prompt = build_light_scan_prompt(
        question, mid_price, news, runner_name=runner_name, market_type=market_type
    )
    raw = call_llm(light_prompt, tier="fast")

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
        )
        deep_raw = call_llm(deep_prompt, tier="deep")

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
    prior_p = state.priors.get(market_id, mid_price)
    p_fair = update_probability(prior_p, sentiment_delta, settings.risk.beta)

    # --- Market detail for accurate spread + liquidity ---
    try:
        detail = get_detail_fn(market_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to fetch market detail for %s: %s", market_id, exc)
        return

    if detail.get("isResolved"):
        logger.debug("Market %s already resolved — skipping.", market_id)
        return

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

    # --- Execution ---
    if mode == "paper":
        fill_price = p_ask if direction == "back" else p_bid
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
        )
        if trade is not None:
            state.priors[market_id] = p_fair
            state_manager.save(state)
            logger.info(
                "Trade logged | market=%s dir=%s filled=%.4f price=%.3f edge=%.3f",
                market_id, direction, trade.filled_size, trade.fill_price, edge,
            )
    else:
        raise NotImplementedError(
            "--live mode is reserved for Phase 6. Remove --live for paper trading."
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

    Equivalent to PaperBroker.check_and_settle_positions() but calls
    betfair_scanner.get_market_detail() instead of the Manifold equivalent.
    PaperBroker is left unchanged — settlement is driven from here.
    """
    for market_id in list(state.positions.keys()):
        try:
            detail = betfair_scanner.get_market_detail(market_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to fetch Betfair detail for open position %s: %s — skipping.",
                market_id,
                exc,
            )
            continue

        if detail.get("isResolved"):
            resolution = detail.get("resolution", "MKT")
            res_prob = detail.get("probability", 0.5)
            try:
                state, _ = broker.settle_position(state, market_id, resolution, res_prob)
            except Exception as exc:  # noqa: BLE001
                logger.error("Settlement failed for %s: %s", market_id, exc)

    return state


def scan_cycle(
    state_manager: StateManager,
    broker: PaperBroker,
    mode: str,
) -> None:
    """Run one full scan cycle with settlement checks and paper execution."""
    is_betfair = mode == "betfair-paper"

    try:
        # Kill switch — scan only, no new trades
        if _KILL_SWITCH_PATH.exists():
            logger.warning("Kill switch active — skipping execution this cycle.")
            return

        # Load state and settle any resolved positions
        state = state_manager.load()
        if is_betfair:
            state = _settle_betfair_positions(state, broker, state_manager)
        else:
            state = broker.check_and_settle_positions(state)

        # Compute risk metrics once at cycle start
        exposure = state_manager.current_exposure(state)
        drawdown = state_manager.drawdown_pct(state)

        if is_betfair:
            markets = betfair_scanner.get_markets()
        else:
            markets = get_markets()
        logger.info("Scan cycle started (%s), found %d markets", mode, len(markets))

        # Randomize order so we don't re-scan the same subset every cycle.
        # betfair_scanner already sorts by market type priority, so shuffle within
        # type-groups by using a stable approach: shuffle then re-sort by priority tier.
        random.shuffle(markets)

        get_detail_fn = betfair_scanner.get_market_detail if is_betfair else get_market_detail

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
                    mode=mode,
                    get_detail_fn=get_detail_fn,
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
    mode = "betfair-paper" if args.betfair_paper else "paper"

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
        kwargs={"state_manager": state_manager, "broker": broker, "mode": mode},
        next_run_time=datetime.now(),
    )
    scheduler.start()
    data_source = "Betfair AU (paper)" if mode == "betfair-paper" else "Manifold (paper)"
    logger.info(
        "Oracle agent started | mode=%s data=%s interval=%d min | Ctrl+C to stop.",
        mode,
        data_source,
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
