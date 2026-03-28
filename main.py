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
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import PROJECT_ROOT, settings
from src.enrichment.news import get_news_summary, rewrite_query
from src.enrichment.stats import get_match_stats
from src.enrichment.trigger import should_trigger_deep
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
from src.storage.rejection_cache import RejectionCache
from src.storage.scan_feed import ScanFeedWriter
from src.storage.state_manager import OracleState, StateManager, Trade
from src.strategy.bayesian import update_probability
from src.strategy.edge import executable_edge
from src.strategy.kelly import apply_oracle_sizing, commission_aware_kelly
from src.strategy.statistical_model import predict_match_odds, select_runner_prob

configure_logging()
logger = logging.getLogger(__name__)

# Betfair event type IDs for sport detection
_EVENT_TYPE_SOCCER = "1"
_EVENT_TYPE_TENNIS = "2"
_EVENT_TYPE_CRICKET = "4"
_EVENT_TYPE_RUGBY_UNION = "5"
_EVENT_TYPE_AFL = "61420"
_EVENT_TYPE_RUGBY_LEAGUE = "1477"
_EVENT_TYPE_BASEBALL = "7511"
_EVENT_TYPE_BASKETBALL = "7522"
_EVENT_TYPE_ICE_HOCKEY = "7524"
# Event types that have team-vs-team stats available
_STATS_ELIGIBLE_EVENT_TYPES = {
    _EVENT_TYPE_SOCCER, _EVENT_TYPE_AFL,
    _EVENT_TYPE_BASKETBALL, _EVENT_TYPE_BASEBALL,
    _EVENT_TYPE_RUGBY_UNION, _EVENT_TYPE_ICE_HOCKEY,
    _EVENT_TYPE_RUGBY_LEAGUE,
    # Cricket removed — no CRICKET_API_KEY configured
}

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
    feed: ScanFeedWriter | None = None,
    rejection_cache: RejectionCache | None = None,
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

    # --- Niche league filter: skip markets where stats coverage is poor ---
    _niche_tags = ("u21", "u23", "u18", "u19", "u20", "reserve", "youth", "women", "(w)", " w)")
    question_lower = question.lower()
    runner_lower = runner_name.lower()
    if any(tag in question_lower or tag in runner_lower for tag in _niche_tags):
        logger.info(
            "Niche league gate | market=%s question=%s — skipping (poor stats coverage)",
            market_id, question[:80],
        )
        if feed:
            feed.log_market(market_id, question, "skipped_niche", reason="poor stats coverage")
        if rejection_cache:
            rejection_cache.reject(market_id, "skipped_niche")
        return

    # --- Statistical model (Phase 5A.2) ---
    p_model: float | None = None
    stats_context = ""
    match_stats = None

    if home_team and away_team and event_type_id in _STATS_ELIGIBLE_EVENT_TYPES:
        if event_type_id == _EVENT_TYPE_AFL:
            sport = "afl"
        elif event_type_id == _EVENT_TYPE_BASKETBALL:
            sport = "basketball"
        elif event_type_id == _EVENT_TYPE_BASEBALL:
            sport = "baseball"
        elif event_type_id == _EVENT_TYPE_RUGBY_UNION:
            sport = "rugby"
        elif event_type_id == _EVENT_TYPE_ICE_HOCKEY:
            sport = "hockey"
        elif event_type_id == _EVENT_TYPE_CRICKET:
            sport = "cricket"
        elif event_type_id == _EVENT_TYPE_RUGBY_LEAGUE:
            sport = "rugby_league"
        else:
            sport = "football"
        competition = market.get("competition_name", "")
        match_stats = get_match_stats(home_team, away_team, sport, competition=competition)
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

    # --- No-model gate: require a statistical model to trade ---
    # LLM-only probability estimates have no edge over the market.
    # Without a calibrated model anchor, skip the market entirely.
    if p_model is None:
        logger.info(
            "No model gate | market=%s question=%s — skipping (no statistical model)",
            market_id, question[:80],
        )
        if feed:
            feed.log_market(market_id, question, "skipped_no_model", reason="no statistical model")
        if rejection_cache:
            rejection_cache.reject(market_id, "skipped_no_model")
        return

    # --- Enrichment ---
    search_query = rewrite_query(question, runner_name=runner_name, market_type=market_type)
    news = get_news_summary(search_query)
    # --- Light scan ---
    light_prompt = build_light_scan_prompt(
        question, mid_price, news,
        runner_name=runner_name, market_type=market_type,
        stats_context=stats_context, model_probability=p_model,
    )
    raw = call_llm(light_prompt, tier="fast", response_schema=light_scan_schema())

    if raw is None:
        logger.debug("Light scan returned None for market %s", market_id)
        if feed:
            feed.log_market(market_id, question, "skipped_llm_fail", reason="light scan returned None")
        return

    try:
        light = LightScanResponse(**raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Light scan parse error for %s: %s | raw: %s", market_id, exc, raw)
        if feed:
            feed.log_market(market_id, question, "skipped_llm_fail", reason=f"parse error: {exc}")
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
    if should_trigger_deep(light.sentiment_delta, volatility_z=0.0):
        logger.info("Deep trigger fired for market %s", market_id)
        deep_prompt = build_deep_trigger_prompt(
            question, mid_price, news,
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

    # Reduce beta when the statistical model is absent — the LLM sentiment
    # signal is the only input so it shouldn't be amplified as aggressively.
    effective_beta = settings.risk.beta if p_model is not None else settings.risk.beta * 0.75
    p_fair = update_probability(prior_p, sentiment_delta, effective_beta)

    # Cap LLM adjustment: the LLM is a fine-tuner, not a probability generator.
    # It should only shift p_fair by a small amount relative to the statistical model.
    _MAX_LLM_ADJUSTMENT = 0.05
    if p_model is not None:
        p_fair_uncapped = p_fair
        p_fair = max(p_model - _MAX_LLM_ADJUSTMENT, min(p_model + _MAX_LLM_ADJUSTMENT, p_fair))
        if abs(p_fair - p_fair_uncapped) > 1e-6:
            logger.debug(
                "LLM adjustment capped | market=%s p_model=%.3f uncapped=%.3f capped=%.3f",
                market_id, p_model, p_fair_uncapped, p_fair,
            )

    # Sanity gate: extreme divergence from mid_price likely means sign confusion.
    # (e.g. LLM returns negative delta for "Under X Goals" at low implied prob.)
    divergence = abs(p_fair - mid_price)
    if divergence > 0.40:
        logger.warning(
            "Extreme divergence | market=%s p_fair=%.3f mid=%.3f delta=%.3f — skipping (likely sign confusion).",
            market_id, p_fair, mid_price, sentiment_delta,
        )
        if feed:
            feed.log_market(
                market_id, question, "skipped_divergence",
                reason=f"p_fair={p_fair:.3f} mid={mid_price:.3f}",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
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

    # Phase 5A.1: update last_seen_price for CLV tracking on open positions.
    # Use raw_probability (None when book is empty/suspended) to avoid writing
    # the 0.5 fallback as if it were a real closing price.
    if market_id in state.positions:
        raw_prob = detail.get("raw_probability")
        if raw_prob is not None:
            state.positions[market_id].last_seen_price = raw_prob

    # Use real Betfair back/lay prices where available; fall back to synthetic spread.
    p_ask = detail.get("p_back") or PaperBroker.derive_spread(detail["probability"])[0]
    p_bid = detail.get("p_lay") or PaperBroker.derive_spread(detail["probability"])[1]
    available_liquidity = detail.get("totalLiquidity", state.bankroll * 0.10)

    # --- Realism gates ---
    # A. Minimum liquidity gate
    min_liq = settings.risk.min_market_liquidity_aud
    if min_liq > 0 and available_liquidity < min_liq:
        logger.info(
            "Liquidity gate | market=%s liquidity=%.2f < min=%.2f",
            market_id, available_liquidity, min_liq,
        )
        if feed:
            feed.log_market(
                market_id, question, "skipped_liquidity",
                reason=f"liquidity={available_liquidity:.2f} < min={min_liq:.2f}",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
            )
        if rejection_cache:
            rejection_cache.reject(market_id, "skipped_liquidity")
        return

    # B. Minimum matched volume gate
    matched_volume = detail.get("volume", 0) or 0
    min_vol = settings.risk.min_matched_volume_aud
    if min_vol > 0 and matched_volume < min_vol:
        logger.info(
            "Volume gate | market=%s volume=%.2f < min=%.2f",
            market_id, matched_volume, min_vol,
        )
        if feed:
            feed.log_market(
                market_id, question, "skipped_volume",
                reason=f"volume={matched_volume:.2f} < min={min_vol:.2f}",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                volume=matched_volume,
            )
        if rejection_cache:
            rejection_cache.reject(market_id, "skipped_volume")
        return

    # C2. In-play safety gate — reject trades on in-play markets
    if not settings.risk.allow_in_play and detail.get("inplay"):
        logger.info(
            "In-play gate | market=%s — skipping (no in-play engine)",
            market_id,
        )
        if feed:
            feed.log_market(
                market_id, question, "skipped_inplay",
                reason="in-play market",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                volume=matched_volume,
            )
        if rejection_cache:
            rejection_cache.reject(market_id, "skipped_inplay")
        return

    # C3. Crossed-book detection — stale/suspended data
    if settings.risk.reject_crossed_book:
        bb = detail.get("best_back_price")
        bl = detail.get("best_lay_price")
        if bb is not None and bl is not None and bb > bl:
            logger.info(
                "Crossed book gate | market=%s best_back=%.2f > best_lay=%.2f — stale data",
                market_id, bb, bl,
            )
            if feed:
                feed.log_market(
                    market_id, question, "skipped_crossed",
                    reason=f"best_back={bb:.2f} > best_lay={bl:.2f}",
                    delta=sentiment_delta, uncertainty=uncertainty_penalty,
                    volume=matched_volume,
                )
            if rejection_cache:
                rejection_cache.reject(market_id, "skipped_crossed")
            return

    # --- Direction selection (pick best positive edge ≥ margin_min_paper) ---
    back_edge = executable_edge(p_fair, p_ask, p_bid, "back")
    lay_edge = executable_edge(p_fair, p_ask, p_bid, "lay")
    margin_min = settings.triggers.margin_min_paper
    commission = settings.risk.commission_pct

    # Compare raw edges against the threshold.
    # Commission is already handled in commission_aware_kelly() — no need to
    # subtract it here (that was double-counting and blocking all trades).
    net_back_edge = back_edge
    net_lay_edge = lay_edge

    if net_back_edge >= margin_min and net_back_edge >= net_lay_edge:
        direction = "back"
        edge = back_edge
        q_market = p_ask
    elif net_lay_edge >= margin_min:
        # C. Extreme-odds lay filter
        max_lay_prob = settings.risk.max_lay_probability
        if p_bid > max_lay_prob:
            logger.info(
                "Extreme lay gate | market=%s p_bid=%.4f > max=%.4f",
                market_id, p_bid, max_lay_prob,
            )
            if feed:
                feed.log_market(
                    market_id, question, "no_edge",
                    reason=f"extreme lay p_bid={p_bid:.4f} > max={max_lay_prob:.4f}",
                    delta=sentiment_delta, uncertainty=uncertainty_penalty,
                    volume=matched_volume, back_edge=back_edge, lay_edge=lay_edge,
                )
            return
        direction = "lay"
        edge = lay_edge
        q_market = p_bid
    else:
        logger.info(
            "No edge | market=%s back=%.3f lay=%.3f net_back=%.3f net_lay=%.3f min=%.3f",
            market_id, back_edge, lay_edge, net_back_edge, net_lay_edge, margin_min,
        )
        if feed:
            feed.log_market(
                market_id, question, "no_edge",
                reason=f"back={back_edge:.3f} lay={lay_edge:.3f} net_back={net_back_edge:.3f} net_lay={net_lay_edge:.3f} min={margin_min:.3f}",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                volume=matched_volume, back_edge=back_edge, lay_edge=lay_edge,
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
            question, mid_price, enriched_news,
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
                # Blend Perplexity-enriched sentiment with prior analysis
                # instead of overwriting — prevents single-source whiplash.
                light_delta = sentiment_delta  # preserve the pre-Perplexity value
                sentiment_delta = 0.6 * deep_resp.sentiment_delta + 0.4 * light_delta
                uncertainty_penalty = min(
                    deep_resp.uncertainty_penalty, uncertainty_penalty,
                )

                # Re-compute Bayesian update and edge with new delta
                prior_p = state.priors.get(market_id, default_prior)
                p_fair = update_probability(prior_p, sentiment_delta, effective_beta)

                # Handshake gate: model and Perplexity must AGREE on direction.
                # Data from 29 trades shows:
                #   Perplexity agrees with model:   +$118.65 (7W/6L)
                #   Perplexity disagrees:           -$689.25 (4W/12L)
                # When they disagree, skip — the "edge" is likely model error.
                if p_model is not None:
                    model_vs_market = p_model - mid_price
                    perp_delta = deep_resp.sentiment_delta
                    # Model says back (p_model > market) but Perplexity says negative
                    # Model says lay (p_model < market) but Perplexity says positive
                    if (model_vs_market > 0 and perp_delta < 0) or \
                       (model_vs_market < 0 and perp_delta > 0):
                        logger.warning(
                            "Handshake failed | market=%s p_model=%.3f "
                            "mid=%.3f perplexity_delta=%.3f — model and Perplexity disagree, skipping.",
                            market_id, p_model, mid_price, perp_delta,
                        )
                        if feed:
                            feed.log_market(
                                market_id, question, "skipped_handshake",
                                reason=f"p_model={p_model:.3f} mid={mid_price:.3f} "
                                       f"perplexity_delta={perp_delta:.3f}",
                                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                                volume=matched_volume,
                            )
                        return

                divergence = abs(p_fair - mid_price)
                if divergence > 0.40:
                    logger.warning(
                        "Extreme divergence after Perplexity enrichment | "
                        "market=%s p_fair=%.3f mid=%.3f — skipping.",
                        market_id, p_fair, mid_price,
                    )
                    if feed:
                        feed.log_market(
                            market_id, question, "skipped_divergence",
                            reason=f"post-Perplexity p_fair={p_fair:.3f} mid={mid_price:.3f}",
                            delta=sentiment_delta, uncertainty=uncertainty_penalty,
                            volume=matched_volume,
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
                net_back_edge = back_edge
                net_lay_edge = lay_edge

                if net_back_edge >= margin_min and net_back_edge >= net_lay_edge:
                    direction = "back"
                    edge = back_edge
                    q_market = p_ask
                elif net_lay_edge >= margin_min:
                    direction = "lay"
                    edge = lay_edge
                    q_market = p_bid
                else:
                    logger.info(
                        "Edge lost after Perplexity re-scan | market=%s back=%.3f lay=%.3f",
                        market_id, back_edge, lay_edge,
                    )
                    if feed:
                        feed.log_market(
                            market_id, question, "edge_lost",
                            reason=f"post-Perplexity back={back_edge:.3f} lay={lay_edge:.3f}",
                            delta=sentiment_delta, uncertainty=uncertainty_penalty,
                            volume=matched_volume, back_edge=back_edge, lay_edge=lay_edge,
                        )
                    return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Perplexity deep parse error for %s: %s", market_id, exc,
                )

    # --- Kelly sizing ---
    conf_score = PaperBroker.derive_conf_score(uncertainty_penalty)

    # Note: p_model is always non-None here (no-model gate above).
    # The confidence penalty for model-absent markets is no longer needed.

    f_star = commission_aware_kelly(
        p_fair, q_market, settings.risk.commission_pct, direction
    )
    if f_star <= 0:
        logger.info("Negative Kelly for %s — no edge after commission, skipping.", market_id)
        if feed:
            feed.log_market(
                market_id, question, "negative_kelly",
                reason="f_star <= 0 after commission",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                volume=matched_volume, back_edge=back_edge, lay_edge=lay_edge,
                direction=direction,
            )
        return

    f_final = apply_oracle_sizing(f_star, conf_score, drawdown, settings)

    # --- Risk gates ---
    approved, gate_reason = check_risk_gates(f_final, conf_score, exposure, drawdown, settings)
    if not approved:
        logger.info("Risk gate blocked %s: %s", market_id, gate_reason)
        if feed:
            feed.log_market(
                market_id, question, "risk_blocked",
                reason=gate_reason,
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                volume=matched_volume, back_edge=back_edge, lay_edge=lay_edge,
                direction=direction, f_final=f_final,
            )
        return

    # --- Execution (paper) ---
    fill_price = p_ask if direction == "back" else p_bid
    depth_ladder = detail.get("depth_back" if direction == "back" else "depth_lay", [])
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
        depth_ladder=depth_ladder,
        margin_min=margin_min,
    )
    if trade is not None:
        state.priors[market_id] = p_fair
        state_manager.save(state)
        logger.info(
            "Trade logged | market=%s dir=%s filled=%.4f price=%.3f edge=%.3f",
            market_id, direction, trade.filled_size, trade.fill_price, edge,
        )
        if feed:
            feed.log_market(
                market_id, question, "traded",
                delta=sentiment_delta, uncertainty=uncertainty_penalty,
                volume=matched_volume, back_edge=back_edge, lay_edge=lay_edge,
                direction=direction, fill_price=trade.fill_price, f_final=f_final,
            )


# ---------------------------------------------------------------------------
# CLV closing-line snapshot
# ---------------------------------------------------------------------------

def _update_closing_lines(*, state_manager: StateManager) -> None:
    """Lightweight job that snapshots pre-play prices for open positions.

    Runs every ~5 minutes (separate from the main scan cycle) so that
    ``last_seen_price`` closely approximates the true closing line when
    the market eventually goes in-play.  Only calls ``get_market_detail``
    for markets we actually hold — typically 0-5 API calls per run.
    """
    state = state_manager.load()
    if not state.positions:
        return  # nothing to snapshot — zero API calls

    dirty = False
    for market_id, pos in list(state.positions.items()):
        try:
            detail = betfair_scanner.get_market_detail(market_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "CLV snapshot | market=%s fetch failed: %s", market_id, exc,
            )
            continue

        raw_prob = detail.get("raw_probability")
        is_inplay = detail.get("inplay", False)
        is_resolved = detail.get("isResolved", False)

        if raw_prob is not None and not is_inplay and not is_resolved:
            pos.last_seen_price = raw_prob
            dirty = True
            logger.debug(
                "CLV snapshot | market=%s price=%.4f inplay=%s",
                market_id, raw_prob, is_inplay,
            )
        elif is_inplay and not pos.clv_snapshot_stale:
            # Market went in-play — check if we ever captured a real
            # pre-play price (i.e. last_seen_price differs from entry).
            if (
                pos.last_seen_price is None
                or abs(pos.last_seen_price - pos.entry_price) < 1e-6
            ):
                pos.clv_snapshot_stale = True
                dirty = True
                logger.warning(
                    "CLV snapshot | market=%s went in-play before a "
                    "pre-play closing line was captured — CLV will be stale.",
                    market_id,
                )

    if dirty:
        state_manager.save(state)


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
    newly_settled: list[Trade] = []

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

        # Phase 5A.1: update last_seen_price while market is pre-play and unresolved.
        # This becomes the closing_price approximation at settlement.
        # Use raw_probability to avoid writing the 0.5 fallback as a real price.
        # IMPORTANT: Do NOT update once in-play — in-play prices reflect live game
        # state (near 1.0 for likely winners) and are not valid closing line values.
        # CLV should measure whether our entry beat the last pre-kickoff price.
        raw_prob = detail.get("raw_probability")
        is_inplay = detail.get("inplay", False)
        if raw_prob is not None and not detail.get("isResolved") and not is_inplay:
            state.positions[market_id].last_seen_price = raw_prob

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
            raw_runner_status = detail.get("runner_status")
            res_prob = detail.get("probability", 0.5)
            closing_price = state.positions[market_id].last_seen_price
            try:
                state, settled_trade = broker.settle_position(
                    state, market_id, resolution, res_prob,
                    closing_price=closing_price,
                    runner_status=raw_runner_status,
                )
                newly_settled.append(settled_trade)
            except Exception as exc:  # noqa: BLE001
                logger.error("Settlement failed for %s: %s", market_id, exc)

    # Post-settlement validation
    if newly_settled:
        _validate_settlements(newly_settled)

    # Persist updated last_seen_price values even when no settlements occurred
    state_manager.save(state)
    return state


def _validate_settlements(trades: list[Trade]) -> None:
    """Post-cycle sanity checks on newly settled trades.

    Logs warnings for any settlement that looks suspicious:
    - Used MKT fallback (runner.status was missing/unknown)
    - exit_price == 0.5 (the old broken-fallback symptom)
    - P&L sign inconsistent with direction + resolution
    """
    for t in trades:
        market_id = t.market_id
        prefix = f"Settlement validator | market={market_id}"

        # 1. MKT fallback — should never happen on Betfair now
        if t.resolution == "MKT" or t.resolution is None:
            logger.warning(
                "%s | ALERT: used MKT fallback (runner_status=%r). "
                "P&L may be wrong.",
                prefix, t.runner_status,
            )

        # 2. exit_price stuck at 0.5 — the old bug symptom
        if t.exit_price is not None and t.exit_price == 0.5:
            logger.warning(
                "%s | ALERT: exit_price=0.500 — possible fallback. "
                "resolution=%s runner_status=%r",
                prefix, t.resolution, t.runner_status,
            )

        # 3. runner_status missing when market was supposedly resolved
        if not t.runner_status:
            logger.warning(
                "%s | ALERT: runner_status is empty/None. "
                "Betfair may not have returned runner data.",
                prefix,
            )

        # 4. P&L sign vs direction+resolution consistency
        if t.pnl is not None and t.resolution in ("YES", "NO"):
            if t.direction == "back" and t.resolution == "YES" and t.pnl < 0:
                logger.warning(
                    "%s | INCONSISTENT: back+YES but pnl=%.4f (negative)",
                    prefix, t.pnl,
                )
            if t.direction == "back" and t.resolution == "NO" and t.pnl > 0:
                logger.warning(
                    "%s | INCONSISTENT: back+NO but pnl=%.4f (positive)",
                    prefix, t.pnl,
                )
            if t.direction == "lay" and t.resolution == "NO" and t.pnl < 0:
                logger.warning(
                    "%s | INCONSISTENT: lay+NO but pnl=%.4f (negative)",
                    prefix, t.pnl,
                )
            if t.direction == "lay" and t.resolution == "YES" and t.pnl > 0:
                logger.warning(
                    "%s | INCONSISTENT: lay+YES but pnl=%.4f (positive)",
                    prefix, t.pnl,
                )

        # 5. All clear
        if (
            t.resolution in ("YES", "NO", "VOID")
            and t.runner_status
            and t.exit_price != 0.5
        ):
            logger.info(
                "%s | OK: resolution=%s runner_status=%s pnl=%.4f",
                prefix, t.resolution, t.runner_status, t.pnl or 0,
            )


def _market_priority(m: dict, now: datetime) -> tuple[int, float, float]:
    """Priority key for market sorting. Lower tuple = higher priority.

    Tier 1: Time-to-kickoff buckets (urgent first)
    Tier 2: Volume descending (liquid markets first)
    Tier 3: Available liquidity descending
    """
    start = m.get("market_start_time")
    if start:
        hours_to_start = (start - now).total_seconds() / 3600
        if hours_to_start <= 3:
            time_tier = 0   # urgent — last chance to trade
        elif hours_to_start <= 12:
            time_tier = 1   # active trading window
        else:
            time_tier = 2   # early scan
    else:
        time_tier = 3       # unknown start time — lowest priority

    volume = m.get("volume", 0) or 0
    liq = m.get("totalLiquidity", 0) or 0
    return (time_tier, -volume, -liq)


def _compute_next_interval(
    total_markets: int,
    rejected_count: int,
    max_per_cycle: int,
) -> int:
    """Compute minutes until next cycle based on market density.

    Scans more frequently when there are more actionable markets,
    less frequently when most are cached/rejected.
    """
    min_min = settings.scanner.min_interval_min
    max_min = settings.scanner.max_interval_min
    actionable = max(total_markets - rejected_count, 1)

    # How many cycles needed to cover 80% of actionable markets?
    cycles_needed = max(int(actionable * 0.80 / max_per_cycle), 1)

    # Spread those cycles over 60 minutes
    interval = max(60 // cycles_needed, min_min)
    interval = min(interval, max_min)
    return interval


def scan_cycle(
    state_manager: StateManager,
    broker: PaperBroker,
    rejection_cache: RejectionCache | None = None,
    scheduler: BackgroundScheduler | None = None,
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

        # Prioritise: soonest kickoff + highest volume first
        now = datetime.now(timezone.utc)
        markets.sort(key=lambda m: _market_priority(m, now))

        max_per_cycle = settings.scanner.max_markets_per_cycle

        feed = ScanFeedWriter()
        feed.begin_cycle(len(markets))

        analysed = 0
        cache_skipped = 0
        new_trades_this_cycle = 0
        max_new_trades = settings.risk.max_new_positions_per_cycle
        for market in markets:
            if analysed >= max_per_cycle:
                break

            if new_trades_this_cycle >= max_new_trades:
                logger.info(
                    "Per-cycle position cap reached (%d) — skipping remaining markets.",
                    max_new_trades,
                )
                break

            # Skip markets where we already hold a position
            if market["id"] in state.positions:
                logger.debug("Already hold position in %s — skipping.", market["id"])
                continue

            # Skip markets that recently failed a hard gate
            if rejection_cache and rejection_cache.is_rejected(market["id"]):
                cache_skipped += 1
                continue

            analysed += 1
            positions_before = len(state.positions)
            try:
                _analyse_and_trade(
                    market=market,
                    state=state,
                    state_manager=state_manager,
                    broker=broker,
                    exposure=exposure,
                    drawdown=drawdown,
                    feed=feed,
                    rejection_cache=rejection_cache,
                )
                # Reload after potential trade to get updated exposure/drawdown
                state = state_manager.load()
                exposure = state_manager.current_exposure(state)
                drawdown = state_manager.drawdown_pct(state)
                if len(state.positions) > positions_before:
                    new_trades_this_cycle += 1

            except Exception as exc:  # noqa: BLE001
                logger.error("Error analysing market %s: %s", market.get("id"), exc)

        feed.end_cycle()

        # Always persist state so the dashboard shows current bankroll/positions
        # even during cycles where no edge was found and no trade was placed.
        state_manager.save(state)

        # Adaptive interval: reschedule based on market density
        rejected_count = rejection_cache.rejected_count(
            [m["id"] for m in markets],
        ) if rejection_cache else 0
        next_min = _compute_next_interval(len(markets), rejected_count, max_per_cycle)

        if scheduler is not None:
            try:
                scheduler.reschedule_job("scan", trigger="interval", minutes=next_min)
            except Exception:  # noqa: BLE001
                pass  # keep current interval if reschedule fails

        # Per-cycle summary
        open_count = len(state.positions)
        settled_count = sum(1 for t in state.trade_history if t.status == "settled")
        total_pnl = sum(t.pnl for t in state.trade_history if t.status == "settled" and t.pnl is not None)
        logger.info(
            "Cycle summary | markets=%d analysed=%d cache_skipped=%d "
            "next_interval=%dmin positions=%d settled=%d "
            "bankroll=%.2f pnl=%.2f exposure=%.2f drawdown=%.2f",
            len(markets), analysed, cache_skipped,
            next_min, open_count, settled_count,
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
    rejection_cache = RejectionCache(ttl_minutes=settings.scanner.rejection_cache_ttl_min)

    interval_min = settings.scanner.poll_interval_sec // 60
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scan_cycle,
        "interval",
        minutes=interval_min,
        id="scan",
        max_instances=1,    # prevents overlapping cycles during slow scans
        coalesce=True,      # if a cycle was missed, run once not multiple times
        kwargs={
            "state_manager": state_manager,
            "broker": broker,
            "rejection_cache": rejection_cache,
            "scheduler": scheduler,
        },
        next_run_time=datetime.now(),
    )
    scheduler.add_job(
        _update_closing_lines,
        "interval",
        minutes=5,
        id="clv_snapshot",
        max_instances=1,
        coalesce=True,
        kwargs={"state_manager": state_manager},
        next_run_time=None,  # first scan_cycle sets initial prices
    )
    scheduler.start()

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"

    logger.info(
        "Oracle agent started | version=%s data=Betfair AU (paper) interval=%d min | Ctrl+C to stop.",
        commit, interval_min,
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
            scan_cycle(
                state_manager=state_manager,
                broker=broker,
                rejection_cache=rejection_cache,
                scheduler=scheduler,
            )
        time.sleep(1)

    logger.info("Shutting down scheduler…")
    scheduler.shutdown(wait=False)
    logger.info("Oracle agent stopped.")


if __name__ == "__main__":
    main()
