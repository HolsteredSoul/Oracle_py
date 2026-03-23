# Oracle Python Agent — Development Plan

**v1.0.0 | March 2026**

---

## Overview

Six sequential phases. Each phase has a clear exit gate — a set of conditions that must be true before moving to the next phase. No phase is skipped. Paper trading precedes live money.

| Phase | Name | Exit Gate |
|---|---|---|
| 1 | Foundation | Project boots, reads config, polls Betfair AU |
| 2 | Intelligence | LLM calls return valid JSON, Bayesian updater works |
| 3 | Strategy & Risk | Kelly math validated by unit tests, risk gates fire correctly |
| 4 | Paper Trading | 50+ paper trades logged, P&L tracking accurate |
| 5 | Backtesting & Tuning | Parameters tuned, Sharpe > 0 on held-out data |
| 6 | Live Betfair | OMS tested, small stakes deployed, scale after 30 days |

---

## Phase 1: Foundation

**Goal:** Working project skeleton that boots, loads config, and fetches live Betfair AU data.

### Tasks

**1.1 — Project setup** ✅ *Completed 2026-03-13*
```
oracle-py/
├── main.py              (empty scheduler stub)
├── config.toml          (copy from whitepaper §5)
├── .env                 (empty template)
├── requirements.txt     (copy from whitepaper §10)
├── .gitignore
└── src/__init__.py
```
- ~~Create virtualenv: `python -m venv .venv`~~
- ~~Install deps: `pip install -r requirements.txt`~~
- ~~Verify imports all resolve~~

**1.2 — Config loader (`src/config.py`)** ✅ *Completed 2026-03-13*
- ~~Use `pydantic-settings` with `BaseSettings`~~
- ~~Load `config.toml` via `toml.load()` + `.env` via `python-dotenv`~~
- ~~All config values typed and validated at startup~~
- ~~Fail fast with clear error if required keys missing~~
- ~~**Test:** `python -c "from src.config import settings; print(settings)"` succeeds~~

**1.3 — Betfair scanner (`src/scanner/betfair_scanner.py`)** ✅ *Completed 2026-03-13*
- ~~`get_markets(country_codes=["AU"])` → list of market dicts~~
- ~~`get_market_detail(market_id)` → dict with probability, back/lay prices, liquidity~~
- ~~Filter: exclude resolved, exclude near-boundary (prob < 0.05 or > 0.95)~~
- ~~Interactive login via `betfairlightweight`~~
- ~~**Test:** Prints 5 live markets with current prices~~

**1.4 — Logging setup** ✅ *Completed 2026-03-13*
- ~~`loguru` or stdlib `logging` with rotating file handler (daily, 7-day retention)~~
- ~~Structured log format: `timestamp | level | module | message`~~
- ~~All modules use `logging.getLogger(__name__)`~~

**1.5 — Scheduler stub (`main.py`)** ✅ *Completed 2026-03-13*
- ~~`APScheduler` `BackgroundScheduler` with one job: scan every 30 minutes~~
- ~~Job just logs: `"Scan cycle started, found N markets"`~~
- ~~Graceful shutdown on Ctrl+C~~

**Phase 1 Exit Gate:** ✅ *Completed 2026-03-14*
- [x] `python main.py` runs without errors for 1 hour
- [x] Logs show regular 30-min scan cycles with Betfair market counts
- [x] Config validation catches a bad value and prints a clear error

---

## Phase 2: Intelligence Layer

**Goal:** LLM calls routed correctly, returning valid JSON, with cost tracking.

### Tasks

**2.1 — OpenRouter client (`src/llm/client.py`)** ✅ *Completed 2026-03-14*
- ~~`call_llm(prompt, tier="fast")` → parsed JSON dict~~
- ~~Tier routing: `fast` → config `fast_model`, `deep` → config `deep_model`~~
- ~~Daily spend tracking: read/write `state/llm_spend.json`; auto-downgrade at 80% of cap~~
- ~~Retry: `tenacity` with exponential backoff on 429, 500, 503~~
- ~~Timeout: 30s per call~~
- ~~**Test:** Single call returns `{"sentiment_delta": ..., "uncertainty_penalty": ..., "rationale": ...}`~~

**2.2 — Prompt templates (`src/llm/prompts.py`)** ✅ *Completed 2026-03-14*
- ~~`build_light_scan_prompt(question, mid_price, news_summary)` → str~~
- ~~`build_deep_trigger_prompt(question, mid_price, news_summary, x_summary)` → str~~
- ~~`build_exit_prompt(position, updated_news, current_price)` → str~~
- ~~Each function enforces JSON-only output instruction in system message~~
- ~~**Test:** All three prompts render without errors for sample inputs~~

**2.3 — Response validation (`src/llm/models.py`)** ✅ *Completed 2026-03-14*
- ~~Pydantic models: `LightScanResponse`, `DeepTriggerResponse`, `ExitResponse`~~
- ~~On parse failure: log the raw response, return `None` (caller skips trade)~~
- ~~Never let a malformed LLM response crash the main loop~~

**2.4 — Enrichment: News (`src/enrichment/news.py`)** ✅ *Completed 2026-03-14*
- ~~`get_news_summary(query, max_articles=5)` → str~~
- ~~NewsData.io API via `httpx`~~
- ~~Summarize: return top 5 headline+description pairs joined as one string~~
- ~~Cache per query per hour (simple dict in memory)~~

**2.5 — Enrichment: X/Twitter (`src/enrichment/x_sentiment.py`)** ✅ *Completed 2026-03-14*
- ~~`get_x_summary(keywords, max_tweets=20)` → str~~
- ~~X API v2 recent search endpoint via `httpx`~~
- ~~Rate-limit aware: track remaining requests from response headers~~
- ~~Return recent tweets joined as summary string; return empty string on rate limit hit (non-fatal)~~

**2.6 — Trigger detector (`src/enrichment/trigger.py`)** ✅ *Completed 2026-03-14*
- ~~`should_trigger_deep(sentiment_delta, volatility_z, x_momentum)` → bool~~
- ~~Pure function, testable without network calls~~
- ~~Reads thresholds from config~~

**Phase 2 Exit Gate:** ✅ *Completed 2026-03-14*
- [x] 10 consecutive light scan calls succeed and return valid JSON
- [x] 2 deep trigger calls succeed and return valid JSON
- [x] Daily spend counter increments correctly and logs warning near cap
- [x] A bad LLM response (simulate by mocking) does not crash the loop

---

## Phase 3: Strategy & Risk

**Goal:** All math is correct and covered by unit tests. No live money logic yet.

### Tasks

**3.1 — Edge calculator (`src/strategy/edge.py`)** ✅ *Completed 2026-03-15*
```python
def executable_edge(p_fair, p_ask, p_bid, direction) -> float:
    # direction: "back" or "lay"
    # Returns signed edge; negative means no edge
```
- ~~Commission not applied here (applied in Kelly)~~
- ~~**Tests:** Back edge positive when p_fair > ask; lay edge positive when p_fair < bid; edge = 0 at breakeven~~

**3.2 — Bayesian updater (`src/strategy/bayesian.py`)** ✅ *Completed 2026-03-15*
```python
def update_probability(p_mid, sentiment_delta, beta) -> float:
    # Returns p_fair in (0, 1) by construction
    # Uses scipy.special.logit / expit
```
- ~~Handle edge cases: `p_mid` near 0 or 1 (clamp to [0.001, 0.999] before logit)~~
- ~~`beta = 0` must return `p_mid` exactly~~
- ~~**Tests:** Positive delta increases probability; negative decreases; β=0 is identity; boundary probs handled~~

**3.3 — Kelly calculator (`src/strategy/kelly.py`)** ✅ *Completed 2026-03-15*
```python
def commission_aware_kelly(p_fair, q_market, gamma, direction) -> float:
    # p_fair: our estimate
    # q_market: implied prob from market price (p_ask for back, p_bid for lay)
    # gamma: commission rate (e.g. 0.05)
    # Returns f* (uncapped); caller applies scaling + caps

def apply_oracle_sizing(f_star, conf_score, drawdown_pct, config) -> float:
    # Applies k, lambda_conf, lambda_dd, and 0.25 hard cap
    # Returns final f (fraction of bankroll to risk)

def translate_to_betfair(f, bankroll, decimal_odds, direction) -> dict:
    # Returns {"stake": ..., "liability": ...}
    # direction: "back" or "lay"
```
- ~~f* > 1.0 logs a warning (hard cap handles safety); f_final hard-capped at 0.25~~
- ~~**Tests:** 10 unit tests + hypothesis property test (500 examples, f_final ≤ 0.25 always)~~

**3.4 — Risk manager (`src/risk/manager.py`)** ✅ *Completed 2026-03-15*
```python
def check_risk_gates(f, conf_score, current_exposure, drawdown_pct, config) -> tuple[bool, str]:
    # Returns (approved: bool, reason: str)
    # Gates: confidence floor, exposure cap reached, trade would breach cap
```
- ~~**Tests:** Each gate fires independently; approved path returns True with "OK" reason~~

**Phase 3 Exit Gate:** ✅ *Completed 2026-03-15*
- [x] `pytest tests/test_kelly.py tests/test_bayesian.py tests/test_edge.py tests/test_risk.py` all pass (31/31)
- [x] No assertions fail on any boundary input
- [x] Code review: no variable naming collisions (commission vs confidence)

---

## Phase 4: Paper Trading

**Goal:** Full end-to-end cycle running on Betfair AU paper trading with logged P&L.

### Tasks

**4.1 — State manager (`src/storage/state_manager.py`)** ✅ *Completed 2026-03-15*
```python
class StateManager:
    def load() -> OracleState
    def save(state: OracleState) -> None  # atomic write (write tmp, rename)
    def add_trade(trade: Trade) -> None
    def update_position(market_id, fill) -> None
```
- ~~`OracleState`: pydantic model with `bankroll`, `positions`, `trade_history`, `priors`~~
- ~~Atomic write: write to `.tmp` then `os.replace()` — never corrupt state on crash~~
- ~~**Test:** Simulate crash mid-write; state file remains valid on reload~~

**4.2 — Paper execution (`src/execution/paper.py`)** ✅ *Completed 2026-03-15*
- ~~Simulate Fill-or-Kill: fills at `p_ask` if size ≤ available liquidity × 0.70, else partial fill~~
- ~~Simulate partial fills: fill 60–90% of requested size (random within range)~~
- ~~Apply commission on close~~
- ~~Update bankroll and positions in StateManager~~
- ~~Log every trade with: timestamp, market_id, direction, requested_size, filled_size, price, edge, p_fair~~

**4.3 — Main loop integration (`main.py`)** ✅ *Completed 2026-03-15*
- ~~Wire all components: Scanner → Enrichment → LLM → Bayesian → Edge → Kelly → Risk → Execution → State → Dashboard~~
- ~~Add CLI flag `--paper` (default) vs `--live` (Phase 6 only)~~
- ~~Graceful exception handling: any unhandled exception in one cycle logs error and continues to next cycle; does not crash the agent~~

**4.4 — Streamlit dashboard (`src/dashboard/app.py`)** ✅ *Completed 2026-03-15*
- ~~All panels from whitepaper §7~~
- ~~Auto-refresh every 60 seconds (use `st.rerun()` with `time.sleep(60)`)~~
- ~~Reads from `state/oracle_state.json` (no direct coupling to running agent)~~
- ~~Run separately: `streamlit run src/dashboard/app.py`~~

**4.5 — Integration test**
- Run paper trading for 48 hours
- Verify: ≥ 3 trades executed, P&L tracked correctly, no state corruption

**4.6 — Betfair paper scanner** ✅ *Completed 2026-03-15*
- ~~`get_markets(country_codes=["AU"])` → list of market dicts~~
- ~~`get_market_detail(market_id)` → dict with `id`, `question`, `probability`, `volume`, `url`, `totalLiquidity`, `isResolved`, `resolution`~~
- ~~`probability = 1 / best_back_price`; `totalLiquidity = sum(back_sizes + lay_sizes)`; `isResolved = status in {CLOSED, SETTLED}`~~
- ~~Interactive login via `betfairlightweight` (no cert files required)~~
- ~~`PaperBroker` uses Betfair market data directly~~
- ~~Settlement uses `_settle_betfair_positions()` in `main.py`~~
- ~~18 unit tests in `tests/test_betfair_scanner.py` (mocked API, no network required)~~

**Phase 4 Exit Gate:** ✅ *Completed 2026-03-23*
- [x] 50 paper trades logged over ≥ 1 week
- [x] P&L accounting matches manual check on 5 random trades (all 38 settled trades verified with zero mismatches)
- [x] Agent survives 72 hours continuous running without crash
- [x] Dashboard displays all panels correctly

**Phase 4B — Betfair-Realistic Paper Trading** ✅ *Completed 2026-03-23*

Paper trading P&L audit revealed 56% of profit came from extreme-odds lays (1.01–1.05) with no real-world liquidity, and fills assumed zero slippage. Five layers of realism added:

1. **Order book depth**: `get_market_detail()` upgraded to `EX_ALL_OFFERS` (full price ladder). `_depth_fill()` walks the ladder level-by-level, computing VWAP fill price and actual fill fraction.
2. **Slippage model**: `_apply_slippage()` degrades fill price proportional to `order_size / available_liquidity`. Supports `linear` and `sqrt` models. Edge rechecked after slippage — trade skipped if edge destroyed.
3. **Pre-trade gates**: Minimum liquidity ($50), minimum matched volume ($500), and max lay probability (0.90) filters block unrealistic trades before any LLM spend.
4. **Extreme lay filter**: Lays with implied probability > 90% (decimal odds < 1.11) are blocked entirely.
5. **Trade history backfill**: Script to recover missing resolution/runner_status metadata from settlement logs for trades settled before the persistence fix.

---

## Phase 5: Backtesting & Tuning

**Goal:** Validate parameter choices on historical data before risking real money.

### Tasks

**5.1 — Historical data collection**
- Collect Betfair AU historical resolved markets from paper trading logs
- Store as `data/betfair_historical_YYYY-MM.json`
- Target: ≥ 3 months of resolved markets with price history

**5.2 — Backtest harness (`src/backtesting/harness.py`)**
```python
class BacktestHarness:
    def run(markets, config) -> BacktestResult
    # Replays price history tick-by-tick
    # Uses real Kelly + Edge logic (not paper.py)
    # Returns: trades, equity_curve, sharpe, max_drawdown, fill_rate
```
- No LLM calls during backtest (use pre-computed `sentiment_delta` from logs or synthetic)
- Slippage model: assume 50% of requested size fills at target price

**5.3 — Parameter sweep**
Tune these parameters via grid search:

| Parameter | Range | Step |
|---|---|---|
| `beta` | 0.05 – 0.30 | 0.05 |
| `margin_min_live` | 0.010 – 0.030 | 0.005 |
| `kelly_base_fraction` | 0.25 – 0.75 | 0.125 |
| `news_sentiment_delta` trigger | 0.20 – 0.40 | 0.04 |

- Use 70/30 train/test split on historical data
- Optimize for: Sharpe ratio on test set (not train set — avoid overfitting)
- Record all results in `data/backtest_results.csv`

**5.4 — Calibration validation**
- Probability calibration plot: bin trades by `p_fair`, check actual win rates match
- If consistently overconfident: lower `beta`
- If consistently underconfident: raise `beta` (unlikely for LLM-based priors)

**Phase 5 Exit Gate:**
- [ ] Backtest Sharpe > 0 on held-out test data
- [ ] Max drawdown in backtest < 40%
- [ ] Calibration plot shows no systematic over/underconfidence > 10%
- [ ] Tuned `config.toml` values documented with supporting backtest results

---

## Phase 6: Live Betfair

**Goal:** Real money, small stakes, validated OMS, scale only after 30 days of positive results.

### Tasks

**6.1 — Betfair authentication**
- `betfairlightweight` library
- Credentials in `.env` (never in `config.toml`)
- Test: authenticate, call `list_event_types()`, log result

**6.2 — Market mapper**
- Map event types to Betfair market types (MATCH_ODDS, DRAW_NO_BET, WINNER)
- Start with football (event_type 1) and AFL (event_type 61)
- Already implemented via `betfair_event_types` config and `_SAFE_MARKET_TYPES` filter

**6.3 — OMS state machine (`src/execution/betfair_oms.py`)**
```python
class Order:
    state: OrderState  # Pending/Placed/Partial/Filled/Trailing/Exited/Cancelled
    bet_id: str
    requested_size: float
    filled_size: float
    ...

class OMS:
    def place_fok(market_id, selection_id, side, price, size) -> Order
    def place_limit_with_tif(market_id, selection_id, side, price, size, tif_sec) -> Order
    def cancel(order) -> None
    def reconcile() -> None  # calls /listCurrentOrders and updates local state
```
- Persistence type: always `LAPSE` on all orders
- Synthetic TIF: store order in pending dict; `APScheduler` job cancels after `tif_sec`
- Reconcile after every cycle

**6.4 — Live safeguards (hard-coded, not configurable)**
```python
MAX_SINGLE_BET_AUD = 20.0   # Phase 6 start: AUD 20 max per bet
MAX_DAILY_LOSS_AUD = 100.0  # Kill switch: stop all trading for 24h if hit
MAX_OPEN_POSITIONS = 5      # No more than 5 concurrent positions
```
- These are code-level constants, not config values — a misconfigured `config.toml` cannot override them
- After 30 profitable days, remove/raise these constants deliberately

**6.5 — Kill switch**
- `state/kill_switch.txt`: if this file exists, agent enters read-only mode (no new bets, monitors existing positions to exit)
- Touch the file to halt remotely: `touch state/kill_switch.txt`

**6.6 — Deployment**
- Run as a `systemd` service or Windows Task Scheduler job
- Or: `nohup python main.py --live >> logs/live.log 2>&1 &`
- Dashboard runs separately on a fixed port (default 8501)
- Daily reconciliation email/notification (optional)

**Phase 6 Exit Gate (30-day review):**
- [ ] ≥ 30 live trades executed
- [ ] Fill rate ≥ 70%
- [ ] Realized P&L within ±30% of backtest projection
- [ ] Zero OMS state divergence incidents
- [ ] Max drawdown stayed below 40%
- [ ] If all pass: raise bet limits and remove Phase 6 safeguards

---

## Testing Strategy

### Unit Tests (run on every code change)
```bash
pytest tests/ -v
```

| File | What it covers |
|---|---|
| `test_kelly.py` | All Kelly formulas, liability translation, boundary inputs |
| `test_bayesian.py` | Logit update, beta sensitivity, boundary probabilities |
| `test_edge.py` | Edge signs, commission adjustment |
| `test_risk.py` | Each gate fires independently |
| `test_oms.py` | State machine transitions, synthetic TIF cancellation |

### Integration Tests (run before phase transitions)
- Phase 3→4: Run `main.py --paper` for 1 hour; assert no exceptions, ≥ 1 scan logged
- Phase 5→6: Run backtest on held-out data; assert Sharpe > 0

### Property-Based Testing (recommended for Kelly)
```python
# Using hypothesis library
@given(p_fair=st.floats(0.01, 0.99), q_market=st.floats(0.01, 0.99), gamma=st.floats(0, 0.10))
def test_kelly_never_exceeds_one(p_fair, q_market, gamma):
    f = commission_aware_kelly(p_fair, q_market, gamma)
    assert f <= 1.0
```

---

## Critical Implementation Order

Do not reorder these. Each step depends on the previous.

```
config.py ──► scanner/betfair_scanner.py ──► llm/client.py ──► llm/prompts.py
    │
    └──► strategy/edge.py ──► strategy/bayesian.py ──► strategy/kelly.py
                                                              │
                                                    risk/manager.py
                                                              │
                                                    storage/state_manager.py
                                                              │
                                              execution/paper.py ──► main.py (wired)
                                                              │
                                                    dashboard/app.py
                                                              │
                                                    backtesting/harness.py
                                                              │
                                                    execution/betfair_oms.py
```

---

## Go/No-Go Checklist Before Live Money

- [ ] All 5 test files pass with zero failures
- [ ] Paper trading ran for ≥ 1 week with no state corruption
- [ ] Backtest Sharpe > 0 on held-out data
- [ ] Calibration plot reviewed and `beta` tuned
- [ ] Kill switch tested: touching `state/kill_switch.txt` halts new bets within one cycle
- [ ] OMS reconciliation tested: manually place a Betfair order in test market, confirm agent detects and tracks it
- [ ] Max drawdown in paper + backtest phases both < 40%
- [ ] `MAX_SINGLE_BET_AUD = 20.0` hard limit verified in code (not just config)
- [ ] `.env` is in `.gitignore` and not committed
- [ ] Dashboard shows real-time P&L, drawdown, and LLM cost before first live bet


---

## Phase 5A: Intelligence Layer Upgrade (NEW — Added March 2026)

**Goal:** Replace LLM-headline signal with a multi-source intelligence stack that has measurable edge. The current LLM-headline approach validates the pipeline but is unlikely to produce sustained profit against Betfair exchange participants. This phase upgrades the brain while keeping the body intact.

### Context: Why This Phase Was Added

During Phase 4 paper trading, the following was observed:
- The pipeline is mechanically sound — scanner, enrichment, LLM, Bayesian updater, Kelly sizing, risk gates, execution, settlement, and dashboard all work end-to-end.
- However, the LLM intelligence layer (Grok-fast reading news headlines) produces signals that the market has already priced in. Betfair exchange prices reflect the collective knowledge of professional syndicates with statistical models, lineup scrapers, and years of domain data. An LLM reading day-old headlines cannot consistently outperform this.
- The "longshot bias" the LLM frequently cites is a real phenomenon in bookmaker markets but is significantly weaker on Betfair (an exchange where participants trade against each other).
- Without a stronger signal source, live money deployment would be gambling with sophisticated accounting — not edge-based trading.

### Tasks

**5A.1 — Closing Line Value (CLV) Tracking** ✅ *Implemented 2026-03-21 (commit 82bd5db)*

Add a diagnostic metric to every trade that measures whether Oracle's entry price was better than the final pre-kickoff price (closing line). This is the industry standard for measuring betting edge.

```python
# In state_manager.py — add to Trade model:
closing_price: Optional[float] = None    # final Betfair price at market suspension
clv: Optional[float] = None             # entry_price - closing_price (positive = good)
```

**Implementation:**
- At settlement, fetch the market's final probability before suspension from Betfair
- Compute `CLV = entry_implied_prob - closing_implied_prob` for backs (`closing - entry` for lays)
- Positive CLV = Oracle entered at a better price than the market closed at
- Track aggregate CLV in the dashboard as a new panel

**Why this matters:** If aggregate CLV is negative after 100+ trades, the signal is not profitable regardless of short-term P&L variance. If CLV is positive, the signal has real edge even if individual trades lose. This answers the fundamental question: "Does Oracle know something the market doesn't?"

**Exit gate:** Run for 100+ paper trades. If aggregate CLV ≤ 0, proceed to 5A.2 (upgrade signal). If CLV > 0, the current signal may have edge — proceed to standard Phase 5 backtesting.

---

**5A.2 — Statistical Data Integration**

Replace LLM-headlines as the primary signal with actual sports statistics. Use lightweight, no-ML models that run deterministically without training data.

**Data sources:**
- `football-data.org` — free, covers major football leagues (EPL, Bundesliga, Serie A, La Liga, etc.). Provides goals scored/conceded per team per season. No API key required for CSV data.
- `Squiggle API` — free, covers AFL. Provides match results, tips, and team ratings. No API key required.

**New module:** `src/enrichment/stats.py`
```python
def get_football_stats(home_team: str, away_team: str, league: str) -> FootballStats:
    """Fetch goals scored/conceded averages from football-data.org CSV files."""
    ...

def get_afl_stats(home_team: str, away_team: str) -> AFLStats:
    """Fetch recent form and margin data from Squiggle API."""
    ...

class FootballStats(BaseModel):
    home_goals_scored_avg: float   # avg goals scored at home
    home_goals_conceded_avg: float # avg goals conceded at home
    away_goals_scored_avg: float   # avg goals scored away
    away_goals_conceded_avg: float # avg goals conceded away
    league_avg_goals: float        # league average goals per game
    ...

class AFLStats(BaseModel):
    home_elo: float                # Squiggle ELO rating
    away_elo: float
    home_recent_margin_avg: float  # avg margin last 5 games
    away_recent_margin_avg: float
    ...
```

**New module:** `src/enrichment/team_mapping.py`
```python
def betfair_to_stats_name(betfair_name: str, sport: str) -> str:
    """Map Betfair runner names to football-data.org / Squiggle team names.
    Uses fuzzy matching with a manual override dict for known mismatches.
    Falls back to Perplexity Sonar when fuzzy matching fails — asks Perplexity
    to match the Betfair name against the full football-data.org team list.
    Results (hits and confirmed misses) are cached persistently to
    state/team_mapping_cache.json so each team is only queried once."""
    ...
```

**New module:** `src/strategy/statistical_model.py`
```python
def predict_football_odds(stats: FootballStats) -> dict[str, float]:
    """Return {"home": p, "draw": p, "away": p} using Independent Poisson model.
    No ML training required — uses attack/defence strength ratios to derive
    lambda_home and lambda_away, then sums Poisson PMF grid for match outcomes."""
    ...

def predict_afl_odds(stats: AFLStats) -> dict[str, float]:
    """Return {"home": p, "away": p} using logistic regression on ELO difference.
    Coefficients derived analytically from historical ELO-vs-outcome data."""
    ...
```

**LLM change:** Switched `fast_model` from `x-ai/grok-4.1-fast` to `google/gemini-2.0-flash` via OpenRouter. Reasons:
- ~50% cheaper per token than Grok-fast
- Native structured output support via `response_format: { type: "json_schema", json_schema: ... }` — eliminates JSON parse failures
- Equivalent quality for the qualitative overlay role (LLM is no longer the primary signal)

**Prompt reframing:** The LLM's role is now explicitly a *qualitative overlay on a statistical prior*, not the primary signal source. Prompts are updated to:
- Receive the statistical model's `p_model` and the reasoning behind it
- Focus on factors statistics cannot capture: injuries, weather, travel, morale, managerial changes
- Return `sentiment_delta` as an adjustment to the statistical prior, not an independent probability estimate

**Integration with existing pipeline:**
- The statistical model produces `p_model` (replaces `p_mid` as the Bayesian prior)
- The LLM still runs but now receives statistical context alongside news
- The Bayesian updater applies the LLM's `sentiment_delta` on top of `p_model` instead of `p_mid`
- This means: `p_fair = expit(logit(p_model) + beta * sentiment_delta)`

**Why this is better:** An Independent Poisson model produces calibrated match probabilities from goal averages alone — no training data or ML pipeline required. The LLM adds qualitative overlays (injuries, morale, travel fatigue) that statistics miss. Neither alone is sufficient; together they cover both quantitative and qualitative edges.

---

**5A.3 — Pre-Match Lineup Ingestion (Optional — Time-Sensitive Alpha)**

Team lineups are announced 60–90 minutes before kickoff. This is a genuine informational edge window: the market takes time to fully adjust to unexpected lineup changes.

**New module:** `src/enrichment/lineups.py`
```python
def get_lineups(match_id: str) -> LineupData | None:
    """Fetch announced lineups via Sportmonks/API-Football.
    Returns None if lineups not yet announced."""
    ...

def lineup_surprise(announced: list[str], expected: list[str]) -> list[str]:
    """Return list of surprise inclusions/exclusions."""
    ...
```

**Integration:** When lineups are available, the deep trigger fires automatically and the LLM receives the lineup diff as additional context. This is where LLMs genuinely add value — interpreting *significance* of changes.

**Timing:** This requires the scan interval to be reduced to 5–10 minutes during the 90-minute pre-match window. Add a `pre_match_fast_scan` mode that activates when events are imminent.

---

**Phase 5A Exit Gate:**
- [x] CLV tracking implemented and running for 100+ paper trades
- [ ] If CLV ≤ 0: statistical model integrated and producing calibrated probabilities
- [ ] Statistical model calibration plot shows < 10% systematic error across probability bins
- [x] LLM prompts updated to receive statistical model output as context
- [x] At least one sports data API integrated and returning match-level statistics (football-data.org, Squiggle, basketball API)
- [ ] Agent paper-trades with upgraded signal for ≥ 1 week before proceeding to Phase 5 backtesting

---

## Updated Go/No-Go Checklist Before Live Money

*Replaces the original checklist — adds CLV and signal-quality requirements.*

- [ ] All test files pass with zero failures
- [ ] Paper trading ran for ≥ 2 weeks with no state corruption
- [ ] **CLV is positive across 100+ paper trades** (most critical gate)
- [ ] Backtest Sharpe > 0 on held-out data
- [ ] Calibration plot reviewed — statistical model + LLM overlay within 10% across all bins
- [ ] Kill switch tested: touching `state/kill_switch.txt` halts new bets within one cycle
- [ ] OMS reconciliation tested: manually place a Betfair order in test market, confirm agent detects and tracks it
- [ ] Max drawdown in paper + backtest phases both < 30%
- [ ] `MAX_SINGLE_BET_AUD = 20.0` hard limit verified in code (not just config)
- [ ] `MAX_DAILY_LOSS_AUD = 100.0` kill switch verified in code
- [ ] `.env` is in `.gitignore` and not committed
- [ ] Dashboard shows real-time P&L, drawdown, CLV, and LLM cost before first live bet
- [ ] Statistical model is integrated — LLM is NOT the sole signal source
- [ ] Market type filter restricts to MATCH_ODDS, DRAW_NO_BET, WINNER only (no multi-runner derivative markets until runner mapping is properly implemented)

---

## Lessons Learned During Phase 4 (March 2026)

Documenting issues discovered during paper trading for future reference:

1. **LLM delta sign confusion on Under/Over markets.** The LLM doesn't inherently know that `sentiment_delta > 0` should mean "this selection is more likely." For markets like "Under 7.5 Goals" at low implied probability, the LLM returned negative delta (meaning "too low") when positive delta was correct. **Fix:** Explicit direction instruction in the prompt telling the LLM what positive/negative delta means for the specific selection.

2. **Runner mapping ambiguity on multi-runner markets.** Betfair Over/Under, Asian Handicap, Correct Score, and BTTS markets have multiple runners. The scanner grabbed the first runner's price, which didn't necessarily correspond to the named selection. **Fix (interim):** Restrict scanning to `MATCH_ODDS`, `DRAW_NO_BET`, and `WINNER` market types where runner identity is unambiguous. **Fix (permanent — Phase 5A):** Match runners by `selection_id` across catalogue and book responses.

3. **Kelly oversizing on extreme edges.** Raw Kelly fractions > 0.90 occurred when the delta sign bug produced extreme `p_fair` divergences. The 25% hard cap was insufficient; two consecutive max-sized trades consumed 40% of bankroll. **Fix:** Reduced hard cap to 10% for paper trading. Added absolute dollar cap ($100 per trade).

4. **Betfair session expiry.** Interactive login sessions expire after several hours. The scanner failed silently, causing all subsequent cycles to error. **Fix:** Health check on every `_get_client()` call with transparent re-authentication on failure.

5. **OneDrive file locking.** Atomic writes (`write .tmp → rename .json`) failed intermittently due to OneDrive syncing. **Fix:** Retry loop with exponential backoff (5 attempts, 0.5s–2.5s delays).

6. **News enrichment returned empty for niche markets.** NewsData.io had zero coverage for Australian greyhound racing, regional horse meets, and niche sports futures. The LLM defaulted to `delta=0` with no information, producing zero edge. **Fix:** Google News RSS fallback, query rewriting for Betfair market names, and event type filtering to focus on major sports with news coverage.

