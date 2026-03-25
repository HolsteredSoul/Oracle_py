# Oracle Python Agent — Development Plan

**v2.0 | March 2026**

---

## Overview

Sequential phases with explicit exit gates. Each gate must pass before advancing. Paper trading precedes live money.

| Phase | Name | Status | Exit Gate |
|-------|------|--------|-----------|
| 1 | Foundation | ✅ Complete | Project boots, reads config, polls Betfair AU |
| 2 | Intelligence | ✅ Complete | LLM calls return valid JSON, Bayesian updater works |
| 3 | Strategy & Risk | ✅ Complete | Kelly math validated by unit tests, risk gates fire correctly |
| 4 | Paper Trading | ✅ Complete | 50+ paper trades logged, P&L tracking accurate |
| 4B | Betfair-Realistic Fills | ✅ Complete | Depth-aware VWAP, slippage model, pre-trade gates |
| 4C | Advanced Realism | ✅ Complete | Queue-position model, in-play gate, crossed-book detection |
| 5A | Intelligence Upgrade | In Progress | CLV tracking, statistical model, team mapping |
| 5 | Backtesting & Tuning | Not Started | Sharpe > 0 on held-out data |
| 6 | Live Betfair | Not Started | OMS tested, small stakes, scale after 30 days |

**Current test count: 220 passing.**

---

## Phase 1: Foundation ✅

*Completed 2026-03-14*

**Goal:** Working project skeleton — config, scanner, scheduler.

- `src/config.py` — Pydantic-validated config from `config.toml` + `.env`
- `src/scanner/betfair_scanner.py` — `get_markets()` and `get_market_detail()` via betfairlightweight
- `main.py` — APScheduler with adaptive scan cycle (10-45 min), priority sort, rejection cache, graceful shutdown
- Stdlib logging with rotating file handler

**Exit gate passed:** Agent ran 1 hour without errors, regular scan cycles logged, config validation catches bad values.

---

## Phase 2: Intelligence Layer ✅

*Completed 2026-03-14*

**Goal:** LLM integration with cost tracking and enrichment.

- `src/llm/client.py` — OpenRouter wrapper with tier routing, daily spend cap, auto-downgrade, tenacity retries
- `src/llm/prompts.py` — JSON-forced prompt templates for light scan and deep trigger
- `src/llm/models.py` — Pydantic response validation (malformed LLM output returns None, never crashes)
- `src/enrichment/news.py` — NewsData.io + Google News RSS
- `src/enrichment/x_sentiment.py` — X API v2 recent search
- `src/enrichment/trigger.py` — Event-driven deep trigger decision

**Exit gate passed:** 10 light scans + 2 deep triggers return valid JSON, spend counter works, bad responses handled gracefully.

---

## Phase 3: Strategy & Risk ✅

*Completed 2026-03-15*

**Goal:** All math correct and covered by unit tests.

- `src/strategy/edge.py` — `executable_edge()` for back and lay
- `src/strategy/bayesian.py` — Logit-space probability updater
- `src/strategy/kelly.py` — Commission-aware Kelly + oracle sizing + liability translation
- `src/risk/manager.py` — Pre-trade gates (confidence floor, exposure cap, drawdown throttle)

**Exit gate passed:** 31 unit tests pass including hypothesis property tests. No boundary failures.

---

## Phase 4: Paper Trading ✅

*Completed 2026-03-23*

**Goal:** Full end-to-end cycle on Betfair AU with logged P&L.

- `src/storage/state_manager.py` — OracleState with atomic JSON persistence
- `src/execution/paper.py` — PaperBroker with FOK simulation, partial fills, settlement, cancellation
- `main.py` — Full pipeline wired: Scanner → Enrichment → LLM → Bayesian → Edge → Kelly → Risk → Execution → State
- `src/dashboard/app.py` — Streamlit dashboard (equity curve, cash curve, P&L, positions, trade log, scan feed, CLV, LLM cost, realism checklist)
- Kill switch: `touch state/kill_switch.txt`

**Exit gate passed:** 50+ paper trades logged over 1+ week, P&L verified on all 38 settled trades (zero mismatches), 72h continuous run without crash, dashboard displays all panels.

### Phase 4B — Betfair-Realistic Fills ✅

*Completed 2026-03-23*

Paper trading P&L audit revealed 56% of profit came from extreme-odds lays (1.01–1.05) with zero real-world liquidity. Five layers of realism added:

1. **Order book depth** — `get_market_detail()` upgraded to `EX_ALL_OFFERS`. `_depth_fill()` walks the ladder level-by-level computing VWAP fill price and actual fill fraction.
2. **Slippage model** — `_apply_slippage()` degrades price proportional to order size / liquidity. Linear and sqrt models. Edge re-checked post-slippage — trade skipped if edge destroyed.
3. **Pre-trade gates** — Minimum liquidity ($50), minimum matched volume ($500), max lay probability (0.90).
4. **Extreme lay filter** — Lays at implied probability > 90% blocked entirely.
5. **Trade history backfill** — Script to recover missing resolution metadata for pre-fix trades.

### Phase 4C — Advanced Realism ✅

*Completed 2026-03-23*

Queue-position model and safety gates to close the paper-vs-live gap:

1. **Queue-position model** — `_depth_fill()` simulates order queue friction. Modes: `"none"` (optimistic), `"linear"` (volume discount per level), `"probabilistic"` (Bernoulli draw — default). New `[paper]` config section.
2. **In-play safety gate** — Blocks all in-play markets (`allow_in_play = false`). Critical for Phase 6.
3. **Crossed-book detection** — Skips markets where `best_back > best_lay` (stale/suspended data).
4. **Fill-rate tracking** — Logs `filled_fraction` per trade for calibration.
5. **Dashboard realism panel** — Scored feature checklist (X/10).

Phase 4C exit criteria (evaluate after 100+ trades under new model):
- Queue-position model active
- In-play gate and crossed-book detection active
- Mean fill fraction ≤ 0.92
- All tests pass (220 tests including 10 new Phase 4C tests)

---

## Phase 5A: Intelligence Upgrade (In Progress)

**Goal:** Replace LLM-headline signal with a multi-source intelligence stack that has measurable edge.

### Context

During Phase 4, the pipeline proved mechanically sound but the LLM intelligence layer (reading news headlines) produces signals the market has already priced in. Betfair exchange prices reflect professional syndicates with statistical models, lineup scrapers, and decades of domain data. Without a stronger signal source, live deployment would be gambling with sophisticated accounting.

### 5A.1 — Closing Line Value (CLV) Tracking ✅

*Implemented 2026-03-21*

Diagnostic metric measuring whether Oracle's entry price beats the closing line (industry standard for betting edge).

- `Trade` model extended with `closing_price` and `clv` fields
- `last_seen_price` updated each scan cycle, becomes `closing_price` at settlement
- CLV panel in dashboard with aggregate metrics and Phase 5A.1 exit gate (100+ trades, CLV > 0)
- Back CLV = `closing_price - entry_price`; Lay CLV = `entry_price - closing_price`

### 5A.2 — Statistical Data Integration ✅

*Implemented 2026-03-22*

Statistical model replaces market mid-price as Bayesian prior. LLM becomes a qualitative overlay.

**Data sources:**
- **football-data.org** — European football: goals scored/conceded, attack/defence strength
- **Squiggle API** — AFL: predicted margins, win probabilities, team ratings
- **basketball-api-sports.io** — NBA and international basketball

**Modules:**
- `src/enrichment/stats.py` — Multi-sport stats fetcher with caching and completeness scoring
- `src/enrichment/team_mapping.py` — Fuzzy name resolution (index + Perplexity Sonar fallback + persistent disk cache)
- `src/strategy/statistical_model.py` — Independent Poisson model (football) / logistic model (AFL/basketball)

**LLM reframing:**
- Switched `fast_model` to Gemini 2.5 Flash Lite (~50% cheaper, native structured output)
- `deep_model` is Claude Sonnet 4.6 for edge-candidate deep analysis
- Prompts now receive `p_model` as statistical context; LLM provides `sentiment_delta` adjustment for qualitative factors only
- Perplexity Sonar web search validates edge candidates before execution

**Pipeline change:** `p_fair = expit(logit(p_model) + β · sentiment_delta)` — statistical model is the prior, LLM delta is the adjustment.

### 5A.3 — Adaptive Scheduling & Scan Feed ✅

*Implemented 2026-03-25*

Replaced the fixed 30-min/15-market scan regime with intelligent scheduling and full scan observability.

**Adaptive scan interval:**
- Interval dynamically adjusts between 10-45 minutes based on market density
- Busy weekends (60-100+ markets): tightens to 10-20 min intervals
- Quiet weekdays (10-20 markets): relaxes to 45 min intervals
- Uses `scheduler.reschedule_job()` at the end of each cycle

**Market prioritisation:**
- Replaced random shuffle with priority sort: time-to-kickoff (≤3h urgent, ≤12h active, >12h early) → matched volume → available liquidity
- Ensures near-kickoff, high-volume markets are always analysed first

**Rejection cache:**
- In-memory TTL cache (2h default) for markets that fail hard gates (volume, liquidity, niche, in-play, crossed book)
- Markets with soft failures (no_edge, negative_kelly) are NOT cached — prices move
- Prevents wasting LLM calls on markets whose conditions haven't changed

**Scan feed dashboard panel:**
- `ScanFeedWriter` records per-market outcomes to `state/scan_feed.json` (48-cycle rolling window)
- Dashboard panel with cycle selector, outcome badges (colour-coded), and detailed entries table
- Outcomes tracked: `traded`, `no_edge`, `edge_lost`, `skipped_volume`, `skipped_liquidity`, `skipped_niche`, `skipped_inplay`, `skipped_crossed`, `skipped_divergence`, `skipped_llm_fail`, `negative_kelly`, `risk_blocked`

**Football team index disk cache:**
- Football-data.org 316-team index cached to `state/fd_team_index.json` with TTL
- Saves 9 API calls per agent restart

**Max markets per cycle raised from 15 → 25** (configurable via `config.toml`).

### 5A.4 — Pre-Match Lineup Ingestion (Not Started)

Team lineups announced 60–90 minutes before kickoff create a genuine informational edge window.

- Lineup API integration (Sportmonks or API-Football)
- Compare announced XI to expected XI; trigger deep LLM analysis on surprise changes
- Requires faster scan interval (5–10 min) during pre-match window

### 5A.5 — Reddit Sentiment Enrichment (Not Started)

Reddit communities (r/tennis, r/soccer, r/sportsbook, r/afl, r/nba) provide pre-match discussion and crowd sentiment that the news pipeline misses. Replaces the removed X/Twitter enrichment with a free alternative.

- Reddit API integration (free tier, OAuth2) via `src/enrichment/reddit_sentiment.py`
- Subreddit search by event keywords, sorted by relevance/recency
- Top comment text concatenated and passed to LLM prompt (same pattern as old X enrichment)
- Optional: extract a quantitative momentum score from upvote ratios / comment volume to feed the deep trigger

### Phase 5A Exit Gate

- [x] CLV tracking implemented and running
- [x] LLM prompts updated to receive statistical model output
- [x] At least one sports data API integrated (football-data.org, Squiggle, basketball)
- [x] Adaptive scheduling with priority sort and rejection cache
- [x] Scan feed dashboard panel for full scan observability
- [x] Football team index cached to disk
- [ ] Statistical model calibration plot shows < 10% systematic error across probability bins
- [ ] Agent paper-trades with upgraded signal for ≥ 1 week
- [ ] CLV positive across 100+ paper trades

---

## Phase 5: Backtesting & Tuning (Not Started)

**Goal:** Validate parameter choices on historical data before risking real money.

### Tasks

**5.1 — Historical data collection**
- Collect Betfair AU resolved markets from paper trading logs
- Store as `data/betfair_historical_YYYY-MM.json`
- Target: ≥ 3 months of resolved markets with price history

**5.2 — Backtest harness (`src/backtesting/harness.py`)**
- Replay price history tick-by-tick using real Kelly + Edge logic
- No LLM calls during backtest (use pre-computed `sentiment_delta` from logs)
- Returns: trades, equity curve, Sharpe, max drawdown, fill rate

**5.3 — Parameter sweep**

| Parameter | Range | Step |
|---|---|---|
| `beta` | 0.05 – 0.30 | 0.05 |
| `margin_min_live` | 0.010 – 0.030 | 0.005 |
| `kelly_base_fraction` | 0.25 – 0.75 | 0.125 |
| `news_sentiment_delta` trigger | 0.20 – 0.40 | 0.04 |

- 70/30 train/test split; optimize Sharpe on test set
- Record results in `data/backtest_results.csv`

**5.4 — Calibration validation**
- Probability calibration plot: bin trades by `p_fair`, check actual win rates match
- Tune `beta` based on systematic over/underconfidence

**Phase 5 Exit Gate:**
- [ ] Backtest Sharpe > 0 on held-out test data
- [ ] Max drawdown in backtest < 40%
- [ ] Calibration plot shows no systematic over/underconfidence > 10%
- [ ] Tuned `config.toml` values documented with supporting backtest results

---

## Phase 6: Live Betfair (Not Started)

**Goal:** Real money, small stakes, validated OMS, scale only after 30 days of positive results.

### Tasks

**6.1 — Betfair authentication**
- Certificate-based login via betfairlightweight
- Credentials in `.env` only

**6.2 — OMS state machine (`src/execution/betfair_oms.py`)**
- Order states: Pending → Placed → Partial → Filled → Trailing → Exited / Cancelled
- Persistence type: always LAPSE
- Synthetic TIF via APScheduler cancellation job
- Reconcile against `/listCurrentOrders` after every cycle

**6.3 — Live safeguards (hard-coded, not configurable)**
```python
MAX_SINGLE_BET_AUD = 20.0   # AUD 20 max per bet for first 30 days
MAX_DAILY_LOSS_AUD = 100.0  # Kill switch: stop all trading for 24h
MAX_OPEN_POSITIONS = 5      # Max concurrent positions
```
These are code-level constants — `config.toml` cannot override them. Raised deliberately after 30 profitable days.

**6.4 — Deployment**
- Windows Task Scheduler or systemd service
- Dashboard on separate port (8501)
- Kill switch: `touch state/kill_switch.txt`

**Phase 6 Exit Gate (30-day review):**
- [ ] ≥ 30 live trades executed
- [ ] Fill rate ≥ 70%
- [ ] Realized P&L within ±30% of backtest projection
- [ ] Zero OMS state divergence incidents
- [ ] Max drawdown stayed below 40%
- [ ] If all pass: raise bet limits and remove Phase 6 safeguards

---

## Testing Strategy

```bash
pytest tests/ -v   # 220 tests
```

| File | Coverage |
|---|---|
| `test_paper.py` | Depth fill, slippage, queue model, settlement, cancellation, CLV, integration |
| `test_kelly.py` | Kelly formulas, liability translation, boundary inputs, hypothesis property tests |
| `test_bayesian.py` | Logit update, beta sensitivity, boundary probabilities |
| `test_edge.py` | Edge signs, commission adjustment |
| `test_risk.py` | Each gate fires independently |
| `test_betfair_scanner.py` | Mocked API, market parsing, resolution detection |
| `test_team_mapping.py` | Fuzzy matching, Perplexity fallback, caching |
| `test_stats.py` | Stats fetcher, completeness scoring |
| `test_settlement_validator.py` | P&L consistency checks |
| `test_structured_output.py` | JSON schema generation and validation |

---

## Go/No-Go Checklist Before Live Money

- [ ] All test files pass with zero failures (currently 220)
- [ ] Paper trading ran for ≥ 2 weeks with no state corruption
- [ ] **CLV is positive across 100+ paper trades** (most critical gate)
- [ ] Backtest Sharpe > 0 on held-out data
- [ ] Calibration plot reviewed — statistical model + LLM overlay within 10% across all bins
- [ ] Kill switch tested: `touch state/kill_switch.txt` halts new bets within one cycle
- [ ] OMS reconciliation tested on Betfair test market
- [ ] Max drawdown in paper + backtest phases both < 30%
- [ ] `MAX_SINGLE_BET_AUD = 20.0` hard limit verified in code (not just config)
- [ ] `MAX_DAILY_LOSS_AUD = 100.0` kill switch verified in code
- [ ] `.env` is in `.gitignore` and not committed
- [ ] Dashboard shows real-time P&L, drawdown, CLV, realism score, and LLM cost
- [ ] Statistical model is integrated — LLM is NOT the sole signal source
- [ ] In-play gate active (`allow_in_play = false`)
- [ ] Queue-position model active (`queue_position_model != "none"`)
- [ ] Market type filter restricts to MATCH_ODDS, DRAW_NO_BET, WINNER only

---

## Lessons Learned (Phase 4)

1. **LLM delta sign confusion on Under/Over markets.** LLM didn't know `delta > 0` means "more likely." Fix: explicit direction instruction in prompt.

2. **Runner mapping ambiguity on multi-runner markets.** Scanner grabbed first runner's price regardless of named selection. Fix: restrict to MATCH_ODDS, DRAW_NO_BET, WINNER; match by `selection_id`.

3. **Kelly oversizing on extreme edges.** Raw Kelly > 0.90 from delta sign bugs. Fix: hard cap reduced to 20%, absolute dollar cap ($200 per trade).

4. **Betfair session expiry.** Interactive login expired silently. Fix: health check on every `_get_client()` with transparent re-auth.

5. **OneDrive file locking.** Atomic writes failed due to sync. Fix: retry loop with exponential backoff.

6. **News enrichment empty for niche markets.** Fix: Google News RSS fallback, query rewriting, event type filtering.
