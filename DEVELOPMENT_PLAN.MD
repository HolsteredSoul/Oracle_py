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
| 5A | Intelligence Upgrade | ✅ Complete | CLV tracking, statistical model, team mapping, ESPN integration |
| 5B | Model Calibration + Safety Gates | ✅ Complete | Handshake gate, edge threshold, competition whitelist, DNB gate, time gate |
| 5 | Backtesting & Tuning | Not Started | Sharpe > 0 on held-out data |
| 6 | Live Betfair | Not Started | OMS tested, small stakes, scale after 30 days |

**Current test count: 238 passing.**

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
- All tests pass (238 tests)

---

## Phase 5A: Intelligence Upgrade ✅

*Completed 2026-03-29*

**Goal:** Replace LLM-headline signal with a multi-source intelligence stack that has measurable edge.

### 5A.1 — Closing Line Value (CLV) Tracking ✅

*Implemented 2026-03-21*

- `Trade` model extended with `closing_price` and `clv` fields
- CLV panel in dashboard with aggregate metrics
- Back CLV = `closing_price - entry_price`; Lay CLV = `entry_price - closing_price`

### 5A.2 — Statistical Data Integration ✅

*Implemented 2026-03-22, expanded through 2026-04-04*

Statistical model replaces market mid-price as Bayesian prior.

**Data sources (current):**
- **football-data.org** — European football (8 leagues): goals scored/conceded
- **Official AFL API** — AFL: match results, form, ladder (replaced dead Squiggle API)
- **ESPN hidden API** — NRL, Super Rugby, Premiership, Top 14, URC: standings + form
- **MLB Stats API** — MLB: full season stats
- **NHL API** — NHL: full season stats
- **TheSportsDB** — Super League fallback only (75% completeness)

**Modules:**
- `src/enrichment/stats/` — Per-sport stats modules with shared models
- `src/enrichment/stats/espn.py` — Shared ESPN client for rugby union + NRL
- `src/enrichment/team_mapping.py` — Fuzzy name resolution (threshold 0.85, min 6 chars, Perplexity fallback)
- `src/strategy/statistical_model.py` — Poisson (football) / logistic (all others, coefficient 0.80, clamp [0.20, 0.80])

**Basketball disabled** (2026-04-02): Injury-blind model produced negative CLV on every NBA trade.

### 5A.3 — Adaptive Scheduling & Scan Feed ✅

*Implemented 2026-03-25*

- Adaptive scan interval: 10-45 min based on market density
- Priority sort: time-to-kickoff -> volume -> liquidity
- Rejection cache: 2h TTL for hard gate failures
- Scan feed: 48-cycle rolling window with Win % column and plain-English reasons
- Football team index disk cache (316 teams)

### 5A.4 — Pre-Match Lineup Ingestion (Not Started)

Highest-value remaining signal source. Team lineups announced 60-90 min before kickoff.

### 5A.5 — Reddit Sentiment Enrichment (Not Started)

Free alternative to X/Twitter. Not prioritized -- statistical model + Perplexity provides sufficient signal.

---

## Phase 5B: Model Calibration + Safety Gates ✅

*Completed 2026-04-04*

**Goal:** Close the gap between model predictions and market prices. Add safety gates for every observed failure mode.

### What was done

1. **Handshake gate** (2026-03-28) — Model + Perplexity must agree on direction. Alone converts -$514 into +$183 on 29 trades.
2. **Edge threshold raised 2% -> 8%** (2026-03-28) — Sub-8% trades were noise (2W/3L, -$25).
3. **Team mapping hardened** (2026-03-29) — Fuzzy threshold 0.70 -> 0.85, both teams required, min 6 chars.
4. **Data completeness raised 50% -> 75%** (2026-03-29) — Prevents garbage predictions from partial data.
5. **Model recalibration** (2026-03-29) — Scoring coefficient 1.5 -> 0.80, clamp [0.05, 0.95] -> [0.20, 0.80].
6. **Competition whitelist** (2026-03-29) — Per-sport Betfair queries with competition IDs. Junk leagues can't crowd out viable markets.
7. **ESPN integration** (2026-03-29) — NRL/rugby union from 38% to 100% completeness.
8. **Basketball disabled** (2026-04-02) — Negative CLV, injury-blind model.
9. **3-hour time gate** (2026-04-02) — Markets > 3h from kickoff scanned but not traded.
10. **AFL standings added** (2026-04-04) — Model was ignoring ladder position (only sport missing it).
11. **NRL normalizers tuned** (2026-04-04) — Scoring 15 -> 20, standings 8 -> 9.
12. **DNB conviction gate** (2026-04-04) — Raw match-odds prob >= 55% for Draw No Bet markets.
13. **Low-value competition cull** (2026-04-04) — Dropped Europa, Primeira, BSA, FA Cup, BL2.

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
pytest tests/ -v   # 238 tests
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

- [ ] All test files pass with zero failures (currently 238)
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
- [x] Statistical model is integrated — LLM is NOT the sole signal source
- [x] In-play gate active (`allow_in_play = false`)
- [x] Queue-position model active (`queue_position_model != "none"`)
- [x] Market type filter restricts to MATCH_ODDS, DRAW_NO_BET, WINNER, MONEYLINE only
- [ ] Handshake gate active (model + Perplexity agree)
- [ ] DNB conviction gate active (raw match-odds >= 55%)
- [ ] 3-hour time gate active
- [ ] Competition whitelist covers all traded sports

---

## Lessons Learned (Phase 4)

1. **LLM delta sign confusion on Under/Over markets.** LLM didn't know `delta > 0` means "more likely." Fix: explicit direction instruction in prompt.

2. **Runner mapping ambiguity on multi-runner markets.** Scanner grabbed first runner's price regardless of named selection. Fix: restrict to MATCH_ODDS, DRAW_NO_BET, WINNER; match by `selection_id`.

3. **Kelly oversizing on extreme edges.** Raw Kelly > 0.90 from delta sign bugs. Fix: hard cap reduced to 20%, absolute dollar cap ($200 per trade).

4. **Betfair session expiry.** Interactive login expired silently. Fix: health check on every `_get_client()` with transparent re-auth.

5. **OneDrive file locking.** Atomic writes failed due to sync. Fix: retry loop with exponential backoff.

6. **News enrichment empty for niche markets.** Fix: Google News RSS fallback, query rewriting, event type filtering.
