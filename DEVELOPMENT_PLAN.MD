# Oracle Python Agent — Development Plan

**v1.0.0 | March 2026**

---

## Overview

Six sequential phases. Each phase has a clear exit gate — a set of conditions that must be true before moving to the next phase. No phase is skipped. Paper trading precedes live money.

| Phase | Name | Exit Gate |
|---|---|---|
| 1 | Foundation | Project boots, reads config, polls Manifold |
| 2 | Intelligence | LLM calls return valid JSON, Bayesian updater works |
| 3 | Strategy & Risk | Kelly math validated by unit tests, risk gates fire correctly |
| 4 | Paper Trading | 50+ paper trades logged, P&L tracking accurate |
| 5 | Backtesting & Tuning | Parameters tuned, Sharpe > 0 on held-out data |
| 6 | Live Betfair | OMS tested, small stakes deployed, scale after 30 days |

---

## Phase 1: Foundation

**Goal:** Working project skeleton that boots, loads config, and fetches live Manifold data.

### Tasks

**1.1 — Project setup**
```
oracle-py/
├── main.py              (empty scheduler stub)
├── config.toml          (copy from whitepaper §5)
├── .env                 (empty template)
├── requirements.txt     (copy from whitepaper §10)
├── .gitignore
└── src/__init__.py
```
- Create virtualenv: `python -m venv .venv`
- Install deps: `pip install -r requirements.txt`
- Verify imports all resolve

**1.2 — Config loader (`src/config.py`)**
- Use `pydantic-settings` with `BaseSettings`
- Load `config.toml` via `toml.load()` + `.env` via `python-dotenv`
- All config values typed and validated at startup
- Fail fast with clear error if required keys missing
- **Test:** `python -c "from src.config import settings; print(settings)"` succeeds

**1.3 — Manifold scanner (`src/scanner/manifold.py`)**
- `get_markets(limit=100, min_volume=500)` → list of market dicts
- `get_market_detail(market_id)` → full order book with best ask/bid
- Filter: exclude resolved, exclude near-boundary (prob < 0.05 or > 0.95)
- HTTP client: `httpx` with 10s timeout, no auth required
- **Test:** Prints 5 live markets with current prices

**1.4 — Logging setup**
- `loguru` or stdlib `logging` with rotating file handler (daily, 7-day retention)
- Structured log format: `timestamp | level | module | message`
- All modules use `logging.getLogger(__name__)`

**1.5 — Scheduler stub (`main.py`)**
- `APScheduler` `BackgroundScheduler` with one job: scan every 30 minutes
- Job just logs: `"Scan cycle started, found N markets"`
- Graceful shutdown on Ctrl+C

**Phase 1 Exit Gate:**
- [ ] `python main.py` runs without errors for 1 hour
- [ ] Logs show regular 30-min scan cycles with Manifold market counts
- [ ] Config validation catches a bad value and prints a clear error

---

## Phase 2: Intelligence Layer

**Goal:** LLM calls routed correctly, returning valid JSON, with cost tracking.

### Tasks

**2.1 — OpenRouter client (`src/llm/client.py`)**
- `call_llm(prompt, tier="fast")` → parsed JSON dict
- Tier routing: `fast` → config `fast_model`, `deep` → config `deep_model`
- Daily spend tracking: read/write `state/llm_spend.json`; auto-downgrade at 80% of cap
- Retry: `tenacity` with exponential backoff on 429, 500, 503
- Timeout: 30s per call
- **Test:** Single call returns `{"sentiment_delta": ..., "uncertainty_penalty": ..., "rationale": ...}`

**2.2 — Prompt templates (`src/llm/prompts.py`)**
- `build_light_scan_prompt(question, mid_price, news_summary)` → str
- `build_deep_trigger_prompt(question, mid_price, news_summary, x_summary)` → str
- `build_exit_prompt(position, updated_news, current_price)` → str
- Each function enforces JSON-only output instruction in system message
- **Test:** All three prompts render without errors for sample inputs

**2.3 — Response validation**
- Pydantic models: `LightScanResponse`, `DeepTriggerResponse`, `ExitResponse`
- On parse failure: log the raw response, return `None` (caller skips trade)
- Never let a malformed LLM response crash the main loop

**2.4 — Enrichment: News (`src/enrichment/news.py`)**
- `get_news_summary(query, max_articles=5)` → str
- NewsData.io API via `httpx`
- Summarize: return top 5 headline+description pairs joined as one string
- Cache per query per hour (simple dict in memory)

**2.5 — Enrichment: X/Twitter (`src/enrichment/x_sentiment.py`)**
- `get_x_summary(keywords, max_tweets=20)` → str
- X API v2 recent search endpoint via `httpx`
- Rate-limit aware: track remaining requests from response headers
- Return recent tweets joined as summary string; return empty string on rate limit hit (non-fatal)

**2.6 — Trigger detector**
- `should_trigger_deep(sentiment_delta, volatility_z, x_momentum)` → bool
- Pure function, testable without network calls
- Reads thresholds from config

**Phase 2 Exit Gate:**
- [ ] 10 consecutive light scan calls succeed and return valid JSON
- [ ] 2 deep trigger calls succeed and return valid JSON
- [ ] Daily spend counter increments correctly and logs warning near cap
- [ ] A bad LLM response (simulate by mocking) does not crash the loop

---

## Phase 3: Strategy & Risk

**Goal:** All math is correct and covered by unit tests. No live money logic yet.

### Tasks

**3.1 — Edge calculator (`src/strategy/edge.py`)**
```python
def executable_edge(p_fair, p_ask, p_bid, direction) -> float:
    # direction: "back" or "lay"
    # Returns signed edge; negative means no edge
```
- Commission not applied here (applied in Kelly)
- **Tests:** Back edge positive when p_fair > ask; lay edge positive when p_fair < bid; edge = 0 at breakeven

**3.2 — Bayesian updater (`src/strategy/bayesian.py`)**
```python
def update_probability(p_mid, sentiment_delta, beta) -> float:
    # Returns p_fair in (0, 1) by construction
    # Uses scipy.special.logit / expit
```
- Handle edge cases: `p_mid` near 0 or 1 (clamp to [0.001, 0.999] before logit)
- `beta = 0` must return `p_mid` exactly
- **Tests:** Positive delta increases probability; negative decreases; β=0 is identity; boundary probs handled

**3.3 — Kelly calculator (`src/strategy/kelly.py`)**
```python
def commission_aware_kelly(p_fair, q_market, gamma) -> float:
    # p_fair: our estimate
    # q_market: implied prob from market price (1/decimal_odds)
    # gamma: commission rate (e.g. 0.05)
    # Returns f* (uncapped); caller applies scaling + caps

def apply_oracle_sizing(f_star, conf_score, drawdown_pct, config) -> float:
    # Applies k, lambda_conf, lambda_dd, and hard caps
    # Returns final f (fraction of bankroll to risk)

def translate_to_betfair(f, bankroll, decimal_odds, direction) -> dict:
    # Returns {"stake": ..., "liability": ...}
    # direction: "back" or "lay"
```
- **Critical assertions:** `f_star > 1.0` raises `ValueError`; `f_final > 0.25` raises `ValueError`
- **Tests:**
  - Zero edge → f* = 0
  - Negative edge → f* < 0 (caller skips)
  - Commission = 0 reduces to standard Kelly
  - Lay translation: stake = liability / (odds - 1)
  - Drawdown > 20% halves output
  - conf_score 50 gives λ_conf = 0.5

**3.4 — Risk manager (`src/risk/manager.py`)**
```python
def check_risk_gates(f, conf_score, current_exposure, drawdown_pct, config) -> tuple[bool, str]:
    # Returns (approved: bool, reason: str)
    # Checks: confidence floor, exposure cap, drawdown throttle
```
- **Tests:** Each gate fires independently; approved path returns True with "OK" reason

**Phase 3 Exit Gate:**
- [ ] `pytest tests/test_kelly.py tests/test_bayesian.py tests/test_edge.py tests/test_risk.py` all pass
- [ ] No assertions fail on any boundary input
- [ ] Code review: no variable naming collisions (commission vs confidence)

---

## Phase 4: Paper Trading

**Goal:** Full end-to-end cycle running on Manifold paper money with logged P&L.

### Tasks

**4.1 — State manager (`src/storage/state_manager.py`)**
```python
class StateManager:
    def load() -> OracleState
    def save(state: OracleState) -> None  # atomic write (write tmp, rename)
    def add_trade(trade: Trade) -> None
    def update_position(market_id, fill) -> None
```
- `OracleState`: pydantic model with `bankroll`, `positions`, `trade_history`, `priors`
- Atomic write: write to `.tmp` then `os.replace()` — never corrupt state on crash
- **Test:** Simulate crash mid-write; state file remains valid on reload

**4.2 — Paper execution (`src/execution/paper.py`)**
- Simulate Fill-or-Kill: fills at `p_ask` if size ≤ available liquidity × 0.70, else partial fill
- Simulate partial fills: fill 60–90% of requested size (random within range)
- Apply commission on close
- Update bankroll and positions in StateManager
- Log every trade with: timestamp, market_id, direction, requested_size, filled_size, price, edge, p_fair

**4.3 — Main loop integration (`main.py`)**
- Wire all components: Scanner → Enrichment → LLM → Bayesian → Edge → Kelly → Risk → Execution → State → Dashboard
- Add CLI flag `--paper` (default) vs `--live` (Phase 6 only)
- Graceful exception handling: any unhandled exception in one cycle logs error and continues to next cycle; does not crash the agent

**4.4 — Streamlit dashboard (`src/dashboard/app.py`)**
- All panels from whitepaper §7
- Auto-refresh every 60 seconds (use `st.rerun()` with `time.sleep(60)`)
- Reads from `state/oracle_state.json` (no direct coupling to running agent)
- Run separately: `streamlit run src/dashboard/app.py`

**4.5 — Integration test**
- Run paper trading for 48 hours
- Verify: ≥ 3 trades executed, P&L tracked correctly, no state corruption

**Phase 4 Exit Gate:**
- [ ] 50 paper trades logged over ≥ 1 week
- [ ] P&L accounting matches manual check on 5 random trades
- [ ] Agent survives 72 hours continuous running without crash
- [ ] Dashboard displays all panels correctly

---

## Phase 5: Backtesting & Tuning

**Goal:** Validate parameter choices on historical data before risking real money.

### Tasks

**5.1 — Historical data collection**
- Download Manifold historical resolved markets via API
- Store as `data/manifold_historical_YYYY-MM.json`
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
- Map Manifold-style question categories to Betfair event types
- Start with a single category (e.g. Politics) to limit scope
- `get_betfair_markets(event_type_id)` → list of Betfair MarketBook objects

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
config.py ──► scanner/manifold.py ──► llm/client.py ──► llm/prompts.py
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
