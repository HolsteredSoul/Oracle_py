# Oracle

An autonomous Python prediction-market agent that maximizes long-term geometric bankroll growth by identifying and sizing high-conviction trading edges on prediction markets.

> Paper trading on Manifold Markets for safe validation; targeting live execution on Betfair Australia.

---

## Overview

Oracle is an event-driven trading agent built around a core principle: **LLM as calibrator, not oracle**. The model outputs sentiment deltas and uncertainty penalties; Bayesian math computes fair probabilities. All sizing is commission-aware Kelly with hard caps and liquidity constraints.

**Key design decisions:**
- News, X/Twitter sentiment, and volatility spikes trigger expensive LLM calls — not a fixed schedule
- Fair probability = `expit(logit(p_mid) + β · sentiment_delta)` (logit-space, numerically stable)
- Every formula prices in Betfair's 5% commission; Lay bets use explicit liability translation
- Atomic JSON state writes (`tmp → replace()`) prevent corruption on crash
- Kill switch: `touch state/kill_switch.txt` to halt new trades while keeping monitoring active

---

## Architecture

```
main.py (30-min scan cycle)
    │
    ├── scanner/manifold.py        — Fetch live Manifold markets (default)
    ├── scanner/betfair_scanner.py — Fetch live Betfair AU markets (--betfair-paper)
    ├── enrichment/trigger.py      — Decide: light scan or deep trigger?
    │       ├── enrichment/news.py       — NewsData.io headlines
    │       └── enrichment/x_sentiment.py — X API v2 tweet search
    │
    ├── llm/client.py              — OpenRouter wrapper (cost tracking, tier routing)
    │       ├── llm/prompts.py           — Three JSON-forced prompt templates
    │       └── llm/models.py            — Pydantic response validation
    │
    ├── strategy/bayesian.py       — Logit-space probability updater
    ├── strategy/kelly.py          — Commission-aware Kelly + liability translation
    ├── strategy/edge.py           — Executable edge calculation
    │
    ├── risk/manager.py            — Pre-trade gates (exposure cap, confidence floor, drawdown)
    ├── execution/paper.py         — FOK simulation, partial fills, settlement
    └── storage/state_manager.py   — OracleState JSON persistence
```

The Streamlit dashboard (`src/dashboard/app.py`) runs as a separate process and reads from the same state file.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (required)
- NewsData.io API key (optional — enrichment only)
- X API v2 Bearer token (optional — enrichment only)

### Installation

```bash
git clone https://github.com/HolsteredSoul/Oracle_py.git
cd Oracle_py
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:

   **Paper trading on Manifold (minimum required):**
   ```
   OPENROUTER_API_KEY=sk-or-v1-...   # required — all LLM calls route through OpenRouter
   MANIFOLD_API_KEY=...               # optional — reserved for future authenticated Manifold calls
   ```

   **Enrichment (optional — agent degrades gracefully without these):**
   ```
   NEWSDATA_API_KEY=...               # NewsData.io — enables news headline triggers
   X_BEARER_TOKEN=...                 # X API v2 — enables tweet momentum triggers
   ```

   **Betfair paper trading (`--betfair-paper`) and live trading (Phase 6):**
   ```
   BETFAIR_USERNAME=...               # Betfair account email
   BETFAIR_PASSWORD=...               # Betfair account password
   BETFAIR_APP_KEY=...                # App key from Betfair Developer portal (1.0-DELAY for paper)
   BETFAIR_CERTS_PATH=...             # Phase 6 only: directory with client-2048.crt and .key
   ```

2. Review `config.toml` — key settings to check before running:
   ```toml
   [llm]
   daily_cap_usd = 5.0          # Daily LLM spend limit

   [triggers]
   margin_min_paper = 0.025     # Minimum edge threshold for paper trades

   [risk]
   max_exposure = 0.85          # Max fraction of bankroll in open positions
   kelly_base_fraction = 0.50   # Half-Kelly (conservative default)

   [scanner]
   manifold_min_volume = 500    # Skip illiquid markets below this volume
   ```

### Running

```bash
# Paper trading — Manifold Markets data (default)
python main.py

# Paper trading — real Betfair AU market data, no bets placed
python main.py --betfair-paper

# Launch the monitoring dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

`--betfair-paper` uses live Betfair exchange prices and volumes for the full pipeline (probability, spread, liquidity) while keeping `PaperBroker` for all execution and settlement. No bets are ever placed.

The dashboard is available at `http://localhost:8501`.

### Running Tests

```bash
pytest tests/ -v
```

---

## Paper Trading

Oracle uses Manifold Markets as a paper trading venue. No real money is involved — Manifold uses play-money (mana). The paper broker simulates realistic exchange behaviour so the same code path runs unmodified when switching to live Betfair later.

### Fill Simulation

Every order is Fill-or-Kill:

```
requested_size = f_final × bankroll

if requested_size ≤ available_liquidity × 0.70:
    fill at 100%
else:
    fill at random.uniform(60%, 90%)   # partial fill
```

The 70% liquidity safety factor ensures Oracle never moves the market against itself.

### Settlement

Positions are checked for resolution at the top of every 30-minute scan cycle via `check_and_settle_positions()`. When a market resolves, P&L is calculated and the position is closed:

| Direction | Resolution | P&L |
|-----------|------------|-----|
| Back | YES | `stake × (1/entry_price − 1) × (1 − commission)` |
| Back | NO | `−stake` |
| Lay | NO | `stake × (1 − commission)` |
| Lay | YES | `−liability` |
| Back or Lay | MKT | partial win using `resolution_probability` as settlement price |

Commission (`commission_pct = 0.05`) is applied to gross winnings only — never to losses.

### State File

All bankroll, positions, and trade history are persisted to `state/oracle_state.json` after every trade and settlement. Writes are atomic (`tmp → replace()`). The file is git-ignored.

```json
{
  "bankroll": 1000.0,
  "peak_bankroll": 1043.20,
  "positions": { "<market_id>": { ... } },
  "trade_history": [ { ... } ],
  "priors": { "<market_id>": 0.62 }
}
```

To reset the paper bankroll, delete `state/oracle_state.json` — it will be recreated on next run.

---

## Core Math

### Probability Update (Bayesian)

```
p_fair = expit(logit(p_mid) + β · sentiment_delta)
```
- `p_mid` — market mid-price (prior)
- `β = 0.15` — conservative update weight, tuned via backtest
- Results are always in `(0, 1)` by construction

### Executable Edge

```
Back bet:  edge = p_fair - p_ask - commission
Lay bet:   edge = p_bid - p_fair - commission
```

Trades execute only if `edge ≥ margin_min`.

### Kelly Sizing

```
f* = raw Kelly fraction
f  = k · f* · λ_conf · λ_dd
       k        = 0.50  (half-Kelly base)
       λ_conf   = clamp(confidence / 100, 0.5, 1.0)
       λ_dd     = 0.50 if drawdown > 20%, else 1.0
f_final = min(f, 0.25)                   # hard cap: never >25% of bankroll
size    = min(f_final · bankroll, liquidity × 0.70)
```

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation — config, scanner, scheduler | Complete |
| 2 | Intelligence — LLM integration, enrichment | Complete |
| 3 | Strategy & Risk — Kelly, Bayesian, edge, risk gates | Complete |
| 4 | Paper Trading — state persistence, execution, dashboard | Complete |
| 5A | Intelligence Upgrade — statistical model, LLM reframing | In progress |
| 5 | Backtesting & Tuning — historical replay, parameter sweep | Not started |
| 6 | Live Betfair — OMS, market mapping, safeguards | Not started |

102 unit tests passing across Kelly, Bayesian, edge, risk, state manager, paper execution, and Betfair scanner modules.

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, scheduler, main loop |
| `config.toml` | All runtime configuration |
| `WHITEPAPER.MD` | Full technical specification and math derivations |
| `DEVELOPMENT_PLAN.MD` | Six-phase roadmap with exit gates |
| `src/strategy/kelly.py` | Kelly formula + Lay liability translation |
| `src/strategy/bayesian.py` | Logit-space probability updater |
| `src/execution/paper.py` | Paper broker simulation |
| `src/dashboard/app.py` | Streamlit monitoring UI |
| `src/enrichment/stats.py` | Statistical data fetcher (football-data.org, Squiggle) |
| `src/enrichment/team_mapping.py` | Betfair → stats API team name resolution |
| `src/strategy/statistical_model.py` | Poisson match outcome prediction model |

---

## Dependencies

Core: `httpx`, `tenacity`, `apscheduler`, `pydantic`, `pydantic-settings`, `scipy`, `pandas`
Dashboard: `streamlit`, `plotly`
Testing: `pytest`, `pytest-asyncio`, `hypothesis`

See `requirements.txt` for pinned versions.

---

## Safety

- **Kill switch**: `touch state/kill_switch.txt` — agent skips all new trades on next cycle
- **Daily LLM cap**: Auto-downgrades to fast model at 80% of cap; returns `None` if cap exceeded
- **Hard bet cap**: Kelly fraction capped at 0.25 — never more than 25% of bankroll on a single trade
- **Liquidity cap**: Never uses more than 70% of available market liquidity
- **Drawdown throttle**: Kelly halved automatically when drawdown exceeds 20%

---

## License

Private repository — all rights reserved.
