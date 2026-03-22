# Oracle

An autonomous Python prediction-market agent that maximizes long-term geometric bankroll growth by identifying and sizing high-conviction trading edges on prediction markets.

> Paper trading on Betfair Australia for safe validation; targeting live execution on Betfair.

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
    ├── scanner/betfair_scanner.py — Fetch live Betfair AU markets
    ├── enrichment/trigger.py      — Decide: light scan or deep trigger?
    │       ├── enrichment/news.py       — NewsData.io + Google News RSS
    │       ├── enrichment/x_sentiment.py — X API v2 tweet search
    │       └── enrichment/stats.py      — Football-data.org + Squiggle stats
    │
    ├── llm/client.py              — OpenRouter wrapper (cost tracking, tier routing)
    │       ├── llm/prompts.py           — JSON-forced prompt templates
    │       └── llm/models.py            — Pydantic response validation
    │
    ├── strategy/bayesian.py       — Logit-space probability updater
    ├── strategy/kelly.py          — Commission-aware Kelly + liability translation
    ├── strategy/edge.py           — Executable edge calculation
    ├── strategy/statistical_model.py — Poisson match prediction model
    │
    ├── risk/manager.py            — Pre-trade gates (exposure cap, confidence floor, drawdown)
    ├── execution/paper.py         — FOK simulation, partial fills, settlement, cancellation
    └── storage/state_manager.py   — OracleState JSON persistence
```

The Streamlit dashboard (`src/dashboard/app.py`) runs as a separate process and reads from the same state file.

---

## Getting Started

### Prerequisites

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (required)
- Betfair account with API access (required)
- NewsData.io API key (optional — enrichment)
- Football-data.org API key (optional — statistical model)
- X API v2 Bearer token (optional — enrichment)

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

   **Required:**
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   BETFAIR_USERNAME=...
   BETFAIR_PASSWORD=...
   BETFAIR_APP_KEY=...
   ```

   **Optional enrichment (agent degrades gracefully without these):**
   ```
   NEWSDATA_API_KEY=...
   FOOTBALL_DATA_API_KEY=...
   X_BEARER_TOKEN=...
   BETFAIR_CERTS_PATH=...        # Phase 6 only: directory with client certs
   ```

2. Review `config.toml` — key settings to check before running:
   ```toml
   [llm]
   daily_cap_usd = 5.0          # Daily LLM spend limit

   [triggers]
   margin_min_paper = 0.030     # Minimum edge threshold for paper trades

   [risk]
   max_exposure = 0.85          # Max fraction of bankroll in open positions
   kelly_base_fraction = 0.50   # Half-Kelly (conservative default)
   ```

### Running

```bash
# Paper trading — real Betfair AU market data, no bets placed
python main.py

# Launch the monitoring dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

The agent uses live Betfair exchange prices and volumes for the full pipeline (probability, spread, liquidity) while keeping `PaperBroker` for all execution and settlement. No bets are ever placed.

The dashboard is available at `http://localhost:8501`.

### Running Tests

```bash
pytest tests/ -v
```

---

## Paper Trading

Oracle uses Betfair AU exchange data as the market feed. No real money is involved — `PaperBroker` simulates realistic exchange behaviour so the same code path runs unmodified when switching to live Betfair execution in Phase 6.

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

Positions are checked for resolution at the top of every 30-minute scan cycle. When a market resolves, P&L is calculated and the position is closed:

| Direction | Resolution | P&L |
|-----------|------------|-----|
| Back | YES | `stake × (1/entry_price − 1) × (1 − commission)` |
| Back | NO | `−stake` |
| Lay | NO | `stake × (1 − commission)` |
| Lay | YES | `−liability` |
| Back or Lay | MKT | partial win using `resolution_probability` as settlement price |

Positions in markets that age beyond `betfair_max_market_age_hours` (168h) are auto-cancelled with escrow refund.

Commission (`commission_pct = 0.05`) is applied to gross winnings only — never to losses.

### State File

All bankroll, positions, and trade history are persisted to `state/oracle_state.json` after every trade and settlement. Writes are atomic (`tmp → replace()`). The file is git-ignored.

To reset the paper bankroll, delete `state/oracle_state.json` — it will be recreated on next run.

---

## Core Math

### Probability Update (Bayesian)

```
p_fair = expit(logit(p_mid) + β · sentiment_delta)
```
- `p_mid` — market mid-price (prior)
- `β = 2.0` — update weight
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
f_final = min(f, kelly_hard_cap)
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
| 5A | Intelligence Upgrade — statistical model, team mapping, Perplexity enrichment | In progress |
| 5 | Backtesting & Tuning — historical replay, parameter sweep | Not started |
| 6 | Live Betfair — OMS, market mapping, safeguards | Not started |

149 unit tests passing across Kelly, Bayesian, edge, risk, state manager, paper execution, team mapping, and Betfair scanner modules.

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
| `src/enrichment/team_mapping.py` | Betfair → stats API team name resolution (dynamic index) |
| `src/strategy/statistical_model.py` | Poisson match outcome prediction model |

---

## Safety

- **Kill switch**: `touch state/kill_switch.txt` — agent skips all new trades on next cycle
- **Daily LLM cap**: Auto-downgrades to fast model at 80% of cap; returns `None` if cap exceeded
- **Hard bet cap**: Kelly fraction capped at `kelly_hard_cap` (configurable)
- **Liquidity cap**: Never uses more than 70% of available market liquidity
- **Drawdown throttle**: Kelly halved automatically when drawdown exceeds 20%
- **Auto-cancel**: Positions in aged-out markets are auto-cancelled with escrow refund

---

## License

Private repository — all rights reserved.
