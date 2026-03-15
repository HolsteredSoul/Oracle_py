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
    ├── scanner/manifold.py        — Fetch live Manifold markets
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
   ```
   OPENROUTER_API_KEY=sk-or-v1-...
   NEWSDATA_API_KEY=...        # optional
   X_BEARER_TOKEN=...          # optional
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
# Start the agent (paper trading mode)
python main.py

# Launch the monitoring dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

The dashboard is available at `http://localhost:8501`.

### Running Tests

```bash
pytest tests/ -v
```

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
| 5 | Backtesting & Tuning — historical replay, parameter sweep | Not started |
| 6 | Live Betfair — OMS, market mapping, safeguards | Not started |

31 unit tests passing across Kelly, Bayesian, edge, risk, state manager, and paper execution modules.

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
