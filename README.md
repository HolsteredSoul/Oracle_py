# Oracle

An autonomous Python prediction-market agent that maximizes long-term geometric bankroll growth by identifying and sizing high-conviction trading edges on prediction markets.

> Paper trading on Betfair Australia for safe validation; targeting live execution on Betfair.

---

## Overview

Oracle is an event-driven trading agent that combines statistical models with LLM-powered web intelligence. The statistical model generates candidate probabilities; Perplexity-enriched web search validates them. Both must agree before any trade is placed.

**Key design decisions:**
- **Handshake gate**: Model and Perplexity must agree on direction before any trade. When they disagree, the "edge" is likely model error. (29-trade analysis: agree = +$183, disagree = -$696)
- **Statistical model as anchor**: Sport-specific models (Poisson for football, logistic for AFL/rugby/basketball) generate fair probabilities from real stats APIs
- **Minimum edge threshold of 8%**: Below this, even "agree" trades are noise (data-driven cutoff)
- **LLM as calibrator, not oracle**: Sentiment deltas and uncertainty penalties fine-tune the model; they don't generate probabilities
- Every formula prices in Betfair's 5% commission; Lay bets use explicit liability translation
- Atomic JSON state writes (`tmp -> replace()`) prevent corruption on crash
- Kill switch: `touch state/kill_switch.txt` to halt new trades while keeping monitoring active

---

## Sports Coverage

| Sport | Betfair ID | Stats API | Model | Status |
|-------|-----------|-----------|-------|--------|
| Soccer | 1 | football-data.org (key required) | Poisson | Working |
| AFL | 61420 | Official AFL API (free) | Logistic | Working |
| Basketball (NBA) | 7522 | nba_api (free) | Logistic | Working |
| Baseball (MLB) | 7511 | MLB statsapi (free) | Logistic | Working |
| Rugby Union | 5 | TheSportsDB (free) | Logistic | Working |
| Rugby League (NRL) | 1477 | TheSportsDB (free) | Logistic | Working |
| Ice Hockey (NHL) | 7524 | NHL API (free) | Logistic | Working |

Non-NBA basketball, non-NHL hockey, and niche football leagues are scanned but skipped when team mapping fails (no stats = no trade).

Women's, youth, reserve, and U-age markets are filtered out automatically.

---

## Architecture

```
main.py (adaptive scan cycle: 10-45 min based on market density)
    |
    +-- scanner/betfair_scanner.py    -- Fetch live Betfair markets
    +-- enrichment/trigger.py         -- Decide: light scan or deep trigger?
    |       +-- enrichment/news.py         -- NewsData.io + Google News RSS
    |       +-- enrichment/stats/          -- Sport-specific stats providers:
    |       |     football.py              -- football-data.org (Poisson inputs)
    |       |     afl.py                   -- Official AFL API (aflapi.afl.com.au)
    |       |     basketball.py            -- nba_api
    |       |     baseball.py              -- MLB statsapi
    |       |     rugby.py                 -- TheSportsDB (rugby union)
    |       |     rugby_league.py          -- TheSportsDB (NRL)
    |       |     hockey.py                -- NHL API (api-web.nhle.com)
    |       +-- enrichment/team_mapping.py -- Betfair name -> stats API name
    |
    +-- llm/client.py                 -- OpenRouter wrapper (cost tracking)
    |       +-- llm/prompts.py             -- JSON-forced prompt templates
    |       +-- llm/models.py             -- Pydantic response validation
    |
    +-- strategy/bayesian.py          -- Logit-space probability updater
    +-- strategy/kelly.py             -- Commission-aware Kelly + liability
    +-- strategy/edge.py              -- Executable edge calculation
    +-- strategy/statistical_model.py -- Sport-specific prediction models
    |
    +-- risk/manager.py               -- Pre-trade gates
    +-- execution/paper.py            -- FOK simulation, fills, settlement
    |
    +-- storage/state_manager.py      -- OracleState JSON persistence
    +-- storage/scan_feed.py          -- Per-market scan outcomes
    +-- storage/rejection_cache.py    -- Skip recently-rejected markets
```

The Streamlit dashboard (`src/dashboard/app.py`) runs separately and reads state files.

---

## Trade Decision Pipeline

```
1. Scanner fetches Betfair markets (filtered by sport, country, prob range)
2. Niche league gate filters out women's, youth, reserve markets
3. Statistical model generates p_model from sport-specific stats API
4. No-model gate: if stats unavailable or completeness < 50%, skip
5. Light LLM scan: news sentiment delta + uncertainty
6. If edge candidate: Perplexity deep scan with grounded web search
7. HANDSHAKE GATE: model and Perplexity must agree on direction
8. Edge must be >= 8% (margin_min_paper = 0.080)
9. Kelly sizing with commission, drawdown throttle, hard cap
10. Paper execution with depth-aware fills and slippage
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (required)
- Betfair account with API access (required)
- NewsData.io API key (optional -- enrichment)
- Football-data.org API key (optional -- football stats)
- Basketball API key (optional -- non-NBA basketball via API-Sports)

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

   **Optional (agent degrades gracefully without these):**
   ```
   NEWSDATA_API_KEY=...
   FOOTBALL_DATA_API_KEY=...
   BASKETBALL_API_KEY=...
   ```

2. Review `config.toml` -- key settings:
   ```toml
   [triggers]
   margin_min_paper = 0.080     # Minimum 8% edge (raised from 2% based on trade data)

   [risk]
   max_exposure = 0.85          # Max fraction of bankroll in open positions
   kelly_base_fraction = 0.75   # Three-quarter Kelly
   kelly_hard_cap = 0.10        # Max 10% of bankroll per trade
   allow_in_play = false        # Block in-play markets

   [scanner]
   betfair_event_types = [1, 5, 61420, 1477, 7522, 7524, 7511]
   betfair_country_codes = ["AU", "GB", "DE", "ES", "IT", "FR", "NL", "PT", "US", "CA"]
   ```

### Running

```bash
# Paper trading -- real Betfair data, no bets placed
python main.py

# Dashboard (separate terminal)
streamlit run src/dashboard/app.py
```

### Running Tests

```bash
pytest tests/ -v
```

238 tests covering Kelly, Bayesian, edge, risk, state manager, paper execution, team mapping, scanner, and statistical model modules.

---

## Core Math

### Statistical Model

Sport-specific models generate `p_model` from stats APIs:
- **Football**: Independent Poisson (goals scored/conceded averages)
- **AFL, Rugby, Basketball, Baseball, Hockey**: Logistic with form differential + home advantage

### Probability Update (Bayesian)

```
p_fair = expit(logit(p_model) + beta * sentiment_delta)
```
- `p_model` -- statistical model probability (prior)
- `beta = 0.50` -- update weight
- LLM adjustment capped at +/-5% from p_model

### Handshake Gate

```
if model says BACK (p_model > market) and Perplexity delta < 0: SKIP
if model says LAY  (p_model < market) and Perplexity delta > 0: SKIP
```

Both must agree. Historical data (29 trades):
- Agree: +$183 (57% win rate)
- Disagree: -$696 (20% win rate)

### Executable Edge

```
Back:  edge = p_fair - p_ask
Lay:   edge = p_bid - p_fair
```

Trade only if `edge >= margin_min` (0.080 = 8%).

### Kelly Sizing

```
f* = raw Kelly fraction (commission-aware)
f  = k * f* * lambda_conf * lambda_dd
       k          = 0.75  (three-quarter Kelly)
       lambda_dd  = 0.50 if drawdown > 25%, else 1.0
f_final = min(f, kelly_hard_cap)
size    = min(f_final * bankroll, liquidity * 0.70)
```

---

## Safety

- **Handshake gate**: Model + Perplexity must agree before any trade
- **Kill switch**: `touch state/kill_switch.txt` -- halts all new trades
- **Daily LLM cap**: Auto-downgrades to fast model at 80% of cap
- **Hard bet cap**: Kelly fraction capped at `kelly_hard_cap`
- **Minimum edge**: 8% threshold -- below this, trades are noise
- **Liquidity gate**: Markets with < $50 available depth rejected
- **Volume gate**: Markets with < $500 matched volume rejected
- **In-play gate**: In-play markets rejected (no in-play engine)
- **Niche league gate**: Women's, youth, reserve markets filtered
- **Slippage model**: Fill prices degrade proportional to order size
- **Depth-aware fills**: Walk real Betfair order book ladder
- **Drawdown throttle**: Kelly halved at 25% drawdown
- **Rejection cache**: Failed markets skipped for 2 hours

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation -- config, scanner, scheduler | Complete |
| 2 | Intelligence -- LLM integration, enrichment | Complete |
| 3 | Strategy & Risk -- Kelly, Bayesian, edge, risk gates | Complete |
| 4 | Paper Trading -- state persistence, execution, dashboard | Complete |
| 5A | Statistical models + handshake gate | Complete |
| 5 | Backtesting & Tuning | Not started |
| 6 | Live Betfair -- OMS, market mapping, safeguards | Not started |

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, adaptive scheduler, trade pipeline |
| `config.toml` | All runtime configuration |
| `src/strategy/statistical_model.py` | Sport-specific prediction models |
| `src/strategy/kelly.py` | Kelly formula + Lay liability |
| `src/strategy/bayesian.py` | Logit-space probability updater |
| `src/enrichment/stats/` | Stats providers (AFL, football, NBA, MLB, NHL, rugby) |
| `src/enrichment/team_mapping.py` | Betfair -> stats API team name resolution |
| `src/execution/paper.py` | Paper broker simulation |
| `src/dashboard/app.py` | Streamlit monitoring UI |

---

## License

Private repository -- all rights reserved.
