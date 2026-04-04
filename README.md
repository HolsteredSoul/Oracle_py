# Oracle

An autonomous Python prediction-market agent that maximizes long-term geometric bankroll growth by identifying and sizing high-conviction trading edges on prediction markets.

> Paper trading on Betfair Australia for safe validation; targeting live execution on Betfair.

---

## Overview

Oracle is an event-driven trading agent that combines statistical models with LLM-powered web intelligence. The statistical model generates candidate probabilities; Perplexity-enriched web search validates them. Both must agree before any trade is placed.

**Key design decisions:**
- **Handshake gate**: Model and Perplexity must agree on direction before any trade. When they disagree, the "edge" is likely model error. (29-trade analysis: agree = +$183, disagree = -$696)
- **Statistical model as anchor**: Sport-specific models (Poisson for football, logistic for AFL/rugby/baseball/hockey) generate fair probabilities from real stats APIs
- **Minimum edge threshold of 8%**: Below this, even "agree" trades are noise (data-driven cutoff)
- **3-hour time gate**: Markets must be within 3h of kickoff before trading. Earlier entry = negative CLV from stale prices.
- **DNB conviction gate**: Draw No Bet markets require the raw match-odds probability > 55% to prevent renormalization from inflating coin-flip predictions into large-looking edges
- **LLM as calibrator, not oracle**: Sentiment deltas and uncertainty penalties fine-tune the model; they don't generate probabilities
- Every formula prices in Betfair's 5% commission; Lay bets use explicit liability translation
- Atomic JSON state writes (`tmp -> replace()`) prevent corruption on crash
- Kill switch: `touch state/kill_switch.txt` to halt new trades while keeping monitoring active

---

## Sports Coverage

| Sport | Betfair ID | Stats API | Model | Status |
|-------|-----------|-----------|-------|--------|
| Soccer | 1 | football-data.org (key required) | Poisson | Active |
| AFL | 61420 | Official AFL API (free) | Logistic (form + standings) | Active |
| Baseball (MLB) | 7511 | MLB statsapi (free) | Logistic | Active |
| Rugby Union | 5 | ESPN hidden API (free) | Logistic | Active |
| Rugby League (NRL) | 1477 | ESPN hidden API (free) | Logistic | Active |
| Ice Hockey (NHL) | 7524 | NHL API (free) | Logistic | Active |
| Basketball (NBA) | 7522 | -- | -- | **Disabled** (injury-blind model = negative CLV) |

**Football competitions** (8 leagues): Premier League, Championship, Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, Champions League.

**Rugby**: NRL + Super Rugby, Premiership, Top 14 via ESPN. Super League falls back to TheSportsDB (partial data, 75% completeness).

**Competition whitelist**: Only leagues with verified stats coverage are fetched from Betfair. Per-sport API queries prevent junk leagues (Serie C, USL, NPB, etc.) from crowding out viable markets in the 200-slot catalogue cap.

Women's, youth, reserve, and U-age markets are filtered out automatically.

---

## Architecture

```
main.py (adaptive scan cycle: 10-45 min based on market density)
    |
    +-- scanner/betfair_scanner.py    -- Per-sport Betfair queries with competition whitelists
    +-- enrichment/trigger.py         -- Decide: light scan or deep trigger?
    |       +-- enrichment/news.py         -- NewsData.io + Google News RSS
    |       +-- enrichment/stats/          -- Sport-specific stats providers:
    |       |     football.py              -- football-data.org (Poisson inputs)
    |       |     afl.py                   -- Official AFL API (aflapi.afl.com.au)
    |       |     baseball.py              -- MLB statsapi
    |       |     rugby.py                 -- ESPN hidden API (rugby union)
    |       |     rugby_league.py          -- ESPN (NRL) + TheSportsDB fallback (Super League)
    |       |     hockey.py                -- NHL API (api-web.nhle.com)
    |       |     espn.py                  -- Shared ESPN client (standings + form)
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
 1. Per-sport Betfair queries with competition whitelist (only viable leagues)
 2. Niche league gate filters out women's, youth, reserve markets
 3. Statistical model generates p_model from sport-specific stats API
 4. No-model gate: if stats unavailable or completeness < 75%, skip
 5. DNB conviction gate: raw match-odds prob must be > 55% for Draw No Bet
 6. Light LLM scan: news sentiment delta + uncertainty
 7. If edge candidate: Perplexity deep scan with grounded web search
 8. Extreme divergence gate: skip if model vs market gap is implausible
 9. HANDSHAKE GATE: model and Perplexity must agree on direction
10. Edge must be >= 8% (margin_min_paper = 0.080)
11. Liquidity, volume, crossed-book, and in-play gates
12. Time gate: market must be within 3h of start (mature prices only)
13. Lay win probability gate: lay bets require >= 35% chance of winning
14. Kelly sizing with commission, drawdown throttle, hard cap
15. Paper execution with depth-aware fills and slippage
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (required)
- Betfair account with API access (required)
- NewsData.io API key (optional -- enrichment)
- Football-data.org API key (optional -- football stats)

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
   min_lay_win_probability = 0.35  # Lay bets must have >= 35% win chance
   dnb_min_match_odds_prob = 0.55  # DNB requires raw match-odds prob >= 55%

   [scanner]
   betfair_event_types = [1, 5, 61420, 1477, 7524, 7511]
   betfair_hours_ahead = 12     # Scan 12h ahead to catch evening slate
   betfair_min_hours_before_start = 3  # Only trade within 3h of kickoff
   # Per-sport competition whitelists (Betfair competition IDs)
   competition_ids_football = [10932509, 7129730, 55, 59, 81, 117, 88, 228]
   competition_ids_baseball = [11196870]   # MLB only
   # ... etc for each sport
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
- **Football**: Independent Poisson (goals scored/conceded averages from football-data.org)
- **AFL**: Logistic with form points differential + ladder standings (official AFL API)
- **Rugby Union/League**: Logistic with net points differential + standings (ESPN API)
- **Baseball, Hockey**: Logistic with net run/goal differential + standings
- All logistic models use coefficient 0.80 and clamp to [0.20, 0.80] -- conservative given small sample sizes from free-tier stats APIs

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

### DNB Conviction Gate

Draw No Bet markets renormalize match-odds probabilities by stripping the draw:
```
p_DNB = p_home / (p_home + p_away)
```
This inflates small match-odds edges into large-looking DNB edges. A 53% match-odds prediction becomes 75% in DNB with a typical 30% draw. The gate requires the raw match-odds probability >= 55% before allowing any DNB trade.

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
- **DNB conviction gate**: Raw match-odds probability must be >= 55% for Draw No Bet trades
- **Time gate**: Markets must be within 3h of kickoff (prevents trading into stale/thin markets)
- **Kill switch**: `touch state/kill_switch.txt` -- halts all new trades
- **Daily LLM cap**: Auto-downgrades to fast model at 80% of cap
- **Hard bet cap**: Kelly fraction capped at `kelly_hard_cap`
- **Minimum edge**: 8% threshold -- below this, trades are noise
- **Extreme divergence gate**: Skip if model vs market gap > 30pp (likely model error)
- **Liquidity gate**: Markets with < $50 available depth rejected
- **Volume gate**: Markets with < $500 matched volume rejected
- **In-play gate**: In-play markets rejected (no in-play engine)
- **Crossed-book gate**: Skip if best_back < best_lay (stale/suspended data)
- **Lay win probability gate**: Lay bets require >= 35% win probability
- **Competition whitelist**: Per-sport Betfair queries with `competition_ids` -- only leagues with stats coverage
- **Niche league gate**: Women's, youth, reserve markets filtered
- **Model clamp**: Logistic models clamped to [0.20, 0.80] -- no overconfident predictions from small samples
- **Slippage model**: Fill prices degrade proportional to order size
- **Depth-aware fills**: Walk real Betfair order book ladder
- **Drawdown throttle**: Kelly halved at 25% drawdown
- **Rejection cache**: Failed markets skipped for 2 hours

---

## APIs Used

| API | Sport | Auth | Notes |
|-----|-------|------|-------|
| football-data.org | Soccer | API key (free tier) | 10 req/min. PL, Championship, Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, UCL. BL2 returns 403 despite documentation. |
| Official AFL API | AFL | None | aflapi.afl.com.au. Full season results + ladder. |
| MLB Stats API | Baseball | None | statsapi.mlb.com. Full season. |
| NHL API | Hockey | None | api-web.nhle.com. Full season. |
| ESPN hidden API | Rugby Union + NRL | None | site.api.espn.com. Standings + 60-day form. No Super League data. |
| TheSportsDB | Super League fallback | None | Free tier: 1 last event, no standings. 75% completeness max. |
| Betfair Exchange API | All | App key + certs | Market data, order book, settlement. |
| OpenRouter | LLM | API key | Perplexity + GPT-4o for enrichment. |
| NewsData.io | News | API key | Headlines for sentiment enrichment. |

**Dead APIs (tested, don't use):** Squiggle (522), API-Sports rugby (empty data), TheSportsDB standings for non-soccer (empty).

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation -- config, scanner, scheduler | Complete |
| 2 | Intelligence -- LLM integration, enrichment | Complete |
| 3 | Strategy & Risk -- Kelly, Bayesian, edge, risk gates | Complete |
| 4 | Paper Trading -- state persistence, execution, dashboard | Complete |
| 5A | Statistical models + handshake gate | Complete |
| 5B | Model calibration + safety gates | Complete |
| 5 | Backtesting & Tuning | Not started |
| 6 | Live Betfair -- OMS, market mapping, safeguards | Not started |

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, adaptive scheduler, trade pipeline, all gates |
| `config.toml` | All runtime configuration |
| `src/strategy/statistical_model.py` | Sport-specific prediction models |
| `src/strategy/kelly.py` | Kelly formula + Lay liability |
| `src/strategy/bayesian.py` | Logit-space probability updater |
| `src/enrichment/stats/` | Stats providers (AFL, football, MLB, NHL, ESPN rugby) |
| `src/enrichment/stats/espn.py` | Shared ESPN client for rugby union + NRL |
| `src/enrichment/team_mapping.py` | Betfair -> stats API team name resolution |
| `src/execution/paper.py` | Paper broker simulation |
| `src/dashboard/app.py` | Streamlit monitoring UI |
| `src/storage/rejection_cache.py` | Skip recently-rejected markets (2h TTL) |

---

## License

Private repository -- all rights reserved.
