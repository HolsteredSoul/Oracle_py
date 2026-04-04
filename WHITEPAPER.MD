# Oracle Python Agent -- Technical Whitepaper

**v2.0 -- Calibrated Model, Multi-Gate Risk Architecture**
**Autonomous Profit-Maximizing Prediction-Market Agent**
**April 2026**
**Author:** Pierre
**Scope:** Betfair Australia paper trading (safe tuning & validation) -> Betfair Australia live (real-money execution)

---

## Executive Summary

Oracle is a Python-native autonomous agent built to **maximize long-term geometric bankroll growth** while aggressively sizing high-conviction edges and surviving Betfair's harsh realities: wide bid-ask spreads, HFT latency advantage, partial fills, market suspensions, and thin liquidity.

It combines sport-specific statistical models with LLM-powered web intelligence through a **multi-gate risk architecture** that requires model-LLM agreement before any capital is deployed. The system has evolved through iterative paper trading, with each losing trade contributing a new safety gate.

**Core design principles:**
- **Statistical model as anchor** -- sport-specific models (Poisson for football, logistic for AFL/rugby/baseball/hockey) generate fair probabilities from free stats APIs
- **Handshake gate** -- model and Perplexity must agree on direction before any trade (29-trade analysis: agree = +$183, disagree = -$696)
- **LLM as calibrator, not oracle** -- outputs `sentiment_delta` and `uncertainty_penalty`; the Bayesian updater does the math
- **Commission-aware Kelly** with explicit liability translation for Lay bets
- **18 pre-trade safety gates** -- each added in response to a specific failure mode
- **Competition whitelist** -- per-sport Betfair queries prevent junk leagues from consuming scan capacity

**Paper trading performance (as of April 2026):**

| Metric | Actual |
|---|---|
| Trades placed | ~30 |
| Active sports | 6 (soccer, AFL, MLB, rugby union, NRL, NHL) |
| Avg CLV | Under review (3h time gate added to improve) |
| Edge threshold | 8% minimum |
| Daily LLM cost | ~$0.87 |

---

## 1. Vision & Measurable Live Targets

**Objective:** Maximize geometric growth rate:

$$G = \mathbb{E}\bigl[\log(W_{t+1}/W_t)\bigr]$$

**Live Betfair targets** (validated via paper trading before small-stakes go-live):

| Metric | Minimum Bar |
|---|---|
| Executable post-commission edge | >= 8% |
| Weekly opportunities | 3-10 |
| Max drawdown | < 40% |
| Fill rate on intended size | >= 70% |
| Commission drag modeled | 5% fully priced in |
| CLV | Positive over 100+ trades |

---

## 2. Core Mathematics

### 2.1 Executable Edge

Let:
- `p_ask` = best Back price (cost to buy YES)
- `p_bid` = best Lay price (cost to sell YES)
- `p_fair` = Oracle's Bayesian probability estimate

**Back YES edge:**
$$\text{edge}_\text{back} = p_\text{fair} - p_\text{ask}$$

**Lay YES edge:**
$$\text{edge}_\text{lay} = p_\text{bid} - p_\text{fair}$$

### 2.2 Trigger Conditions

Execute Back only if:
$$p_\text{fair} > p_\text{ask} + \delta_\text{min}$$

Execute Lay only if:
$$p_\text{fair} < p_\text{bid} - \delta_\text{min}$$

Where `delta_min = 0.080` (8% edge threshold, raised from 2% based on 29-trade analysis showing sub-8% trades are noise).

### 2.3 Commission-Aware Kelly

Let `c` = commission rate (0.05 for 5%).

**Back bet** -- result is the **stake** as a fraction of bankroll:

$$f_\text{back} = \frac{p_\text{fair} - p_\text{ask} - c \cdot p_\text{fair} \cdot (1 - p_\text{ask})}{(1 - p_\text{ask})(1 - c)}$$

**Lay bet** -- result is the **liability** as a fraction of bankroll:

$$f_\text{lay} = \frac{p_\text{bid} - p_\text{fair} - c \cdot p_\text{bid} \cdot (1 - p_\text{fair})}{p_\text{bid} \cdot (1 - c)}$$

> `f_lay` outputs a **liability fraction**, not a stake fraction. The 2.4 translation step is mandatory before placing a Lay order.

**Oracle sizing** (confidence-scaled fractional Kelly with drawdown throttle):

$$f = k \cdot f^{*}_\text{comm} \cdot c_\text{conf} \cdot \lambda_\text{dd}$$

Where:
- `k` = base Kelly fraction (0.75 three-quarter Kelly)
- `c_conf` = LLM confidence scaler = `conf_score / 100`, clamped to [0.5, 1.0]
- `lambda_dd` = drawdown throttle: 1.0 if drawdown < 25%, else 0.5
- Hard caps: per-bet <= 10% bankroll (`kelly_hard_cap`), total open exposure <= 85%

### 2.4 Liability vs Stake Translation

- **Back bets:** `Stake = f_back x Bankroll`
- **Lay bets:** `Liability = f_lay x Bankroll` -> `Stake = Liability / (decimal_odds - 1)`

### 2.5 Liquidity-Capped Sizing

$$\text{size}_\text{final} = \min\bigl(f \times \text{Bankroll},\ L_{\text{avail}} \times 0.70\bigr)$$

Where $L_{\text{avail}}$ = available liquidity at the target price tier. The 0.70 safety factor prevents Oracle from consuming > 70% of available liquidity.

---

## 3. Intelligence Layer

### 3.1 Statistical Model (Primary Signal)

Sport-specific models generate calibrated `p_model` from free stats APIs. This replaces the raw market mid-price as the Bayesian prior.

| Sport | Model | Data Source | Inputs |
|---|---|---|---|
| Soccer | Independent Poisson | football-data.org (API key, free tier) | Goals scored/conceded averages -> lambda -> goal matrix |
| AFL | Logistic | Official AFL API (free, no auth) | Form points differential + ladder standings |
| Baseball (MLB) | Logistic | MLB Stats API (free, no auth) | Net run differential + standings |
| Rugby Union | Logistic | ESPN hidden API (free, no auth) | Net points differential + standings |
| Rugby League (NRL) | Logistic | ESPN hidden API (free, no auth) | Net points differential + standings |
| Ice Hockey (NHL) | Logistic | NHL API (free, no auth) | Net goal differential + standings |

All logistic models use:
- **Scoring coefficient**: 0.80 (reduced from 1.5 -- free-tier APIs return 1-5 game samples; at 1.5 a single blowout produces 93%+ probabilities)
- **Standings coefficient**: 0.50
- **Probability clamp**: [0.20, 0.80] (no head-to-head match is genuinely 95% on Betfair)

**Football (Poisson):** Computes `P(home scores X) x P(away scores Y)` across a 0-7 goal matrix, then aggregates into home/draw/away probabilities. Inherently more stable than logistic models due to averaging over combinatorial outcomes.

**Team name resolution** (`team_mapping.py`): Three-tier matching -- exact index lookup -> fuzzy SequenceMatcher (threshold 0.85, min 6 chars) -> Perplexity Sonar fallback. Persistent disk cache. Abbreviation expansion for rugby names (qld -> queensland, etc.).

### 3.2 LLM Qualitative Overlay

The LLM acts as a **calibrator, not an oracle**. It receives the statistical model's `p_model` and news/sentiment context, then returns a `sentiment_delta` adjustment for factors statistics cannot capture: injuries, weather, travel fatigue, morale, managerial changes.

**LLM routing via OpenRouter:**

| Tier | Use Case |
|---|---|
| Fast (80% of calls) | Light batch scans every 30 min |
| Deep (triggered only) | Perplexity Sonar web search for edge candidates |

**LLM output contract:**

```json
{
  "sentiment_delta": 0.18,
  "uncertainty_penalty": 0.12,
  "rationale": "One-sentence explanation"
}
```

- `sentiment_delta` in [-1.0, +1.0]: directional signal strength
- `uncertainty_penalty` in [0.0, 1.0]: reduces effective confidence
- LLM adjustment capped at +/-5% from p_model

### 3.3 Bayesian Updater

Starting from the statistical model probability as prior:

$$p_\text{fair} = \sigma\bigl(\text{logit}(p_\text{model}) + \beta \cdot \Delta_{\text{sent}}\bigr)$$

Where `sigma` is the sigmoid function, `beta = 0.50` (sentiment weight), and `Delta_sent` = `sentiment_delta`.

### 3.4 Handshake Gate

```
if model says BACK (p_model > market) and Perplexity delta < 0: SKIP
if model says LAY  (p_model < market) and Perplexity delta > 0: SKIP
```

Both must agree on direction. Historical data (29 trades):
- Agree: +$183 (57% win rate)
- Disagree: -$696 (20% win rate)

This gate alone would have converted -$514 total P&L into +$183.

### 3.5 DNB Conviction Gate

Draw No Bet markets renormalize match-odds probabilities by stripping the draw:

$$p_\text{DNB} = \frac{p_\text{home}}{p_\text{home} + p_\text{away}}$$

This inflates small match-odds edges into large-looking DNB edges. Example: a 53% match-odds prediction becomes 75% in DNB with a typical 30% draw. A 13pp match-odds disagreement becomes a 20pp DNB "edge."

**Gate:** Raw match-odds probability must be >= 55% (`dnb_min_match_odds_prob`) before allowing any DNB trade. This ensures the model genuinely favours the team before DNB math inflates the edge.

**Origin:** Charlton v Bristol City DNB -- model had Charlton at ~33% match odds (predicted loss), but DNB renormalized to 75.4%. System bet $96, lost. Gate added to prevent recurrence.

### 3.6 CLV Tracking (Diagnostic)

Closing Line Value is tracked on every trade. At settlement, the last observed market price before suspension becomes `closing_price`.

- Back: `CLV = closing_price - entry_price`
- Lay: `CLV = entry_price - closing_price`

Positive CLV over 100+ trades indicates genuine edge. Negative CLV means the market was smarter than the model regardless of short-term P&L.

---

## 4. Trade Decision Pipeline

```
 1. Per-sport Betfair queries with competition whitelist
      Each sport gets its own 200-slot catalogue query with competition_ids.
      Sports can't crowd each other out.

 2. Niche league gate
      Women's, youth, reserve, U-age markets filtered by regex.

 3. Statistical model generates p_model
      Sport-specific model from stats API data.
      Data completeness must be >= 75% (6/8 stat fields).

 4. No-model gate
      If stats unavailable or completeness < 75%, skip market entirely.
      LLM-only probability estimates have no edge over the market.

 5. DNB conviction gate
      For DRAW_NO_BET markets: raw match-odds probability >= 55%.
      Prevents renormalization inflation of coin-flip predictions.

 6. Light LLM scan
      Cheap model produces sentiment_delta + uncertainty_penalty.

 7. Bayesian update
      p_fair = expit(logit(p_model) + beta * sentiment_delta)
      LLM adjustment capped at +/-5% from p_model.

 8. Edge candidate check
      If net_edge < margin_min (8%), skip.

 9. Deep Perplexity scan
      Grounded web search for edge candidates only.
      Fresh internet context before execution.

10. Extreme divergence gate
      If |p_fair - market_mid| > 30pp, skip.
      Likely model error, not a real edge.

11. HANDSHAKE GATE
      Model and Perplexity must agree on direction.
      Any disagreement blocks the trade. No threshold.

12. Realism gates
      Volume >= $500 matched. Liquidity >= $50 available.
      Not in-play. Not crossed book. Edge >= 8%.

13. Time gate
      Market must be within 3h of start time.
      Earlier entry = negative CLV from stale/thin prices.

14. Lay win probability gate
      Lay bets require >= 35% chance of winning (1 - p_fair >= 0.35).
      Prevents laying heavy favourites where payout requires a rare upset.

15. Kelly sizing
      Commission-aware Kelly with confidence scaling and drawdown throttle.
      Hard cap at 10% of bankroll per trade.

16. Risk gates
      Total exposure <= 85%. Max 3 new positions per cycle.

17. Paper execution
      Depth-aware VWAP fills walking the real Betfair order book.
      Linear slippage model. Edge re-checked post-slippage.

18. State persistence + scan feed logging
      Atomic JSON writes. Per-market outcome logged for dashboard.
```

---

## 5. Sports Coverage

### 5.1 Active Sports

**Soccer** (8 competitions):
Premier League, Championship, Bundesliga, Serie A, La Liga, Ligue 1, Eredivisie, Champions League.
Stats: football-data.org (API key, free tier, 10 req/min).

**AFL:**
Official AFL API (aflapi.afl.com.au, free, no auth). Full season match results, ladder computed from results. Model uses form points differential + standings position.

**Baseball (MLB):**
MLB Stats API (statsapi.mlb.com, free, no auth). Full season.

**Rugby Union** (4 competitions):
Super Rugby, Premiership, Top 14, URC via ESPN hidden API (site.api.espn.com, free, no auth). Standings + 60-day form (last 5 matches).

**Rugby League:**
NRL via ESPN. Super League falls back to TheSportsDB (free tier, 1 last event, no standings, 75% completeness max).

**Ice Hockey (NHL):**
NHL API (api-web.nhle.com, free, no auth). Full season.

### 5.2 Disabled Sports

**Basketball (NBA):** Disabled 2026-04-02. Injury-blind model produced negative CLV. Every NBA market had model 20-40% off from market. The model provides false confidence without injury data. Do not re-enable without an injury-aware data source.

### 5.3 Dead APIs (Tested, Do Not Reuse)

| API | Problem | Date Tested |
|---|---|---|
| Squiggle (AFL) | HTTP 522 -- server unreachable | 2026-03-28 |
| API-Sports (rugby) | All 142 leagues return null fields | 2026-03-28 |
| TheSportsDB standings | Empty for non-soccer leagues | 2026-03-28 |
| ESPN (Super League) | Returns empty teams/standings | 2026-03-29 |
| football-data.org BL2 | 403 Forbidden despite docs | 2026-04-04 |

---

## 6. Configuration (`config.toml`)

```toml
[llm]
daily_cap_usd = 5.0
downgrade_threshold_pct = 0.80

[triggers]
margin_min_paper = 0.080               # 8% edge minimum (raised from 2% based on trade data)
margin_min_live = 0.060                # Tighter for live -- but still high

[risk]
max_exposure = 0.85
drawdown_throttle_pct = 0.25           # Apply 50% throttle above 25% drawdown
drawdown_throttle_factor = 0.50
confidence_floor = 35
commission_pct = 0.05                  # Betfair standard (5%)
beta = 0.50                            # Bayesian update weight
liquidity_safety_factor = 0.70
kelly_base_fraction = 0.75             # Three-quarter Kelly
kelly_hard_cap = 0.10                  # Max 10% of bankroll per trade
min_market_liquidity_aud = 50.0
min_matched_volume_aud = 500.0
max_lay_probability = 0.90             # Block lays at odds < 1.11
min_lay_win_probability = 0.35         # Lay bets must have >= 35% win chance
dnb_min_match_odds_prob = 0.55         # DNB: raw match-odds prob >= 55%
slippage_model = "linear"
slippage_factor = 0.10
allow_in_play = false
reject_crossed_book = true
max_new_positions_per_cycle = 3

[scanner]
betfair_event_types = [1, 5, 61420, 1477, 7524, 7511]
betfair_hours_ahead = 12               # Scan 12h ahead to catch evening slate
betfair_min_hours_before_start = 3     # Only trade within 3h of kickoff
# Per-sport competition whitelists (Betfair competition IDs)
competition_ids_football = [10932509, 7129730, 55, 59, 81, 117, 88, 228]
competition_ids_baseball = [11196870]  # MLB only
competition_ids_hockey = [7825509]     # NHL only
competition_ids_afl = [11897406]       # AFL
competition_ids_rugby = [...]          # Super Rugby, Premiership, Top 14
competition_ids_rugby_league = [10564377, 11300061]  # NRL, Super League
```

---

## 7. Safety Architecture (18 Gates)

Each gate was added in response to a specific failure mode during paper trading.

| Gate | Trigger | Origin |
|---|---|---|
| Handshake | Model + Perplexity disagree on direction | 29-trade analysis: disagree = -$696 |
| DNB conviction | Raw match-odds prob < 55% for DNB markets | Charlton -$96 loss from inflated DNB edge |
| Edge threshold | Edge < 8% | Sub-8% trades: 2W/3L, -$25 |
| Time gate | Market > 3h from kickoff | Negative CLV from entering thin/stale markets |
| Extreme divergence | Model vs market gap > 30pp | Model error, not real edge |
| No-model gate | Stats unavailable or completeness < 75% | Harrogate -$49 from 50% completeness data |
| Lay win probability | 1 - p_fair < 35% for lay bets | Laying near-certainties = high risk/reward ratio |
| Volume gate | Matched volume < $500 | Thin markets have stale/unrepresentative prices |
| Liquidity gate | Available depth < $50 | Can't fill orders meaningfully |
| In-play gate | Market is in-play | No in-play engine |
| Crossed-book gate | Best back < best lay | Stale/suspended market data |
| Competition whitelist | Market not in whitelisted competition | Junk leagues crowd out viable markets |
| Niche league gate | Women's, youth, reserve, U-age | No stats coverage, unreliable model |
| Model clamp | Logistic output clamped to [0.20, 0.80] | Prevents overconfidence from small samples |
| Kill switch | `state/kill_switch.txt` exists | Manual emergency stop |
| Drawdown throttle | Kelly halved at 25% drawdown | Survive losing streaks |
| Rejection cache | Recently-failed markets cached for 2h | Prevents retrying dead markets |
| Max positions | Max 3 new positions per cycle | Prevents concentration spikes |

---

## 8. OMS Requirements (Betfair Live Phase)

### 8.1 Order State Machine

```
Pending --> Placed --> Partial --> Filled --> Trailing --> Exited
                |                     |
                +---> Cancelled       +---> Expired (TIF)
```

### 8.2 Order Types

| Type | When Used | Implementation |
|---|---|---|
| Fill-or-Kill (FOK) | Crossing the spread | Cancel immediately if not filled at target price |
| Limit + Synthetic TIF | Resting on book | Python timer cancels after 300s if unfilled |
| Persistence: LAPSE | All orders | Auto-cancelled by Betfair on in-play or suspension |

### 8.3 Operational Rules

- **Partial fills:** Recalculate remaining Kelly fraction on next cycle using updated exposure
- **Rate limiting:** `tenacity` exponential backoff on `TOO_MUCH_DATA` and API throttle responses
- **Suspension handling:** Clear all resting orders on `MarketSuspended` event; re-evaluate on resume
- **Reconciliation:** After every cycle, reconcile local state with Betfair `/listCurrentOrders`

---

## 9. Dashboard & Observability

**Streamlit + Plotly real-time dashboard:**

| Panel | Content |
|---|---|
| Top metrics | Bankroll, P&L, win rate, CLV, exposure, drawdown |
| Scan feed | Live per-market outcomes with cycle selector, Win %, outcome badges, entries table |
| Equity curve | Running bankroll vs time |
| Cash curve | Bankroll (including escrowed capital) vs time |
| CLV chart | Closing line value per trade + cumulative trend |
| Position table | Open positions with entry price, P&L, last seen price |
| Trade log | Full history with filters (settled/open/cancelled) |
| LLM cost tracker | Daily spend vs cap with progress bar |
| Realism checklist | Scored feature checklist |
| CSV export | Full trade history download |

---

## 10. File Structure

```
oracle-py/
+-- main.py                     # Entry point: scheduler, trade pipeline, all 18 gates
+-- config.toml                 # All settings (LLM, triggers, risk, scanner)
+-- .env                        # API keys + Betfair credentials
+-- requirements.txt            # Pinned pip dependencies
+-- README.md
+-- WHITEPAPER.MD
+-- DEVELOPMENT_PLAN.MD
+-- .gitignore
|
+-- logs/                       # Rotating daily log files (7-day retention)
+-- state/                      # oracle_state.json, scan_feed.json, fd_team_index.json
|
+-- src/
|   +-- config.py               # Pydantic-validated config from config.toml + .env
|   |
|   +-- enrichment/
|   |   +-- news.py             # NewsData.io + Google News RSS
|   |   +-- team_mapping.py     # Betfair <-> stats API name resolution
|   |   +-- trigger.py          # Event-driven deep trigger decision
|   |   +-- stats/
|   |       +-- __init__.py     # get_match_stats() dispatcher
|   |       +-- models.py       # MatchStats dataclass + completeness scoring
|   |       +-- football.py     # football-data.org (Poisson inputs)
|   |       +-- afl.py          # Official AFL API (form + ladder)
|   |       +-- baseball.py     # MLB Stats API
|   |       +-- rugby.py        # ESPN (rugby union)
|   |       +-- rugby_league.py # ESPN (NRL) + TheSportsDB fallback (Super League)
|   |       +-- hockey.py       # NHL API
|   |       +-- espn.py         # Shared ESPN client (standings + form)
|   |
|   +-- llm/
|   |   +-- client.py           # OpenRouter wrapper: routing, cost tracking, retries
|   |   +-- prompts.py          # JSON-forced prompt templates
|   |   +-- models.py           # Pydantic response validation
|   |
|   +-- strategy/
|   |   +-- bayesian.py         # Logit-space probability updater
|   |   +-- kelly.py            # Commission-aware Kelly + liability translation
|   |   +-- edge.py             # Executable edge calculation
|   |   +-- statistical_model.py# Poisson (football) / logistic (all others)
|   |
|   +-- scanner/
|   |   +-- betfair_scanner.py  # Per-sport Betfair queries with competition whitelists
|   |
|   +-- risk/
|   |   +-- manager.py          # Exposure gates, drawdown throttle, confidence floor
|   |
|   +-- execution/
|   |   +-- paper.py            # Paper broker: FOK, depth-aware fills, settlement
|   |
|   +-- dashboard/
|   |   +-- app.py              # Streamlit app with all dashboard panels
|   |
|   +-- storage/
|       +-- state_manager.py    # JSON persistence: load/save/atomic write
|       +-- scan_feed.py        # Per-market scan outcomes (48-cycle rolling window)
|       +-- rejection_cache.py  # In-memory TTL cache (2h)
|
+-- tests/                      # 238 tests
    +-- test_paper.py           # Depth fill, slippage, queue model, settlement, CLV
    +-- test_kelly.py           # Kelly formulas, liability, boundary inputs, hypothesis
    +-- test_bayesian.py        # Logit update, beta sensitivity, boundary probabilities
    +-- test_edge.py            # Edge signs, commission adjustment
    +-- test_risk.py            # Each gate fires independently
    +-- test_betfair_scanner.py # Mocked API, market parsing, resolution detection
    +-- test_team_mapping.py    # Fuzzy matching, Perplexity fallback, caching
    +-- test_stats.py           # Stats fetcher, completeness scoring
    +-- test_settlement_validator.py  # P&L consistency checks
    +-- test_structured_output.py     # JSON schema generation and validation
    +-- test_statistical_model.py     # Poisson + logistic model validation
```

---

## 11. Risk Considerations

This section is non-optional reading.

- **Betfair is a real-money exchange.** Bugs in Kelly sizing, liability translation, or OMS state can result in outsized losing bets. All Betfair code must pass unit tests and paper-mode validation before live deployment.
- **LLM outputs are probabilistic, not deterministic.** `sentiment_delta` is a noisy signal. `beta = 0.50` is calibrated but still amplifies noise. The handshake gate exists because the LLM is frequently wrong.
- **Statistical models are only as good as their data.** Free-tier APIs return 1-5 game samples early in the season. The [0.20, 0.80] clamp and 0.80 scoring coefficient exist to prevent overconfidence from small samples.
- **DNB renormalization inflates edge.** A marginal match-odds advantage becomes a large-looking DNB edge. The 55% conviction gate prevents backing teams the model barely favours.
- **Negative CLV is the real signal, not P&L.** Positive P&L with negative CLV means running hot on variance. The 3h time gate was added after discovering consistently negative CLV from early entries.
- **Partial fills are the norm, not the exception.** The system must handle them gracefully. Never assume an order is fully filled until confirmed.
- **Commission is not optional math.** At 5%, it eliminates edges below ~2.5%. Every formula prices commission in.
- **Kelly fractions > 1.0 must be hard-blocked.** A bug producing `f* > 1` would risk more than the bankroll.
- **Injury-blind models are worse than no model.** Basketball was disabled after every NBA trade showed model 20-40% off from market due to missing injury data.

---

## 12. Dependencies

```
# Core
httpx>=0.27
tenacity>=8.2
apscheduler>=3.10
pydantic>=2.7
pydantic-settings>=2.3
python-dotenv>=1.0
scipy>=1.13
toml>=0.10

# Dashboard
streamlit>=1.35
plotly>=5.22
pandas>=2.2

# Testing
pytest>=8.2
pytest-asyncio>=0.23

# Betfair
betfairlightweight
```

**External APIs (no additional SDKs required -- all via httpx):**
- OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
- NewsData.io (REST, API key, 200 credits/day)
- football-data.org (REST, API key, 10 req/min)
- Official AFL API (`aflapi.afl.com.au`, free, no auth)
- MLB Stats API (`statsapi.mlb.com`, free, no auth)
- NHL API (`api-web.nhle.com`, free, no auth)
- ESPN hidden API (`site.api.espn.com`, free, no auth)
- TheSportsDB (`thesportsdb.com`, free tier, fallback only)
- Betfair Exchange API (app key + session token)

---

## 13. Config Changes: Paper -> Live

| Parameter | Paper (Current) | Live (Phase 6) | Rationale |
|---|---|---|---|
| `margin_min` | 0.080 | 0.060 | Slightly tighter -- real spreads may be narrower |
| `confidence_floor` | 35 | 52-55 | Only trade when genuinely confident with real money |
| `kelly_base_fraction` | 0.75 | 0.25 | Quarter-Kelly reduces variance for live trading |
| `max_exposure` | 0.85 | 0.50 | Don't have >50% of bankroll at risk simultaneously |
| `drawdown_throttle_pct` | 0.25 | 0.15 | Throttle earlier with real money |
| `kelly_hard_cap` | 0.10 | 0.05 | Conservative until signal validated |
| `allow_in_play` | `false` | `false` | Keep blocked until dedicated in-play engine built |
| `MAX_SINGLE_BET_AUD` (code) | N/A | 20.0 | Hard-coded ceiling for first 30 days |
| `MAX_DAILY_LOSS_AUD` (code) | N/A | 100.0 | Kill switch if daily P&L hits this |

---
