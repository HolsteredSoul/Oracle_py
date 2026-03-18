# Oracle Python Agent – Technical Whitepaper

**v1.0.0 – Live Betfair Hardened, Execution-Ready Blueprint**
**Autonomous Profit-Maximizing Prediction-Market Agent**
**March 2026**
**Author:** Pierre (@bandersbase)
**Scope:** Manifold paper-money (safe tuning & validation) → Betfair Australia live (real-money execution)

---

## Executive Summary

Oracle is a Python-native autonomous agent built to **maximize long-term geometric bankroll growth** while aggressively sizing high-conviction edges and surviving Betfair's harsh realities: wide bid-ask spreads, HFT latency advantage, partial fills, market suspensions, and thin liquidity.

It supersedes the prior Rust prototype, whose main weaknesses were low trade frequency and wasted LLM budget. Oracle fixes this through:

- **Event-driven triggers** — news, X/Twitter sentiment, and volatility spikes gate expensive LLM calls
- **LLM as calibrator, not oracle** — outputs `sentiment_delta` and `uncertainty_penalty`; the Bayesian updater does the math
- **Commission-aware Kelly** with explicit liability translation for Lay bets
- **Spread-adjusted execution thresholds** — no bet placed unless post-commission edge is real
- **Liquidity-capped sizing** — never move the market against yourself
- **Full OMS state machine** for Betfair phase (FOK, synthetic TIF, LAPSE)
- **Interactive Streamlit dashboard** for monitoring and post-trade analysis

**Projected realistic live Betfair performance:**

| Metric | Target |
|---|---|
| Trades per week | 3–10 high-quality |
| Post-spread & commission edge | 2.5–7% |
| Max drawdown | < 40% |
| Fill rate | ≥ 70% |
| Daily LLM cost | $0.50–4.00 |

---

## 1. Vision & Measurable Live Targets

**Objective:** Maximize geometric growth rate:

$$G = \mathbb{E}\bigl[\log(W_{t+1}/W_t)\bigr]$$

**Live Betfair targets** (validated via simulation before small-stakes go-live):

| Metric | Minimum Bar |
|---|---|
| Executable post-commission edge | ≥ 2.5–4% |
| Weekly opportunities | 3–10 |
| Max drawdown | < 40% |
| Fill rate on intended size | ≥ 70% |
| Commission drag modeled | 2–5% fully priced in |

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

Where `delta_min` ≈ 0.015–0.030 post-commission (tuned in backtest; tighter in live phase where spreads are narrower).

### 2.3 Commission-Aware Kelly

Let `c` = commission rate (e.g. 0.05 for 5%).

**Back bet** — result is the **stake** as a fraction of bankroll:

$$f_\text{back} = \frac{p_\text{fair} - p_\text{ask} - c \cdot p_\text{fair} \cdot (1 - p_\text{ask})}{(1 - p_\text{ask})(1 - c)}$$

**Lay bet** — result is the **liability** as a fraction of bankroll:

$$f_\text{lay} = \frac{p_\text{bid} - p_\text{fair} - c \cdot p_\text{bid} \cdot (1 - p_\text{fair})}{p_\text{bid} \cdot (1 - c)}$$

> `f_lay` outputs a **liability fraction**, not a stake fraction. The §2.4 translation step is mandatory before placing a Lay order.

**Oracle sizing** (confidence-scaled fractional Kelly with drawdown throttle):

$$f = k \cdot f^{*}_\text{comm} \cdot c_\text{conf} \cdot \lambda_\text{dd}$$

Where:
- `k` = base Kelly fraction (0.5 half-Kelly default)
- `c_conf` = LLM confidence scaler = `conf_score / 100`, clamped to [0.5, 1.0]
- `λ_dd` = drawdown throttle: 1.0 if drawdown < 20%, else 0.5
- Hard caps: per-bet ≤ 25% bankroll, total open exposure ≤ 85%
- **Assert `f_back` and `f_lay` ≤ 1.0 before applying Oracle sizing** — any result > 1 indicates a data error and must be rejected

### 2.4 Liability vs Stake Translation

This is critical for Betfair Lay bets:

- **Back bets:** `Stake = f_back × Bankroll`
- **Lay bets:** `Liability = f_lay × Bankroll` → `Stake = Liability / (decimal_odds − 1)`

### 2.5 Liquidity-Capped Sizing

$$\text{size}_\text{final} = \min\bigl(f \times \text{Bankroll},\ L_{\text{avail}} \times 0.70\bigr)$$

Where $L_{\text{avail}}$ = available liquidity at the target price tier. The 0.70 safety factor prevents Oracle from consuming > 70% of available liquidity.

---

## 3. Intelligence Layer — LLM as Sentiment Calibrator

### 3.1 LLM Routing via OpenRouter

Oracle routes LLM calls by cost/depth tradeoff:

| Tier | Models | Use Case |
|---|---|---|
| Fast (80% of calls) | Cheap fast models (e.g. flash/mini variants) | Light batch scans every 30 min |
| Deep (triggered only) | Reasoning-class models | News spikes, volatility events |
| Fallback | Free-tier models | During tuning phase when daily cap hit |

> **Implementation note:** Specific model IDs are configured in `config.toml` and can be updated as new models are released. Do not hardcode model strings in Python source.

**Daily spend cap enforced in code.** Auto-downgrade to free/cheap models if cap is within 20% of limit.

### 3.2 LLM Output Contract

The LLM never outputs a raw probability. It outputs a structured JSON:

```json
{
  "sentiment_delta": 0.18,
  "uncertainty_penalty": 0.12,
  "rationale": "One-sentence explanation"
}
```

- `sentiment_delta` ∈ [-1.0, +1.0]: directional signal strength
- `uncertainty_penalty` ∈ [0.0, 1.0]: reduces effective confidence

### 3.3 Bayesian Updater

Starting from the market mid-price as a prior:
$$p_\text{mid} = \frac{p_\text{ask} + p_\text{bid}}{2}$$

Apply logit-space update:
$$p_\text{fair} = \sigma\bigl(\text{logit}(p_\text{mid}) + \beta \cdot \Delta_{\text{sent}}\bigr)$$

Where `σ` is the sigmoid (logistic) function, `β` is the sentiment weight (default 0.15, tuned in backtest), and $\Delta_{\text{sent}}$ = `sentiment_delta`. This ensures `p_fair` remains in (0, 1) by construction.

**Effective confidence** (used for `c_conf`):
$$\text{conf} = c_{\text{base}} \times (1 - u_{\text{pen}})$$

Where $c_{\text{base}}$ = `base_conf` and $u_{\text{pen}}$ = `uncertainty_penalty`.

### 3.4 Prompt Templates (JSON-Forced)

**Template 1: Light Batch Scan**
```
You are an expert prediction-market trader.
Given: question text, current mid-price, brief news/X summary.
Output JSON only — no prose:
{
  "sentiment_delta": <float -1.0 to 1.0>,
  "uncertainty_penalty": <float 0.0 to 1.0>,
  "rationale": "<one sentence>"
}
```

**Template 2: Deep Trigger Analysis**
```
You are an expert prediction-market trader. Think step-by-step:
1. Synthesize all provided news and X/Twitter sentiment signals.
2. Estimate directional impact and magnitude.
3. Note key uncertainties.
Output JSON only:
{
  "sentiment_delta": <float -1.0 to 1.0>,
  "uncertainty_penalty": <float 0.0 to 1.0>,
  "key_factors": ["<factor1>", "<factor2>"],
  "rationale": "<two to three sentences>"
}
```

**Template 3: Exit Decision**
```
Given: current position details, updated news/X sentiment, current market price.
Decide whether to hold or exit.
Output JSON only:
{
  "decision": "HOLD" | "TRAIL-SELL" | "FULL-EXIT",
  "trailing_stop_pct": <float>,
  "target_price": <float>,
  "rationale": "<one to two sentences>"
}
```

---

## 4. Event-Driven Architecture

### 4.1 Main Cycle

```text
START: Timer (30 min) OR Trigger (news / X / volatility spike)
  │
  ├─ Trigger strength check:
  │   sentiment_delta > 0.28  OR  volatility_z > 2.3  OR  x_momentum > 0.32?
  │
  ├─ No  ──► Light Batch Scan (cheap LLM) → get sentiment_delta
  │
  └─ Yes ──► Deep LLM + Full Enrichment → compute Bayesian p_fair
                  │
                  ▼
       Executable edge > delta_min?
       (p_fair > ask + 0.015  OR  p_fair < bid - 0.015)
  │
  ├─ No  ──► Log missed opportunity → sleep
  │
  └─ Yes ──► Confidence-scaled half-Kelly + liquidity cap
                  │
                  ▼
       Risk gates pass?
       (exposure ≤ 85%,  conf ≥ 52,  drawdown throttle OK)
  │
  ├─ No  ──► Log skip → sleep
  │
  └─ Yes ──► Execute via OMS
             (FOK if crossing spread; Limit + 300s synthetic TIF if resting)
                  │
                  ▼
       Monitor: partial fills, trailing stop, suspensions, opportunistic exit
                  │
                  ▼
       Reconcile bankroll/positions → update dashboard → LOOP
```

### 4.2 Sizing Decision Tree

```text
INPUT: Bayesian edge (post-spread, post-commission)
  │
  ▼
Confidence ≥ 52%?
  ├─ No  ──► Skip (log reason)
  │
  └─ Yes ──►  Drawdown > 20%?
                ├─ Yes ──► Force k = 0.5 (half-Kelly only)
                │
                └─ No  ──► k scales 0.5→1.0 based on confidence
                               │
                               ▼
                    Apply hard cap: 85% total exposure
                               │
                               ▼
                    Back:  Stake    = f × Bankroll
                    Lay:   Liability = f × Bankroll
                           Stake    = Liability / (odds − 1)
                               │
                               ▼
                    Final size = min(Stake/Liability, liquidity × 0.70)
```

---

## 5. Configuration (`config.toml`)

```toml
[llm]
# Update model IDs here as new models are released — never hardcode in source
fast_model = "grok-3-mini-fast"         # example; check OpenRouter for current cheapest
deep_model = "claude-sonnet-4-6"        # example; use a reasoning-class model
daily_cap_usd = 5.0
downgrade_threshold_pct = 0.80          # switch to free models at 80% of daily cap

[triggers]
news_sentiment_delta = 0.28
x_momentum = 0.32
volatility_z = 2.3
margin_min_paper = 0.025                # looser for paper trading
margin_min_live = 0.015                 # tighter for live execution

[risk]
max_exposure = 0.85
drawdown_throttle_pct = 0.20            # apply 50% throttle above this drawdown level
drawdown_throttle_factor = 0.50
confidence_floor = 52                   # minimum effective confidence to trade
commission_pct = 0.05                   # Betfair standard (5%)
beta = 0.15                             # Bayesian update weight (tune in backtest)
liquidity_safety_factor = 0.70
kelly_base_fraction = 0.50              # half-Kelly

[scanner]
poll_interval_sec = 1800                # 30 min base timer
manifold_min_volume = 500               # ignore thin markets
manifold_min_prob_range = [0.05, 0.95]  # avoid nearly-resolved markets

[betfair]
# Phase 4 only — leave empty during paper trading
api_key = ""
session_token = ""
min_bet_aud = 2.00
synthetic_tif_sec = 300
```

---

## 6. OMS Requirements (Betfair Live Phase)

### 6.1 Order State Machine

```
Pending ──► Placed ──► Partial ──► Filled ──► Trailing ──► Exited
                │                     │
                └──► Cancelled        └──► Expired (TIF)
```

### 6.2 Order Types

| Type | When Used | Implementation |
|---|---|---|
| Fill-or-Kill (FOK) | Crossing the spread | Cancel immediately if not filled at target price |
| Limit + Synthetic TIF | Resting on book | Python timer cancels after 300s if unfilled |
| Persistence: LAPSE | All orders | Auto-cancelled by Betfair on in-play or suspension |

### 6.3 Operational Rules

- **Partial fills:** Recalculate remaining Kelly fraction on next cycle using updated exposure
- **Rate limiting:** `tenacity` exponential backoff on `TOO_MUCH_DATA` and API throttle responses
- **Suspension handling:** Clear all resting orders on `MarketSuspended` event; re-evaluate on resume
- **Reconciliation:** After every cycle, reconcile local state with Betfair `/listCurrentOrders` to catch any divergence

---

## 7. Dashboard & Observability

**Streamlit + Plotly real-time dashboard:**

| Panel | Content |
|---|---|
| Equity curve | Running bankroll vs time |
| Edge histogram | Post-spread executable edge distribution |
| Fill rate log | Intended vs actual fill size per trade |
| Drawdown chart | Rolling max drawdown |
| Trigger log | All trigger events with signal strengths |
| Missed-opportunity log | Edges below threshold (for tuning `delta_min`) |
| Position table | Open positions with current P&L |
| LLM cost tracker | Daily spend vs cap |
| CSV export | Full trade history download |

---

## 8. File Structure

```
oracle-py/
├── main.py                     # Entry point: scheduler, main loop, CLI args
├── config.toml                 # All settings (LLM, triggers, risk, Betfair)
├── .env                        # OpenRouter key, X API key, Betfair credentials
├── requirements.txt            # Pinned pip dependencies
├── README.md
├── WHITEPAPER.MD
├── .gitignore                  # Must exclude .env, state/, logs/
│
├── data/                       # Historical Manifold dumps (CSV/JSON for backtesting)
├── logs/                       # Rotating daily log files
├── state/                      # oracle_state.json (bankroll, positions, priors, history)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Loads config.toml + .env via pydantic-settings
│   │
│   ├── enrichment/
│   │   ├── __init__.py
│   │   ├── news.py             # NewsData.io API client
│   │   └── x_sentiment.py      # X API v2 client + keyword aggregation
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # OpenRouter wrapper: routing, cost tracking, retries
│   │   └── prompts.py          # All 3 JSON prompt templates
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── bayesian.py         # logit-space updater + β calibration
│   │   ├── kelly.py            # Commission-aware Kelly + liability translation
│   │   └── edge.py             # Executable edge calculation (ask/bid/commission)
│   │
│   ├── scanner/
│   │   ├── __init__.py
│   │   └── manifold.py         # Manifold API: market list, prices, volumes
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── manager.py          # Exposure gates, drawdown throttle, confidence floor
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── paper.py            # Manifold paper-money simulation
│   │   └── betfair_oms.py      # Phase 4: OMS state machine, FOK, synthetic TIF
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py              # Streamlit app: all panels listed in §7
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── harness.py          # Historical replay, parameter sweep
│   │
│   └── storage/
│       ├── __init__.py
│       └── state_manager.py    # JSON persistence: load/save/atomic write
│
└── tests/
    ├── __init__.py
    ├── test_kelly.py           # Kelly formula unit tests (edge cases, liability math)
    ├── test_bayesian.py        # Bayesian updater tests (boundary probs, β sensitivity)
    ├── test_edge.py            # Edge calculation tests
    ├── test_risk.py            # Risk gate tests
    └── test_oms.py             # OMS state machine tests (Phase 4)
```

---

## 9. Risk Considerations

This section is non-optional reading.

- **Betfair is a real-money exchange.** Bugs in Kelly sizing, liability translation, or OMS state can result in outsized losing bets. All Betfair code must pass unit tests and paper-mode validation before live deployment.
- **LLM outputs are probabilistic, not deterministic.** `sentiment_delta` is a noisy signal. `β = 0.15` is intentionally conservative. Do not increase β above 0.30 without extended backtesting evidence.
- **Partial fills are the norm, not the exception.** The system must handle them gracefully. Never assume an order is fully filled until confirmed via `/listCurrentOrders`.
- **Commission is not optional math.** At 5%, it eliminates edges below ~2.5%. Every formula in §2 prices commission in. Do not skip it during paper-mode testing.
- **Kelly fractions > 1.0 must be hard-blocked.** A bug producing `f* > 1` would risk more than the bankroll. Add an assertion in `kelly.py`.
- **Drawdown recovery is slow.** The 40% max drawdown target implies starting recovery from a 40% loss. Tighten early limits (max_exposure = 0.60) for the first month of live trading.

---

## 10. Dependencies

```
# requirements.txt
httpx>=0.27
tenacity>=8.2
apscheduler>=3.10
streamlit>=1.35
plotly>=5.22
pandas>=2.2
pydantic>=2.7
pydantic-settings>=2.3
python-dotenv>=1.0
scipy>=1.13
toml>=0.10
pytest>=8.2
pytest-asyncio>=0.23
```

> **OpenRouter:** Call via `httpx` with `Authorization: Bearer <key>` to `https://openrouter.ai/api/v1/chat/completions`. No additional SDK required.
> **X API v2:** Use `httpx` with OAuth 2.0 Bearer token. Free tier limits apply.
> **NewsData.io:** REST API via `httpx`. Free tier: 200 credits/day.
> **Manifold:** Public REST API, no auth required for read-only access.
> **Betfair:** `betfairlightweight` library (Phase 4 only).


---

## 11. Edge Source Assessment & Upgrade Path

*Added March 2026 after Phase 4 paper trading validation.*

### 11.1 Honest Assessment of the Current Intelligence Layer

The infrastructure (scanner, risk management, Kelly sizing, settlement, dashboard) is production-quality and battle-tested through Phase 4. However, **the current intelligence layer — LLM reading news headlines — is unlikely to generate sustained profit on live Betfair exchange markets.** This section explains why and outlines the upgrade path.

**The adversary is strong.** Betfair exchange prices are not soft bookmaker odds. They are set by thousands of participants including professional syndicates with statistical models, real-time team sheet scrapers, injury databases, Elo rating systems, expected goals (xG) models, and decades of domain expertise. By the time a price is live on the exchange, it already incorporates most publicly available information.

**The current signal is weak.** The agent reads a few news headlines (via NewsData.io and Google News RSS) and asks an LLM (Grok-fast or Claude-Sonnet) to return a `sentiment_delta`. This is a qualitative, vibes-based assessment — e.g., "Macarthur lost 4-1 recently, so they're probably worse than priced." That reasoning, while correct, is exactly what the market has already priced in. The headlines the LLM reads are hours to days old; the market moved on that information long before the agent saw it.

**The math is marginal.** Betfair takes 5% commission on net winnings. The real spread on match-odds markets is typically 2–4%. To overcome commission + spread, the agent needs to be right by more than ~7–9% consistently. The LLM's delta adjustments (typically 0.05–0.20 in probability space) are in the right ballpark mathematically, but they need to be *systematically correct* — not just plausible-sounding. There is no evidence yet that "LLM reads headlines" produces calibrated probability improvements over Betfair consensus.

**The longshot bias is already priced.** The LLM frequently cites "longshot bias" (punters overbet longshots, underbet favorites) as a reason to trade. This is a real phenomenon in bookmaker markets, but Betfair is an exchange — the bias is significantly weaker because participants are trading against each other, not against a bookmaker's margin. The LLM does not distinguish between bookmaker odds and exchange odds.

### 11.2 What Would Make This Profitable

The platform is ready. The intelligence layer needs upgrading. These are ranked by expected impact:

#### Tier 1: Statistical Data Integration (Highest Impact)

Replace or supplement LLM-headline analysis with actual sports data. APIs such as football-data.org, Sportmonks, FBref, or AFL Tables provide match statistics — expected goals (xG), shots on target, possession, head-to-head records, team form over last 5/10/20 matches.

A simple logistic regression or gradient-boosted model trained on xG and form data would produce better-calibrated probability estimates than an LLM reading headlines. The LLM can still add value for qualitative factors (manager sacked, dressing room turmoil, travel fatigue) but should not be the primary signal source.

**Implementation path:**
```
src/enrichment/stats.py          — fetch xG, form, H2H from a sports data API
src/strategy/statistical_model.py — logistic/XGBoost model producing p_fair
src/llm/prompts.py               — LLM receives statistical model output + news
                                    as inputs; provides adjustment delta on top
```

The Bayesian framework already supports this: replace `p_mid` (market price) as the prior with the statistical model's `p_model`, then use the LLM delta as a secondary adjustment.

#### Tier 2: Pre-Match Team Sheet Edge (Time-Sensitive Alpha)

Team lineups are announced 60–90 minutes before kickoff. If the agent ingests lineups and compares them to expected lineups (key player missing, unexpected formation change, debut player starting), that is a genuine informational edge that moves markets. The window is narrow (60–90 minutes), and the agent must act before the market fully adjusts.

**Implementation path:**
- Subscribe to a lineup API (e.g., Sportmonks, API-Football)
- Compare announced XI to predicted XI
- Trigger a deep LLM analysis with lineup diff as context
- Execute within the pre-match window before market adjusts

This is where LLMs genuinely add value — interpreting the *significance* of a lineup change (e.g., "star striker rested" vs "star striker injured") is qualitative reasoning that statistical models struggle with.

#### Tier 3: Closing Line Value (CLV) Tracking (Diagnostic, Not Directly Profitable)

Closing line value is the industry-standard diagnostic for whether a bettor has real edge. Track whether the agent's entry price is better than the closing price (the final price when the market suspends at kickoff).

If Oracle consistently beats the closing line, it has real edge — even if individual bets lose. If it does not beat the closing line, it is not profitable long-term regardless of short-term results.

**Implementation:** Add a `closing_price` field to the Trade record. At settlement, fetch the final pre-kickoff price from Betfair and compute `CLV = entry_price - closing_price`. Track aggregate CLV in the dashboard.

#### Tier 4: Market Microstructure Signals (Advanced)

Betfair provides real-time order book depth. Price movements, volume spikes, and order-book imbalances in the minutes before an event can signal informed money entering the market. This is effectively front-running sharp bettors — not by seeing their orders, but by detecting market footprint patterns.

This is significantly more complex to implement and requires high-frequency data collection. It is a Phase 6+ enhancement.

### 11.3 Recommended Transition Plan

| Phase | Signal Source | Purpose |
|---|---|---|
| Phase 4 (current) | LLM + headlines | Validate pipeline mechanics, collect trade data |
| Phase 5A (new) | Add CLV tracking | Diagnose whether current signal has value |
| Phase 5B (new) | Add statistical model (xG/form) | Replace LLM as primary signal; LLM becomes secondary |
| Phase 5C (new) | Add lineup ingestion | Time-sensitive alpha during pre-match window |
| Phase 6 | Statistical model + LLM + lineups | Live money with validated multi-source signal |

### 11.4 Config Changes: Paper → Live

| Parameter | Paper (Phase 4) | Live (Phase 6) | Rationale |
|---|---|---|---|
| `margin_min` | 0.030 | 0.025–0.030 | Wider margin compensates for real slippage |
| `confidence_floor` | 35 | 52–55 | Only trade when genuinely confident with real money |
| `kelly_base_fraction` | 0.50 | 0.25 | Quarter-Kelly reduces variance for live trading |
| `max_exposure` | 0.85 | 0.50 | Don't have >50% of bankroll at risk simultaneously |
| `drawdown_throttle_pct` | 0.20 | 0.15 | Throttle earlier with real money |
| `_HARD_CAP` (code) | 0.10 | 0.15 | Raise slightly once signal is validated |
| `MAX_SINGLE_BET_AUD` (code) | N/A | 20.0 | Hard-coded ceiling for first 30 days |
| `MAX_DAILY_LOSS_AUD` (code) | N/A | 100.0 | Kill switch if daily P&L hits this |

---

