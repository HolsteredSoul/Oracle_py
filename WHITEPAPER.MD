# Oracle Python Agent – High-Level Whitepaper

**v0.9.2 – Live Betfair Hardened, Execution-Ready Blueprint**  
**Autonomous Profit-Maximizing Prediction-Market Agent**  
**March 2026**  
**Author:** Pierre (@bandersbase)  
**Scope:** Manifold paper-money (safe tuning & validation) → Betfair Australia live (real-money execution)

## Executive Summary

Oracle is a Python-native autonomous agent built to **maximize long-term geometric bankroll growth** while aggressively sizing high-conviction edges and surviving Betfair’s harsh realities: wide bid-ask spreads, HFT latency advantage, partial fills, market suspensions, and thin liquidity.

It solves the Rust version’s main problems (low trade frequency, wasted LLM fees) through:

- Event-driven triggers (news/X/volatility spikes)
- LLM as sentiment calibrator (not raw probability generator)
- Correct commission-aware Kelly with liability translation
- Spread-adjusted execution thresholds
- Liquidity-capped sizing
- Robust Order Management System (OMS)
- Interactive Streamlit dashboard

**Projected realistic live Betfair performance**  

- 3–10 high-quality triggered trades per week  
- Average post-spread & commission edge: 2.5–7%  
- Max drawdown: <40%  
- Daily LLM cost: $0.5–4

## 1. Vision & Measurable Live Targets

Maximize geometric growth rate \( G = \mathbb{E}[\log(W_{t+1}/W_t)] \).

**Live Betfair targets** (validated via simulation + small-stakes testing):

- Executable post-spread edge: ≥2.5–4% after commission
- Trading frequency: 3–10 meaningful opportunities per week
- Max drawdown: <40%
- Fill rate on intended size: ≥70% (partial fills expected)
- Commission drag: fully modeled (2–5%)

## 2. Core Mathematics (Spread, Liquidity & Liability Hardened)

**Executable Edge**  
Let \( p_{\text{ask}} \) = best Back price (to buy YES), \( p_{\text{bid}} \) = best Lay price (to sell YES).  

For Backing YES:  
\[ \text{Executable Edge}_{\text{back}} = p_{\text{fair}} - p_{\text{ask}} \]

For Laying YES:  
\[ \text{Executable Edge}_{\text{lay}} = p_{\text{bid}} - p_{\text{fair}} \]

**Trigger condition**  
Execute Back only if \( p_{\text{fair}} > p_{\text{ask}} + \text{margin}_{\text{min}} \)  
Execute Lay only if \( p_{\text{fair}} < p_{\text{bid}} - \text{margin}_{\text{min}} \)  
(margin_min ≈ 0.015–0.03 post-commission)

**Commission-aware Kelly**  
For buying YES:  
\[ f^*_{\text{comm}} = \frac{p_{\text{fair}} - p_{\text{market}} - c \cdot p_{\text{fair}}(1 - p_{\text{market}})}{(1 - p_{\text{market}})(1 - c)} \]

**Oracle sizing** (confidence-scaled half-Kelly + drawdown throttle):  
\[ f = 0.5 \times f^*_{\text{comm}} \times c \times d \]  

- Base: half-Kelly  
- \( c \): LLM confidence scaler (confidence / 100, capped 0.5–1.0)  
- \( d \): drawdown throttle (1.0 if drawdown <20%, else 0.5)  
- Hard caps: per-bet ≤25% bankroll, total exposure ≤85%

**Liability vs Stake Translation** (critical for Betfair)  

- **Back bets**: Stake = f × Bankroll  
- **Lay bets**: Liability = f × Bankroll → Stake = Liability / (decimal_odds - 1)

**Liquidity-capped sizing**  
\[ \text{size}_{\text{max}} = \min\left( f \times \text{bankroll}, \text{available_liquidity_at_target_price} \times 0.7 \right) \]

## 3. Intelligence Layer – LLM as Sentiment Calibrator

**LLM routing via OpenRouter** (hobbyist-first):  

- 80% calls: Grok 4.1 Fast (X-native, cheap) + Gemini 3.1 Flash Lite + DeepSeek V3.2  
- Deep triggers only: Claude Sonnet 4.6 or DeepSeek R1  
- Daily spend cap + auto-downgrade to free models during tuning

**LLM role**: Outputs **sentiment_delta** (-1 to +1) and **uncertainty_penalty** (0-1) — **not** raw probability.  
Final \( p_{\text{fair}} \):  
\[ p_{\text{fair}} = \text{logit}^{-1}\left( \text{logit}(p_{\text{mid}}) + \beta \times \text{sentiment_delta} \right) \]  

- β start: 0.10–0.20 (tune in backtest)  
- p_mid = (p_ask + p_bid)/2 (prior)

**Python libraries** (minimal):  
`openrouter`, `httpx`, `tenacity`, `apscheduler`, `streamlit`, `plotly`, `pandas`, `pydantic`, `python-dotenv`, `scipy` (for logit)

**Prompt Templates** (JSON-forced)

1. **Light Batch Scan**  
   “Expert prediction trader. Given question + current price + brief news/X summary, output JSON only: sentiment_delta (-1 to +1), uncertainty_penalty (0-1), one-sentence rationale.”

2. **Deep Trigger Analysis**  
   “Step-by-step: 1. Synthesize news + X sentiment. 2. Estimate impact magnitude. Output JSON: sentiment_delta (-1 to +1), uncertainty_penalty (0-1), key_factors, rationale.”

3. **Exit Decision**  
   “Current position + updated news/X + sentiment_delta. Decide: HOLD / TRAIL-SELL / FULL-EXIT. Output JSON: decision, trailing_stop_pct, target_price, rationale.”

## 4. Event-Driven Flowcharts (Text Versions)

**Main Cycle**  

```text
Start: Timer (30 min) OR Trigger (news/X/vol spike)
   ↓
Is signal strong? (sentiment delta >0.28 OR volatility z >2.3 OR X momentum >0.32)
   ├─ No ──► Fast Batch Scan → get sentiment delta
   └─ Yes ──► Deep LLM + Full Enrichment → calculate Bayesian p_fair
                      ↓
Is executable edge > margin_min? (p_fair > ask + 0.015 OR p_fair < bid - 0.015)
   ├─ No ──► Log missed opportunity → sleep
   └─ Yes ──► Confidence-scaled half-Kelly + liquidity cap
                      ↓
Risk gates pass? (exposure ≤85%, confidence ≥52, drawdown throttle)
   ├─ No ──► Log skip → sleep
   └─ Yes ──► Execute via OMS (FOK if crossing, Limit + 300s TIF if resting)
                      ↓
Monitor: partial fills, trailing stop, suspensions, opportunistic exit
                      ↓
Reconcile bankroll/positions → update dashboard → loop
```

**Sizing Decision Tree**  

```text
Input: Bayesian Edge (post-spread, post-comm)
   ↓
Confidence > 55% ?
   ├─ No ──► Skip (log)
   └─ Yes ──► Check drawdown
             ↓
       Drawdown > 20% ?
         ├─ Yes ──► Force half-Kelly (k = 0.5)
         └─ No ──► Scale k = 0.5 to 1.0 (based on confidence)
                   ↓
             Apply hard cap: 85% total exposure
                   ↓
             Translate: Back Stake = f × Bankroll
                        Lay Liability = f × Bankroll → Stake = Liability / (odds - 1)
                   ↓
             Bet Size = min(Stake/Liability, liquidity × 0.7)
```

## 5. Triggers & Config Example (config.toml)

```toml
[llm]
primary = "grok-4.1-fast"
deep = "claude-sonnet-4.6"
daily_cap_usd = 5.0

[triggers]
news_sentiment_delta = 0.28
x_momentum = 0.32
volatility_z = 2.3
margin_min_paper = 0.025
margin_min_live = 0.015

[risk]
max_exposure = 0.85
drawdown_throttle = 0.5
confidence_floor = 52
commission_pct = 0.05
beta = 0.15
liquidity_safety_factor = 0.7
```

## 6. OMS Requirements (Betfair Live Phase)

- State machine: Pending → Placed → Partial → Filled → Trailing → Exited  
- Order types:  
  - Fill-or-Kill (FOK) when crossing spread (no hanging risk)  
  - Limit orders with synthetic Time-in-Force (Python timer cancels after 300s)  
- PersistenceType = LAPSE (auto-cancel on in-play/suspension)  
- Partial fills: Resize remaining Kelly fraction on next cycle  
- Rate-limit backoff: tenacity exponential retry (handle TOO_MUCH_DATA, API limits)

## 7. Dashboard & Observability

Streamlit + Plotly:  

- Live equity curve  
- Executable edge histogram (post-spread)  
- Fill rate & partial-fill log  
- Drawdown chart  
- Trigger & missed-opportunity log  
- CSV export

## 8. 20-Step Development Roadmap (Final)

1. Create `oracle-py` + venv + pip install listed libraries  
2. Add `.env` + `config.toml` (copy above)  
3. Write `main.py` skeleton with APScheduler  
4. Implement Manifold scanner (prices, volumes)  
5. Build enrichment: X API v2 + NewsData.io  
6. Code trigger detector  
7. Create OpenRouter wrapper  
8. Implement Light Batch Prompt + delta parsing  
9. Add Deep Trigger Prompt  
10. Build Bayesian updater (logit + β)  
11. Code correct commission-aware Kelly (YES/NO + liability translation)  
12. Implement confidence-scaled half-Kelly + liquidity cap  
13. Add risk gates  
14. Write paper-mode execution simulator  
15. Implement trailing stop + opportunistic exit  
16. Create Streamlit dashboard  
17. Add state JSON persistence  
18. Build backtesting harness (Manifold historical)  
19. Tune margin_min, β, triggers in backtest  
20. Phase 4: Betfair auth + OMS state machine (FOK, synthetic TIF, LAPSE) → small live stakes

## 9. Example File Structure

oracle-py/
├── main.py                    # Entry point: scheduler, main loop, CLI
├── config.toml                # All settings (LLM, triggers, risk, margins)
├── .env                       # OpenRouter key + secrets
├── requirements.txt           # Exact pip list
├── README.md
├── WHITEPAPER.MD              # Your final doc
├── .gitignore
├── data/                      # Historical Manifold dumps (CSV/JSON for backtesting)
├── logs/                      # Daily log files (rotating)
├── state/                     # oracle_state.json (bankroll, positions, priors, history)
├── src/
│   ├── **init**.py
│   ├── config.py              # Loads config.toml + .env (pydantic settings)
│   │
│   ├── enrichment/            # News, X sentiment, GDELT
│   │   ├── **init**.py
│   │   ├── news.py
│   │   └── x_sentiment.py
│   │
│   ├── llm/                   # OpenRouter client + prompts
│   │   ├── **init**.py
│   │   ├── client.py          # Routing, cost cap, retries
│   │   └── prompts.py         # All 3 JSON templates
│   │
│   ├── strategy/              # Core decision engine
│   │   ├── **init**.py
│   │   ├── bayesian.py        # logit update + β tuning
│   │   ├── kelly.py           # Correct commission-aware + liquidity cap
│   │   └── edge.py            # Executable edge calculation (ask/bid)
│   │
│   ├── scanner/               # Manifold market fetching
│   │   └── manifold.py
│   │
│   ├── risk/                  # Gates, drawdown throttle, exposure
│   │   └── manager.py
│   │
│   ├── execution/             # Paper mode now, Betfair later
│   │   ├── **init**.py
│   │   ├── paper.py           # Manifold simulation
│   │   └── betfair_oms.py     # Future OMS (state machine, FOK, synthetic TIF)
│   │
│   ├── dashboard/             # Streamlit app
│   │   ├── **init**.py
│   │   └── app.py             # Equity curve, edge histogram, etc.
│   │
│   ├── backtesting/           # Replay harness
│   │   └── harness.py
│   │
│   └── storage/               # State persistence
│       └── state_manager.py
├── tests/                     # Optional (pytest)
│   └── test_kelly.py
│
└── scripts/                   # One-off utilities (e.g., backtest runner)
