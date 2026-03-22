"""Oracle Paper Trading Dashboard.

Reads state/oracle_state.json and state/llm_spend.json.
Zero coupling to the running agent process — pure JSON reader.

Run separately:
    streamlit run src/dashboard/app.py

Auto-refreshes every 60 seconds.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_STATE_PATH = Path("state/oracle_state.json")
_SPEND_PATH = Path("state/llm_spend.json")
_REFRESH_INTERVAL_SEC = 60
_DAILY_CAP_USD = 5.0   # default; overridden by spend file if available
_LOCAL_TZ = timezone(timedelta(hours=8))  # WAST (Perth)
_DISPLAY_FMT = "%b %d, %I:%M %p"          # e.g. "Mar 22, 09:30 PM"


def _to_local(iso_str: str, fmt: str = _DISPLAY_FMT) -> str:
    """Parse an ISO-8601 UTC string and return it formatted in local time."""
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_LOCAL_TZ).strftime(fmt)
    except Exception:
        return iso_str


def _df_timestamps_to_local(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """Convert a UTC-aware datetime column to local timezone for display."""
    if col in df.columns and hasattr(df[col], "dt"):
        df[col] = df[col].dt.tz_convert(_LOCAL_TZ)
    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_state() -> dict:
    """Load raw oracle state dict. Returns {} on missing/corrupt file."""
    if not _STATE_PATH.exists():
        return {}
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_spend() -> dict:
    """Load LLM spend dict. Returns {} on missing file."""
    if not _SPEND_PATH.exists():
        return {}
    try:
        return json.loads(_SPEND_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Data transformations
# ---------------------------------------------------------------------------

def _trade_entry_cost(t: dict) -> float:
    """Return the capital deployed at entry for a trade.

    For back bets, cost = stake_abs (the stake is escrowed).
    For lay bets, cost = liability_abs if available, else derived from
    filled_size * bankroll_before.
    """
    if t.get("direction") == "lay":
        # Prefer stored liability; fall back to filled_size * bankroll_before
        liab = t.get("liability_abs")
        if liab is not None:
            return liab
        return t.get("filled_size", 0) * t.get("bankroll_before", 0)
    return t.get("stake_abs", 0)


def compute_equity_curve(trade_history: list[dict], initial_bankroll: float) -> pd.DataFrame:
    """Reconstruct equity curve from trade history.

    Equity = cash (bankroll_after) + capital deployed in open positions.
    Each trade entry deploys capital; each settlement returns it (± P&L).
    This tracks total equity, not just cash on hand.
    """
    # Build a chronological event stream of entries and settlements
    events: list[dict] = []
    for t in trade_history:
        cost = _trade_entry_cost(t)
        events.append({
            "timestamp": t.get("timestamp", ""),
            "type": "entry",
            "cost": cost,
            "pnl": 0,
        })
        if t.get("status") in ("settled", "cancelled") and t.get("exit_timestamp"):
            events.append({
                "timestamp": t["exit_timestamp"],
                "type": "settle",
                "cost": cost,
                "pnl": t.get("pnl", 0) or 0,
            })

    rows = [{"timestamp": "start", "equity": initial_bankroll}]
    equity = initial_bankroll
    deployed = 0.0

    for ev in sorted(events, key=lambda e: e["timestamp"]):
        if ev["type"] == "entry":
            deployed += ev["cost"]
        else:  # settle — capital returns, P&L realised
            deployed -= ev["cost"]
            equity += ev["pnl"]
        rows.append({"timestamp": ev["timestamp"], "equity": equity})

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    _df_timestamps_to_local(df)
    return df


def compute_cash_curve(trade_history: list[dict], initial_bankroll: float) -> pd.DataFrame:
    """Reconstruct cash (bankroll) curve from trade history.

    Cash drops on entry (escrow) and returns on settlement (cost + P&L).
    This is the actual bankroll trajectory, not total equity.
    """
    events: list[dict] = []
    for t in trade_history:
        cost = _trade_entry_cost(t)
        events.append({"timestamp": t.get("timestamp", ""), "delta": -cost})
        if t.get("status") in ("settled", "cancelled") and t.get("exit_timestamp"):
            pnl = t.get("pnl", 0) or 0
            events.append({"timestamp": t["exit_timestamp"], "delta": cost + pnl})

    rows = [{"timestamp": "start", "cash": initial_bankroll}]
    cash = initial_bankroll
    for ev in sorted(events, key=lambda e: e["timestamp"]):
        cash += ev["delta"]
        rows.append({"timestamp": ev["timestamp"], "cash": cash})

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    _df_timestamps_to_local(df)
    return df



def holding_hours(entry_timestamp: str) -> float:
    """Compute hours since entry_timestamp (ISO-8601 UTC)."""
    try:
        entry = datetime.fromisoformat(entry_timestamp)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - entry
        return round(delta.total_seconds() / 3600, 1)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------------

def render_equity_panel(equity_df: pd.DataFrame) -> None:
    """Panel 2: Equity curve — total equity vs time."""
    st.subheader("Equity Curve")
    if equity_df.empty or len(equity_df) < 2:
        st.info("Not enough trades to display equity curve yet.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["timestamp"],
        y=equity_df["equity"],
        mode="lines+markers",
        line=dict(color="#00CC96", width=2),
        name="Equity",
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Equity (AUD)",
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cash_panel(cash_df: pd.DataFrame) -> None:
    """Panel 3: Cash / liquidity tracker — available bankroll vs time."""
    st.subheader("Cash (Available Liquidity)")
    if cash_df.empty or len(cash_df) < 2:
        st.info("Not enough data for cash chart yet.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cash_df["timestamp"],
        y=cash_df["cash"],
        fill="tozeroy",
        fillcolor="rgba(99,110,250,0.20)",
        line=dict(color="#636EFA", width=2),
        name="Cash",
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cash (AUD)",
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_panel(trade_history: list[dict]) -> None:
    """Panel 4: Realised P&L — bar per settled trade + cumulative line."""
    st.subheader("Realised P&L")
    settled = [t for t in trade_history if t.get("status") == "settled" and t.get("pnl") is not None]
    if not settled:
        st.info("No settled trades yet.")
        return

    df = pd.DataFrame(settled)[["timestamp", "pnl", "market_id"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    _df_timestamps_to_local(df)
    df["cumulative_pnl"] = df["pnl"].cumsum()
    colors = ["#00CC96" if p >= 0 else "#EF553B" for p in df["pnl"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["pnl"],
        marker_color=colors,
        name="Trade P&L",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["cumulative_pnl"],
        mode="lines",
        line=dict(color="#636EFA", width=2, dash="dot"),
        name="Cumulative P&L",
        yaxis="y2",
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="P&L (AUD)",
        yaxis2=dict(title="Cumulative P&L", overlaying="y", side="right"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_clv_panel(trade_history: list[dict]) -> None:
    """Phase 5A.1: Closing Line Value panel — per-trade CLV + aggregate."""
    st.subheader("Closing Line Value (CLV)")

    settled_with_clv = [
        t for t in trade_history
        if t.get("status") == "settled" and t.get("clv") is not None
    ]

    if not settled_with_clv:
        st.info("No settled trades with CLV data yet. CLV is recorded at settlement.")
        return

    df = pd.DataFrame(settled_with_clv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    _df_timestamps_to_local(df)

    avg_clv = df["clv"].mean()
    total_trades_clv = len(df)
    positive_clv_pct = (df["clv"] > 0).sum() / total_trades_clv * 100
    cumulative_clv = df["clv"].cumsum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg CLV", f"{avg_clv:+.4f}",
                delta="Edge detected" if avg_clv > 0 else "No edge",
                delta_color="normal" if avg_clv > 0 else "inverse")
    col2.metric("CLV+ Rate", f"{positive_clv_pct:.0f}%",
                delta=f"{total_trades_clv} trades")
    col3.metric("Cumulative CLV", f"{cumulative_clv.iloc[-1]:+.4f}")

    colors = ["#00CC96" if c > 0 else "#EF553B" for c in df["clv"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(df) + 1)),
        y=df["clv"],
        marker_color=colors,
        name="Trade CLV",
    ))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=cumulative_clv,
        mode="lines",
        line=dict(color="#636EFA", width=2),
        name="Cumulative CLV",
        yaxis="y2",
    ))
    fig.update_layout(
        xaxis_title="Trade #",
        yaxis_title="CLV",
        yaxis2=dict(title="Cumulative CLV", overlaying="y", side="right"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=280,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Phase 5A.1 exit gate interpretation
    if total_trades_clv >= 100:
        if avg_clv > 0.005:
            st.success(
                f"Positive CLV across {total_trades_clv} trades — signal has real edge. "
                f"Phase 5A.1 exit gate: PASS. Proceed to Phase 5 backtesting."
            )
        elif avg_clv > 0:
            st.warning(
                f"Marginally positive CLV ({avg_clv:+.4f}) — edge is thin. "
                f"Consider proceeding to Phase 5A.2 (statistical model) to strengthen signal."
            )
        else:
            st.error(
                f"Negative CLV across {total_trades_clv} trades — signal does not beat closing line. "
                f"Phase 5A.1 exit gate: FAIL. Proceed to Phase 5A.2 (statistical data integration). "
                f"Do NOT deploy live money with current signal."
            )
    elif total_trades_clv >= 30:
        direction_text = "positive" if avg_clv > 0 else "negative"
        st.caption(
            f"Preliminary CLV is {direction_text} ({avg_clv:+.4f}) across {total_trades_clv} trades. "
            f"Need 100+ for Phase 5A.1 exit gate."
        )
    else:
        st.caption(f"Collecting data — {total_trades_clv}/100 trades with CLV recorded.")


def render_position_table(positions: dict, trade_history: list[dict] | None = None) -> None:
    """Panel 5: Open positions table with model prediction data."""
    st.subheader(f"Open Positions ({len(positions)})")
    if not positions:
        st.info("No open positions.")
        return

    # Build trade lookup for enriching positions with model data
    trade_by_id: dict[str, dict] = {}
    if trade_history:
        for t in trade_history:
            trade_by_id[t.get("trade_id", "")] = t

    rows = []
    for mkt_id, pos in positions.items():
        mst = pos.get("market_start_time", "")
        event_date = ""
        if mst:
            event_date = _to_local(mst)
        cost = (
            pos.get("liability_abs", 0)
            if pos.get("direction") == "lay"
            else pos.get("stake_abs", 0)
        )

        # Enrich from matching Trade record
        trade = trade_by_id.get(pos.get("trade_id", ""), {})
        edge = trade.get("edge")
        p_fair = pos.get("p_fair_at_entry") or trade.get("p_fair")
        conf = trade.get("conf_score")
        kelly = trade.get("kelly_f_final")

        rows.append({
            "Market": pos.get("question", "")[:60],
            "Event": event_date,
            "Dir": pos.get("direction", ""),
            "Entry": f"{pos.get('entry_price', 0):.3f}",
            "P(fair)": f"{p_fair:.3f}" if p_fair else "—",
            "Edge": f"{edge:.1%}" if edge else "—",
            "Conf": f"{conf:.0f}" if conf else "—",
            "Kelly": f"{kelly:.3f}" if kelly else "—",
            "Cost (AUD)": f"{cost:.2f}",
            "Holding (h)": holding_hours(pos.get("entry_timestamp", "")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_trade_log(trade_history: list[dict]) -> None:
    """Panel 6: Recent trade log — last 20 trades, newest first."""
    st.subheader("Trade Log (last 20)")
    if not trade_history:
        st.info("No trades logged yet.")
        return

    cols = ["timestamp", "question", "direction", "filled_size",
            "fill_price", "edge", "p_fair", "conf_score", "pnl", "status"]
    rows = []
    for t in reversed(trade_history[-20:]):
        rows.append({c: t.get(c, "") for c in cols})

    df = pd.DataFrame(rows)
    # Convert timestamps to local readable format
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(
            lambda v: _to_local(v) if isinstance(v, str) and v else v
        )
    df.rename(columns={"question": "Market"}, inplace=True)
    # Format numerics
    for col in ["filled_size", "fill_price", "edge", "p_fair", "conf_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
    if "pnl" in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").round(2)

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_llm_cost_panel(spend: dict) -> None:
    """Panel 7: LLM cost — today's spend vs daily cap."""
    st.subheader("LLM Cost")
    today = str(datetime.now(_LOCAL_TZ).date())
    today_spend = spend.get(today, 0.0)
    cap = _DAILY_CAP_USD

    col1, col2 = st.columns(2)
    col1.metric("Today's Spend", f"${today_spend:.4f}")
    col2.metric("Daily Cap", f"${cap:.2f}")

    progress = min(today_spend / cap, 1.0) if cap > 0 else 0.0
    color = "normal" if progress < 0.80 else "inverse"
    st.progress(progress)
    st.caption(f"{progress * 100:.1f}% of daily cap used")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Oracle Dashboard",
        page_icon="🔮",
        layout="wide",
    )
    st.title("Oracle Paper Trading Dashboard")

    state_raw = load_state()
    spend = load_spend()

    if not state_raw:
        st.warning("No state file found at state/oracle_state.json. Is the agent running?")
        time.sleep(_REFRESH_INTERVAL_SEC)
        st.rerun()
        return

    trade_history = state_raw.get("trade_history", [])
    positions = state_raw.get("positions", {})
    bankroll = state_raw.get("bankroll", 1000.0)
    peak = state_raw.get("peak_bankroll", 1000.0)
    last_updated = state_raw.get("last_updated", "unknown")
    created_at = state_raw.get("created_at", "")

    # Compute equity = cash + capital deployed in open positions
    deployed_capital = 0.0
    for pos in positions.values():
        if pos.get("direction") == "lay":
            deployed_capital += pos.get("liability_abs", 0)
        else:
            deployed_capital += pos.get("stake_abs", 0)
    equity = bankroll + deployed_capital

    # --- Top metrics row ---
    st.caption(f"Last updated: {_to_local(last_updated)}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity", f"${equity:.2f}")
    col2.metric("Cash", f"${bankroll:.2f}")
    col3.metric("Deployed", f"${deployed_capital:.2f}")
    col4.metric("Open Positions", len(positions))
    settled_count = sum(1 for t in trade_history if t.get("status") == "settled")
    col5.metric("Settled Trades", settled_count)

    st.divider()

    # --- Charts ---
    initial_bankroll = 1000.0  # DEFAULT_BANKROLL
    equity_df = compute_equity_curve(trade_history, initial_bankroll)
    cash_df = compute_cash_curve(trade_history, initial_bankroll)

    left, right = st.columns(2)
    with left:
        render_equity_panel(equity_df)
    with right:
        render_cash_panel(cash_df)

    render_pnl_panel(trade_history)
    render_clv_panel(trade_history)    # Phase 5A.1

    st.divider()

    render_position_table(positions, trade_history)
    render_trade_log(trade_history)

    st.divider()

    render_llm_cost_panel(spend)

    # --- Auto-refresh ---
    time.sleep(_REFRESH_INTERVAL_SEC)
    st.rerun()


if __name__ == "__main__":
    main()
