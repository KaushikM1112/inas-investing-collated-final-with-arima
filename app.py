
import os, io, json, math
from datetime import datetime
import typing as t

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from hotfix_ma import moving_average_forgiving, to_series_close
from insights_helpers import ema, moving_averages, volatility_annualised, sharpe_ratio, lin_forecast, recommend_allocation
from insights_forecast import best_effort_forecast

try:
    import yfinance as yf
except Exception:
    yf = None

# Optional Google Sheets (kept off by default; uses local JSON fallback)
GSPREAD_OK = True
try:
    import gspread
    from google.oauth2 import service_account
except Exception:
    GSPREAD_OK = False

st.set_page_config(page_title="Investment – Collated Final (with ARIMA Insights)", layout="wide")
st.title("📈 Investment Dashboard – Collated Final (with ARIMA Insights)")

DEFAULT_BASE_CURRENCY = "AUD"
FX_TICKER_AUDUSD = "AUDUSD=X"
GOLD_TICKER_USD = "GC=F"
BTC_TICKER_USD = "BTC-USD"

def fmt_money(x, cur="AUD"):
    try:
        return f"{cur} {x:,.2f}"
    except Exception:
        return f"{cur} {x}"

def safe_yf_download(ticker, period="1y", interval="1d"):
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, threads=False, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
        return None
    except Exception:
        return None

def get_last_price(ticker: str) -> t.Tuple[float, str]:
    df = safe_yf_download(ticker, period="5d", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return (float("nan"), "no-data")
    close_series = pd.to_numeric(to_series_close(df["Close"]), errors="coerce").dropna()
    if close_series.empty:
        return (float("nan"), "no-data")
    last = float(close_series.iloc[-1])
    try:
        ts = df.index[-1].to_pydatetime().strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts = "unknown-ts"
    return (last, ts)

def drift_adjust(price: float, hours_since: float, drift_bps_per_hour: float = 2.0) -> float:
    if not np.isfinite(price) or not np.isfinite(hours_since):
        return price
    return price * (1.0 + (drift_bps_per_hour / 10000.0) * hours_since)

@st.cache_data(show_spinner=False)
def load_local_json():
    try:
        with open("holdings.json", "r") as f:
            return json.load(f)
    except Exception:
        try:
            with open("example_holdings.json", "r") as f:
                return json.load(f)
        except Exception:
            return {"base_currency": DEFAULT_BASE_CURRENCY, "targets": {}, "positions": []}

def save_local_json(data: dict):
    try:
        with open("holdings.json", "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def get_gsheet_client():
    if not GSPREAD_OK:
        return None
    try:
        secrets = st.secrets.get("gcp_service_account", None)
        sheet_key = st.secrets.get("sheet_id", None)
        if not secrets or not sheet_key:
            return None
        creds = service_account.Credentials.from_service_account_info(dict(secrets), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        return gc, sheet_key
    except Exception:
        return None

def read_from_sheet():
    cli = get_gsheet_client()
    if not cli:
        return None
    gc, sheet_key = cli
    try:
        sh = gc.open_by_key(sheet_key)
        ws = sh.worksheet("holdings")
    except Exception:
        return None
    rows = ws.get_all_records()
    pos = []
    targets = {}
    base = DEFAULT_BASE_CURRENCY
    for r in rows:
        tck = str(r.get("Ticker","")).strip()
        if not tck:
            continue
        qty = float(r.get("Quantity", 0) or 0)
        cost = float(r.get("CostBasis_AUD", 0) or 0)
        tgt = r.get("TargetWeight", None)
        typ = r.get("Type", "")
        if tgt is not None and tgt != "":
            targets[tck] = float(tgt)
        pos.append({"ticker": tck, "qty": qty, "cost_aud": cost, "type": typ})
    return {"base_currency": base, "targets": targets, "positions": pos}

def write_to_sheet(data: dict):
    cli = get_gsheet_client()
    if not cli:
        return False
    gc, sheet_key = cli
    try:
        sh = gc.open_by_key(sheet_key)
        try:
            ws = sh.worksheet("holdings")
        except Exception:
            ws = sh.add_worksheet(title="holdings", rows=200, cols=10)
        rows = [["Ticker","Quantity","CostBasis_AUD","TargetWeight","Type"]]
        for p in data.get("positions", []):
            rows.append([p.get("ticker",""), p.get("qty",0), p.get("cost_aud",0), data.get("targets",{}).get(p.get("ticker",""), ""), p.get("type","")])
        ws.clear()
        ws.update("A1", rows)
        return True
    except Exception:
        return False

# Sidebar
st.sidebar.header("Data Source")
use_sheets = st.sidebar.checkbox("Use Google Sheets (if configured in secrets)", value=False)
if st.sidebar.button("Load Holdings"):
    data = read_from_sheet() if use_sheets else load_local_json()
    if not data:
        st.error("Failed to load holdings from the selected source.")
    else:
        st.session_state["holdings"] = data
        st.success("Holdings loaded.")
if st.sidebar.button("Save Holdings"):
    data = st.session_state.get("holdings", None)
    if not data:
        st.warning("Nothing to save — load or edit first.")
    else:
        ok = write_to_sheet(data) if use_sheets else save_local_json(data)
        st.success("Saved." if ok else "Save failed.")

holdings = st.session_state.get("holdings", load_local_json())

# Editors
st.subheader("🎯 Targets & Positions")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown("**Target Allocation (by ticker, %)**")
    targets = holdings.get("targets", {})
    editable_targets = [{"Ticker": k, "Target %": v} for k, v in targets.items()]
    tgt_df = st.data_editor(pd.DataFrame(editable_targets), num_rows="dynamic", use_container_width=True)
    new_targets = {row["Ticker"]: float(row["Target %"]) for _, row in tgt_df.dropna().iterrows() if str(row["Ticker"]).strip() != ""}
    s = sum(new_targets.values()) or 1.0
    new_targets = {k: v/s*100.0 for k, v in new_targets.items()}
with c2:
    st.markdown("**Positions**")
    pos = holdings.get("positions", [])
    pos_df = st.data_editor(pd.DataFrame(pos), num_rows="dynamic", use_container_width=True)
    new_positions = pos_df.fillna({"qty":0,"cost_aud":0,"type":""}).to_dict(orient="records")
holdings["targets"] = new_targets
holdings["positions"] = new_positions
st.session_state["holdings"] = holdings

# Live prices & valuation
st.subheader("💹 Live Prices & Valuation")
tickers = [p["ticker"] for p in new_positions if str(p.get("ticker","")).strip()]
unique_tickers = sorted(set([t for t in tickers if t]))
prices, timestamps = {}, {}

col_fx, col_opts = st.columns([3,2])
# FX & reference proxies
fx_price, fx_ts = get_last_price(FX_TICKER_AUDUSD)
gold_usd, gold_ts = get_last_price(GOLD_TICKER_USD)
btc_usd, btc_ts = get_last_price(BTC_TICKER_USD)
with col_fx:
    st.write(f"**AUDUSD** last: {fx_price if np.isfinite(fx_price) else 'n/a'} @ {fx_ts}")
    audusd = fx_price if np.isfinite(fx_price) and fx_price>0 else float('nan')
    gold_aud = gold_usd / audusd if np.isfinite(gold_usd) and np.isfinite(audusd) else float('nan')
    btc_aud = btc_usd / audusd if np.isfinite(btc_usd) and np.isfinite(audusd) else float('nan')
    st.write(f"**Gold USD**: {gold_usd if np.isfinite(gold_usd) else 'n/a'} | **Gold AUD est**: {gold_aud if np.isfinite(gold_aud) else 'n/a'}")
    st.write(f"**BTC USD**: {btc_usd if np.isfinite(btc_usd) else 'n/a'} | **BTC AUD est**: {btc_aud if np.isfinite(btc_aud) else 'n/a'}")
with col_opts:
    drift_on = st.checkbox("Apply drift model to stale quotes", value=True)
    drift_bps_per_hour = st.number_input("Drift (bps/hour)", 0.0, 20.0, 2.0, step=0.5)
    fee_bps = st.number_input("Trading fee (bps)", 0.0, 200.0, 7.0, step=1.0)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 200.0, 5.0, step=1.0)

for tck in unique_tickers:
    px, ts = get_last_price(tck)
    if drift_on and isinstance(ts, str) and "UTC" in ts and np.isfinite(px):
        try:
            ts_dt = datetime.strptime(ts.replace(" UTC",""), "%Y-%m-%d %H:%M")
            hours = (datetime.utcnow() - ts_dt).total_seconds()/3600.0
            px = drift_adjust(px, hours, drift_bps_per_hour)
        except Exception:
            pass
    prices[tck], timestamps[tck] = px, ts

rows = []
for p in new_positions:
    tck = p.get("ticker","").strip()
    if not tck:
        continue
    qty = float(p.get("qty",0) or 0)
    cost = float(p.get("cost_aud",0) or 0)
    typ = p.get("type","")
    px = prices.get(tck, float('nan'))
    mkt = qty * (px if np.isfinite(px) else 0.0)
    rows.append({"Ticker": tck, "Type": typ, "Qty": qty, "Price": px, "MarketValue": mkt, "Cost_AUD": cost})
val_df = pd.DataFrame(rows)
total_mv = float(val_df["MarketValue"].sum()) if not val_df.empty else 0.0
st.dataframe(val_df.fillna("n/a"), use_container_width=True)
st.metric("Portfolio Market Value", fmt_money(total_mv))

# Allocation
st.subheader("🧭 Allocation")
if not val_df.empty:
    alloc = val_df.groupby("Ticker")["MarketValue"].sum()
    alloc_pct = (alloc / max(total_mv, 1)) * 100.0
    alloc_df = pd.DataFrame({"Ticker": alloc.index, "Weight %": alloc_pct.values}).sort_values("Weight %", ascending=False)
    st.dataframe(alloc_df, use_container_width=True)
    fig, ax = plt.subplots()
    ax.pie(alloc_pct.values, labels=alloc_pct.index, autopct="%1.1f%%")
    ax.set_title("Portfolio Allocation")
    st.pyplot(fig)
else:
    alloc_pct = pd.Series(dtype=float)

# Rebalance
st.subheader("🔧 Rebalance Advisor")
tgt_series = pd.Series(new_targets, dtype=float)
cur_series = alloc_pct if 'alloc_pct' in locals() else pd.Series(dtype=float)
combined = pd.DataFrame({"Target %": tgt_series, "Current %": cur_series}).fillna(0.0)
combined["Diff %"] = combined["Target %"] - combined["Current %"]
st.dataframe(combined.sort_values("Diff %", ascending=True), use_container_width=True)

# Greedy Planner
st.subheader("🧮 Greedy Planner (fees, slippage, lot sizes, FX)")
cash_aud = st.number_input("Available cash (AUD)", 0.0, 1e9, 10000.0, step=100.0)
lot_size = st.number_input("Lot size (min units per trade)", 1, 10000, 1, step=1)
if st.button("Plan Buys"):
    desired_weights = combined["Target %"] / combined["Target %"].sum()
    desired_values = desired_weights * (total_mv + cash_aud)
    current_values = (cur_series / 100.0) * total_mv
    gap = (desired_values - current_values).fillna(0.0)
    plan = []
    prices_vec = {t: prices.get(t, float('nan')) for t in desired_weights.index}
    fees = fee_bps / 10000.0
    slip = slippage_bps / 10000.0
    remaining_cash = cash_aud
    safety = 20000
    while remaining_cash > 0 and safety > 0:
        safety -= 1
        if gap.empty or gap.max() <= 0:
            break
        tgt = gap.sort_values(ascending=False).index[0]
        px = prices_vec.get(tgt, float('nan'))
        if not np.isfinite(px) or px <= 0:
            gap[tgt] = 0
            continue
        unit_cost = px * (1 + fees + slip)
        qty = max(0, int(min(remaining_cash // unit_cost, math.ceil((gap[tgt] / unit_cost)))))
        qty = (qty // lot_size) * lot_size
        if qty <= 0:
            break
        spend = qty * unit_cost
        remaining_cash -= spend
        gap[tgt] -= spend
        plan.append({"Ticker": tgt, "Qty": qty, "EstPrice": px, "EstSpend": spend})
    plan_df = pd.DataFrame(plan)
    if plan_df.empty:
        st.info("No feasible buys with current cash/lot size/fees.")
    else:
        st.dataframe(plan_df, use_container_width=True)
        st.metric("Cash leftover", fmt_money(remaining_cash))
        csv_buf = io.StringIO(); plan_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download plan CSV", data=csv_buf.getvalue(), file_name="buy_plan.csv", mime="text/csv")

# Alerts
st.subheader("🚨 Alerts")
alert_rows = []
ma_window = st.number_input("MA window (days)", 5, 200, 20)
for tck in unique_tickers:
    df = safe_yf_download(tck, period="90d", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        continue
    last_close = pd.to_numeric(to_series_close(df["Close"]), errors="coerce").dropna()
    if last_close.empty:
        continue
    last = float(last_close.iloc[-1])
    ma = moving_average_forgiving(df["Close"], ma_window)
    cur_w = float(alloc_pct.get(tck, 0.0)) if 'alloc_pct' in locals() else 0.0
    tgt_w = new_targets.get(tck, 0.0)
    above = (last > ma) if np.isfinite(ma) else None
    alert_rows.append({"Ticker": tck, "Last": last, f"MA{ma_window}": ma, "AboveMA": above, "AllocDrift %": cur_w - tgt_w})
if alert_rows:
    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True)

# Prediction & Insights (with ARIMA)
st.subheader("🔮 Prediction & Insights (with ARIMA)")
colA, colB = st.columns([2, 3])
with colA:
    horizon = st.number_input("Forecast horizon (days)", 7, 180, 45, step=1)
    arima_p = st.number_input("ARIMA p", 0, 5, 1)
    arima_d = st.number_input("ARIMA d", 0, 2, 1)
    arima_q = st.number_input("ARIMA q", 0, 5, 1)
    cap_move = st.number_input("Max tilt vs Target (pp)", 1.0, 25.0, 7.5, step=0.5)
    tickers_to_analyse = st.multiselect("Tickers to analyse", options=unique_tickers, default=unique_tickers)

exp_returns = {}
vols = {}

with colB:
    for tck in tickers_to_analyse:
        df_hist = safe_yf_download(tck, period="1y", interval="1d")
        if df_hist is None or df_hist.empty or "Close" not in df_hist.columns:
            st.write(f"**{tck}** — no data")
            continue
        close_series = pd.to_numeric(to_series_close(df_hist["Close"]), errors="coerce").dropna()
        if close_series.empty:
            st.write(f"**{tck}** — no close prices")
            continue
        mas = moving_averages(close_series, windows=(7,30,90))
        vol = volatility_annualised(close_series)
        sr = sharpe_ratio(close_series)
        fc_df, er, model_used = best_effort_forecast(close_series, horizon_days=int(horizon), order=(int(arima_p), int(arima_d), int(arima_q)))
        exp_returns[tck] = er
        vols[tck] = vol
        st.markdown(f"**{tck}**  ·  Model: **{model_used}**  ·  Vol (ann): {vol:.2%}  ·  Sharpe: {sr:.2f}  ·  {int(horizon)}d exp: {er:.2f}%")
        fig2, ax2 = plt.subplots()
        ax2.plot(close_series.index, close_series.values, label="Close")
        if fc_df is not None and not fc_df.empty:
            ax2.plot(fc_df["date"], fc_df["forecast"], label=f"{model_used} forecast")
            ax2.fill_between(fc_df["date"], fc_df["lower"], fc_df["upper"], alpha=0.2)
        if "MA30" in mas:
            ax2.plot(mas["MA30"].index, mas["MA30"].values, label="MA30", linestyle="--")
        if "MA90" in mas:
            ax2.plot(mas["MA90"].index, mas["MA90"].values, label="MA90", linestyle=":")
        ax2.set_xlabel("Date"); ax2.set_ylabel("Price"); ax2.legend()
        st.pyplot(fig2)

# Optimal recommendations
if not alloc_pct.empty:
    cur_w = alloc_pct
else:
    cur_w = pd.Series({t: 0.0 for t in new_targets.keys()})
tgt_w = pd.Series(new_targets, dtype=float)
rec_df = recommend_allocation(current_weights=cur_w, targets=tgt_w, exp_returns=exp_returns, vols=vols, cap_move=float(cap_move))
st.subheader("✅ Optimal Recommendations")
st.dataframe(rec_df, use_container_width=True)
buf = io.StringIO(); rec_df.to_csv(buf, index=True)
st.download_button("⬇️ Download recommendations (CSV)", data=buf.getvalue(), file_name="recommendations_with_arima.csv", mime="text/csv")

st.caption("This app provides educational analytics only and is not financial advice.")
