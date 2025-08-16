
# insights_helpers.py
import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Series(dtype=float, index=series.index)
    e = s.ewm(span=span, adjust=False).mean()
    return e.reindex(series.index)

def moving_averages(series: pd.Series, windows=(7, 30, 90)) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = {}
    for w in windows:
        out[f"MA{w}"] = s.rolling(w).mean().reindex(series.index)
    return out

def daily_returns(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.pct_change().dropna()

def volatility_annualised(series: pd.Series, trading_days: int = 252) -> float:
    r = daily_returns(series)
    if r.empty:
        return float("nan")
    return float(r.std() * np.sqrt(trading_days))

def sharpe_ratio(series: pd.Series, rf_daily: float = 0.0, trading_days: int = 252) -> float:
    r = daily_returns(series)
    if r.empty:
        return float("nan")
    excess = r - rf_daily
    if excess.std() == 0 or np.isnan(excess.std()):
        return float("nan")
    sr_daily = excess.mean() / excess.std()
    return float(sr_daily * np.sqrt(trading_days))

def lin_forecast(series: pd.Series, horizon_days: int = 30):
    y = pd.to_numeric(series, errors="coerce").dropna()
    if len(y) < 10:
        return None, float("nan")
    import numpy as np
    x = np.arange(len(y))
    a, b = np.polyfit(x, y.values, 1)
    y_hat = a * x + b
    resid = y.values - y_hat
    sigma = resid.std() if len(resid) > 1 else 0.0
    x_future = np.arange(len(y), len(y) + horizon_days)
    f = a * x_future + b
    lower = f - 1.28 * sigma
    upper = f + 1.28 * sigma
    last_price = float(y.iloc[-1])
    end_price = float(f[-1])
    exp_return = (end_price / last_price - 1.0) * 100.0 if last_price > 0 else float("nan")
    dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    out = pd.DataFrame({"date": dates, "forecast": f, "lower": lower, "upper": upper})
    return out, float(exp_return)

def recommend_allocation(current_weights: pd.Series, targets: pd.Series, exp_returns: dict, vols: dict, cap_move: float = 7.5):
    idx = sorted(set(list(current_weights.index) + list(targets.index) + list(exp_returns.keys())))
    base = targets.reindex(idx).fillna(0.0)
    cur = current_weights.reindex(idx).fillna(0.0)
    scores = {}
    for t in idx:
        er = exp_returns.get(t, np.nan)
        v = vols.get(t, np.nan)
        if v and np.isfinite(v) and v > 0 and np.isfinite(er):
            scores[t] = er / (v * 100.0)
        else:
            scores[t] = 0.0
    svals = np.array(list(scores.values()), dtype=float)
    if np.all(np.isnan(svals)) or np.all(svals == 0):
        suggested = base
    else:
        if np.nanmax(svals) != np.nanmin(svals):
            s_norm = (svals - np.nanmean(svals)) / (np.nanstd(svals) if np.nanstd(svals) > 0 else 1.0)
        else:
            s_norm = np.zeros_like(svals)
        tilts = np.clip(s_norm, -2.0, 2.0) / 2.0 * cap_move
        tilt_series = pd.Series(tilts, index=idx)
        suggested = base.add(tilt_series, fill_value=0.0)
    tot = suggested.sum() or 1.0
    suggested = suggested / tot * 100.0
    out = pd.DataFrame({
        "Current %": cur,
        "Target %": base,
        "Suggested %": suggested
    }).fillna(0.0)
    out["Change vs Target (pp)"] = out["Suggested %"] - out["Target %"]
    out["Change vs Current (pp)"] = out["Suggested %"] - out["Current %"]
    return out.sort_values("Change vs Target (pp)", ascending=False)
