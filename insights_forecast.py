
# insights_forecast.py
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    _ARIMA_OK = True
except Exception:
    _ARIMA_OK = False

def _linear_forecast_core(y: pd.Series, horizon_days: int = 30):
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 10:
        return None, float("nan")
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

def arima_forecast(y: pd.Series, horizon_days: int = 30, order=(1,1,1)):
    if not _ARIMA_OK:
        return None, float("nan")
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < sum(order) + 10:
        return None, float("nan")
    try:
        model = ARIMA(y, order=order)
        fit = model.fit(method_kwargs={"warn_convergence": False})
        fc = fit.get_forecast(steps=horizon_days)
        mean = fc.predicted_mean
        conf = fc.conf_int(alpha=0.2)
        lower = conf.iloc[:, 0]
        upper = conf.iloc[:, 1]
        last_price = float(y.iloc[-1])
        end_price = float(mean.iloc[-1])
        exp_return = (end_price / last_price - 1.0) * 100.0 if last_price > 0 else float("nan")
        dates = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        out = pd.DataFrame({"date": dates, "forecast": mean.values, "lower": lower.values, "upper": upper.values})
        return out, float(exp_return)
    except Exception:
        return None, float("nan")

def best_effort_forecast(y: pd.Series, horizon_days: int = 30, order=(1,1,1)):
    fc, er = arima_forecast(y, horizon_days=horizon_days, order=order)
    if fc is not None:
        return fc, er, "ARIMA"
    fc, er = _linear_forecast_core(y, horizon_days=horizon_days)
    if fc is not None:
        return fc, er, "Linear"
    return None, float("nan"), "None"
