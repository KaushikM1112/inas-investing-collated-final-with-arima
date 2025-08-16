
# hotfix_ma.py
import pandas as pd
import numpy as np

def to_series_close(close_like):
    if isinstance(close_like, pd.Series):
        return close_like
    if isinstance(close_like, pd.DataFrame):
        try:
            if isinstance(close_like.columns, pd.MultiIndex):
                for col in close_like.columns:
                    s = close_like[col]
                    if isinstance(s, pd.Series):
                        return s
                    if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                        return s.iloc[:, 0]
            if close_like.shape[1] == 1:
                return close_like.iloc[:, 0]
            for c in close_like.columns:
                if pd.api.types.is_numeric_dtype(close_like[c]):
                    return close_like[c]
            return pd.Series(dtype=float)
        except Exception:
            return pd.Series(dtype=float)
    try:
        return pd.Series(close_like)
    except Exception:
        return pd.Series(dtype=float)

def moving_average_forgiving(close_like, window: int = 20) -> float:
    s = to_series_close(close_like)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return float('nan')
    if len(s) < window:
        return float(s.mean())
    return float(s.tail(window).mean())
