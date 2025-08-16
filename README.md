
# Investment Dashboard – Collated Final (with ARIMA Insights)

Single-file Streamlit app that merges the collated dashboard, hotfixes, Prediction & Insights tab, and **ARIMA** forecasting (with linear fallback).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Google Sheets (optional)
Create `.streamlit/secrets.toml` with your service account + `sheet_id`, then enable in the sidebar.

## Notes
- ARIMA order defaults to (1,1,1); adjust in the **Prediction & Insights** panel.
- If ARIMA can't fit, app falls back to linear trend automatically.
- Educational purposes only; not financial advice.
