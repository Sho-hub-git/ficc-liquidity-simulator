"""Data layer: FRED macro series -> cleaned daily DataFrame. Falls back to synthetic data so the app runs offline."""
import numpy as np, pandas as pd, requests

FRED = {"DGS2": "UST2Y", "DGS10": "UST10Y", "VIXCLS": "VIX", "BAMLH0A0HYM2": "HY_OAS"}  # yields/OAS in %


def _fred(sid, key, start):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=20,
                     params=dict(series_id=sid, api_key=key, file_type="json", observation_start=start))
    r.raise_for_status()
    o = pd.DataFrame(r.json()["observations"])
    return pd.Series(pd.to_numeric(o.value, errors="coerce").values, index=pd.to_datetime(o.date), name=FRED[sid])


def _synthetic(start):
    idx = pd.bdate_range(start, pd.Timestamp.today()); n = len(idx); g = np.random.default_rng(7)
    v = np.empty(n); v[0] = 18
    for t in range(1, n):  # mean-reverting VIX with occasional jumps (crisis episodes)
        v[t] = max(9, v[t-1] + 0.08 * (18 - v[t-1]) + 1.1 * g.standard_normal() + (g.random() < 0.004) * g.uniform(8, 25))
    v = pd.Series(v).rolling(3, min_periods=1).mean().values
    hy = 3.0 + 0.15 * (v - 18) + 0.03 * g.standard_normal(n).cumsum() * 0.2
    u10 = np.clip(3 + np.cumsum(0.03 * g.standard_normal(n)), 0.5, 6)
    u2 = np.clip(u10 - 0.5 - 0.3 * np.sin(np.arange(n) / 400) + 0.02 * g.standard_normal(n), 0.1, 6)
    return pd.DataFrame({"UST2Y": u2, "UST10Y": u10, "VIX": v, "HY_OAS": np.clip(hy, 1.5, 20)}, index=idx)


def etl(df):
    """Business-day reindex -> interpolate short gaps -> forward-fill -> zero-fill any leading gaps."""
    df = df.reindex(pd.bdate_range(df.index.min(), df.index.max()))
    return df.interpolate(limit=5).ffill().fillna(0)


def load_macro(key=None, start="2007-01-01"):
    """Returns (DataFrame, source_label)."""
    if key:
        try:
            df = pd.concat([_fred(s, key, start) for s in FRED], axis=1)
            return etl(df), "FRED API (live)"
        except Exception as e:  # bad key / no network
            return etl(_synthetic(start)), f"Synthetic fallback (FRED failed: {type(e).__name__})"
    return etl(_synthetic(start)), "Synthetic demo data (add a FRED key for live data)"


def load_etf_adv(tickers, days=60):
    """Optional: average dollar volume ($mm) and 20d realised vol from Yahoo Finance. Returns DataFrame or None."""
    try:
        import yfinance as yf
        d = yf.download(list(tickers), period=f"{days}d", progress=False, auto_adjust=True)
        adv = (d["Close"] * d["Volume"]).mean() / 1e6
        vol = d["Close"].pct_change().tail(20).std() * np.sqrt(252) * 100
        return pd.DataFrame({"adv_mm": adv, "vol_pct": vol})
    except Exception:
        return None
