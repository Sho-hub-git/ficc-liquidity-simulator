"""Prediction engine, liquidity metrics, and RL (PPO) liquidation optimizer."""
import numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Illustrative book. pos/adv in $mm; spread in bps; dur in years; b_* = stress sensitivities (editable in the UI).
ASSETS = pd.DataFrame([
    ("UST Long (TLT)", "TLT", 800, 2500, 1.0, 17.0, 0.05, 0.10, 0.30),
    ("UST 7-10Y (IEF)", "IEF", 600, 1200, 1.0, 7.5, 0.05, 0.10, 0.20),
    ("IG Credit (LQD)", "LQD", 500, 1800, 3.0, 8.5, 0.12, 0.40, 0.25),
    ("HY Credit (HYG)", "HYG", 300, 1500, 6.0, 3.5, 0.25, 1.00, 0.30),
    ("EM Debt (EMB)", "EMB", 200, 400, 8.0, 7.0, 0.22, 0.80, 0.30),
    ("USD / FX (UUP)", "UUP", 250, 400, 2.0, 0.0, 0.06, 0.15, 0.10),
    ("Gold (GLD)", "GLD", 200, 3000, 2.0, 0.0, 0.04, 0.05, 0.05),
    ("Crude Oil (USO)", "USO", 150, 300, 5.0, 0.0, 0.15, 0.30, 0.10)],
    columns=["name", "ticker", "pos", "adv", "spread", "dur", "b_vix", "b_hy", "b_rate"])


# ---------- 1. Features, labels, ensemble classifier ----------
def features(m):
    f = pd.DataFrame({"vix": m.VIX, "hy": m.HY_OAS * 100, "curve": (m.UST10Y - m.UST2Y) * 100,
                      "d10": m.UST10Y.diff(20) * 100, "dvix": m.VIX.diff(20), "dhy": m.HY_OAS.diff(20) * 100})
    return f.dropna()


def labels(m, f, h=20):
    """1 if, within the next h days, VIX exceeds 30 or HY OAS widens >75bps (liquidity-deterioration proxy)."""
    fv = m.VIX[::-1].rolling(h).max()[::-1].shift(-1)
    fh = (m.HY_OAS.shift(-h) - m.HY_OAS) * 100
    y = ((fv > 30) | (fh > 75)).astype(float)
    y[fv.isna() | fh.isna()] = np.nan
    return y.reindex(f.index)


def _ensemble():
    return VotingClassifier([
        ("rf", RandomForestClassifier(200, min_samples_leaf=20, random_state=0, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(random_state=0)),
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)))], voting="soft")


def train(m):
    """Returns (fitted model, out-of-sample AUC on last 25% of history, feature names)."""
    f = features(m); d = f.join(labels(m, f).rename("y")).dropna(); X, y = d[f.columns], d["y"]
    k = int(len(d) * 0.75); auc = np.nan
    if y[:k].nunique() > 1 and y[k:].nunique() > 1:
        auc = roc_auc_score(y[k:], _ensemble().fit(X[:k], y[:k]).predict_proba(X[k:])[:, 1])
    return _ensemble().fit(X, y), auc, list(f.columns)


def shocked_row(last, dvix=0, drate_bps=0, dhy_bps=0):
    r = last.copy()
    r["vix"] += dvix; r["dvix"] += dvix
    r["hy"] += dhy_bps; r["dhy"] += dhy_bps
    r["d10"] += 0.6 * drate_bps; r["curve"] -= 0.4 * drate_bps  # front-end moves more than long-end (flattening)
    return r


def p_stress(model, row):
    return float(model.predict_proba(pd.DataFrame([row]))[0][1])


def ar1(s, h=20):
    """Mean-reverting AR(1) trend forecast of a macro series."""
    x = s.dropna().values[-750:]; b, a = np.polyfit(x[:-1], x[1:], 1); out = [x[-1]]
    for _ in range(h): out.append(a + b * out[-1])
    return np.array(out[1:])


# ---------- 2. Liquidity metrics ----------
def asset_metrics(assets, vix, hy_bps, rate_bps, p, part=0.10):
    a = assets.copy()
    mult = (1 + a.b_vix * max(vix - 15, 0) + a.b_hy * max(hy_bps - 350, 0) / 100
            + a.b_rate * max(rate_bps, 0) / 100) * (1 + 1.5 * p)
    a["spread_x"] = mult; a["spread_bps"] = a.spread * mult
    a["adv_stress"] = a.adv / (1 + 0.5 * (mult - 1))            # volume dries up as spreads widen
    a["cap"] = a.adv_stress * part                              # $mm/day sellable at `part` participation
    a["horizon_d"] = a.pos / a.cap                              # liquidity horizon (days)
    a["score"] = (100 * np.exp(-0.12 * a.horizon_d - 0.08 * (mult - 1))).clip(0, 100)
    return a


def buffer_metrics(assets, buffer_mm, rate_bps, red_pct, margin_pct, p):
    mtm = (assets.pos * assets.dur * abs(rate_bps) / 1e4).sum()        # $mm mark-to-market loss
    margin = mtm * margin_pct / 100                                    # variation margin call, day 0
    burn = assets.pos.sum() * red_pct / 100 * (1 + 2 * p)              # $mm/day outflow
    cash0 = max(buffer_mm - margin, 0)
    return dict(mtm=mtm, margin=margin, burn=burn, cash0=cash0, days=cash0 / burn if burn > 0 else np.inf)


# ---------- 3. RL liquidation optimizer (PPO) ----------
class LiqEnv(gym.Env):
    """Each day choose the fraction (0-1) of each asset's daily capacity to sell. Reward = -(fire-sale cost + 5x shortfall)."""
    def __init__(s, pos, cap, cost_bps, cash0, burn, T=20):
        s.pos, s.cap, s.cost = (np.asarray(x, float) for x in (pos, cap, cost_bps))
        s.cash0, s.burn, s.T, n = cash0, burn, T, len(pos)
        s.scale = max(burn, 1e-6)
        s.observation_space = spaces.Box(-50, 50, (n + 2,), np.float32)
        s.action_space = spaces.Box(0, 1, (n,), np.float32)

    def _obs(s):
        return np.r_[s.rem / s.pos, s.cash / (s.scale * s.T), s.t / s.T].astype(np.float32)

    def reset(s, seed=None, options=None):
        super().reset(seed=seed); s.rem, s.cash, s.t = s.pos.copy(), float(s.cash0), 0
        return s._obs(), {}

    def step(s, a):
        a = np.clip(a, 0, 1); sell = a * np.minimum(s.cap, s.rem)
        fee = sell * s.cost / 1e4 * (1 + a)                            # impact rises with selling intensity
        s.rem -= sell
        s.cash += (sell - fee).sum() - s.burn * (1 + 0.15 * s.np_random.standard_normal())
        short = max(-s.cash, 0); s.t += 1
        return s._obs(), float(-(fee.sum() + 5 * short) / s.scale), s.t >= s.T, False, {"sell": sell, "fee": fee.sum()}


def greedy(env, obs):
    """Baseline: sell cheapest-to-trade assets first to cover today's outflow."""
    need, a = env.burn, np.zeros(len(env.pos))
    for i in np.argsort(env.cost):
        c = min(env.cap[i], env.rem[i])
        if need > 0 and c > 0: x = min(c, need); a[i] = x / c; need -= x
    return a


def prorata(env, obs):
    room = np.minimum(env.cap, env.rem)
    return np.full(len(env.pos), min(1.0, env.burn / max(room.sum(), 1e-9)))


def train_ppo(env, steps=8000):
    from stable_baselines3 import PPO
    m = PPO("MlpPolicy", env, n_steps=512, batch_size=128, seed=0, verbose=0).learn(steps)
    return lambda e, o: m.predict(o, deterministic=True)[0]


def rollout(env, pol, names, seed=0):
    obs, _ = env.reset(seed=seed); rows, done, fees, low = [], False, 0.0, env.cash0
    while not done:
        obs, _, done, _, info = env.step(pol(env, obs))
        fees += info["fee"]; low = min(low, env.cash)
        rows.append({"day": env.t, **dict(zip(names, info["sell"].round(1))), "cash_mm": round(env.cash, 1)})
    return pd.DataFrame(rows).set_index("day"), dict(fire_sale_cost_mm=round(fees, 2), min_cash_mm=round(low, 1),
                                                     total_sold_mm=round(env.pos.sum() - env.rem.sum(), 1))
