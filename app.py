import os
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from data import load_macro
import engine as E

st.set_page_config("FICC Liquidity Shock Simulator", layout="wide")
st.title("FICC Liquidity Shock Simulator & Dynamic Hedging Dashboard")


@st.cache_data(show_spinner="Running ETL...")
def get_macro(key):
    return load_macro(key or None)


@st.cache_resource(show_spinner="Training ensemble...")
def get_model(_m, tag):
    return E.train(_m)


# ---------------- Sidebar: scenario injection ----------------
sb = st.sidebar
key = sb.text_input("FRED API key (optional)", os.getenv("FRED_API_KEY", ""), type="password")
sb.header("Stress scenario")
dvix = sb.slider("VIX shock (pts)", 0, 50, 20)
drate = sb.slider("Rate hike (bps)", 0, 300, 100, 25)
dhy = sb.slider("HY spread shock (bps)", 0, 600, 200, 25)
sb.header("Funding profile")
buf = sb.number_input("Cash buffer ($mm)", 50, 5000, 400, 50)
red = sb.slider("Daily redemptions / outflows (% of AUM)", 0.0, 2.0, 0.25, 0.05)
mpct = sb.slider("Margin call as % of MTM loss", 0, 100, 50, 5)
thr = sb.slider("Liquidity score alert threshold", 10, 90, 50)
part = sb.slider("Max market participation", 0.02, 0.30, 0.10, 0.01)

macro, src = get_macro(key)
model, auc, cols = get_model(macro, f"{src}-{len(macro)}")
sb.caption(f"Data: {src}  \nThrough {macro.index[-1].date()}  \nOOS AUC: {auc:.2f}" if not np.isnan(auc) else f"Data: {src}")

with st.expander("Portfolio & stress-sensitivity assumptions (editable)"):
    assets = st.data_editor(E.ASSETS, hide_index=True, use_container_width=True)

f = E.features(macro); last = f.iloc[-1]
row = E.shocked_row(last, dvix, drate, dhy)
p0, p1 = E.p_stress(model, last), E.p_stress(model, row)
base = E.asset_metrics(assets, last.vix, last.hy, 0, p0, part)
shk = E.asset_metrics(assets, row.vix, row.hy, drate, p1, part)
buf0 = E.buffer_metrics(assets, buf, 0, red, mpct, p0)
buf1 = E.buffer_metrics(assets, buf, drate, red, mpct, p1)

c = st.columns(5)
c[0].metric("P(liquidity stress, 20d)", f"{p1:.0%}", f"{p1 - p0:+.0%}", delta_color="inverse")
c[1].metric("Avg spread widening", f"{shk.spread_x.mean():.1f}x", f"{shk.spread_x.mean() - base.spread_x.mean():+.1f}x", delta_color="inverse")
c[2].metric("Portfolio liquidity horizon", f"{(shk.pos.sum() / shk.cap.sum()):.1f} d",
            f"{(shk.pos.sum() / shk.cap.sum()) - (base.pos.sum() / base.cap.sum()):+.1f} d", delta_color="inverse")
c[3].metric("Margin call", f"${buf1['margin']:.0f}mm")
c[4].metric("Buffer depletion", f"{buf1['days']:.1f} d" if np.isfinite(buf1['days']) else "n/a",
            f"{buf1['days'] - buf0['days']:+.1f} d" if np.isfinite(buf1['days']) else None, delta_color="normal")

t1, t2, t3, t4 = st.tabs(["Macro & forecast", "Liquidity heatmap", "Asset metrics", "AI recommendations"])

with t1:
    w = macro.tail(750)
    fig = go.Figure()
    for col, nm in [("VIX", "VIX"), ("HY_OAS", "HY OAS (%)")]:
        fig.add_scatter(x=w.index, y=w[col], name=nm)
        fut = pd.bdate_range(w.index[-1], periods=21)[1:]
        fig.add_scatter(x=fut, y=E.ar1(macro[col]), name=f"{nm} AR(1) forecast", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.line(w[["UST2Y", "UST10Y"]], title="US Treasury yields (%)"), use_container_width=True)

with t2:
    shocks = [0, 10, 20, 30, 40, 50]
    grid = {}
    for s in shocks:
        r = E.shocked_row(last, s, drate, dhy)
        grid[f"VIX +{s}"] = E.asset_metrics(assets, r.vix, r.hy, drate, E.p_stress(model, r), part).set_index("name").score
    hm = pd.DataFrame(grid)
    fg = px.imshow(hm, text_auto=".0f", color_continuous_scale="RdYlGn", zmin=0, zmax=100, aspect="auto",
                   title="Liquidity score (100 = easy to exit) by asset and VIX shock, at current rate/HY shocks")
    st.plotly_chart(fg, use_container_width=True)

with t3:
    show = shk[["name", "pos", "spread", "spread_bps", "spread_x", "horizon_d", "score"]].copy()
    show["base_horizon_d"] = base.horizon_d
    st.dataframe(show.round(2).rename(columns={"spread": "base_spread_bps", "spread_bps": "stressed_spread_bps"}),
                 hide_index=True, use_container_width=True)

with t4:
    weak = shk[shk.score < thr].sort_values("score")
    if buf1["days"] < 20: st.error(f"Buffer exhausted in {buf1['days']:.1f} days under this scenario. Start raising liquidity now.")
    if weak.empty: st.success("No asset below the liquidity threshold.")
    for _, a in weak.iterrows():
        hedge = "hedge duration (Treasury futures / payer swaps)" if a.dur > 0 else "hedge via options or futures overlay"
        st.warning(f"**{a['name']}**: score {a.score:.0f}, exit horizon {a.horizon_d:.1f}d, spread {a.spread_bps:.1f}bps "
                   f"({a.spread_x:.1f}x). Begin staged reduction now, or {hedge} instead of selling into the widening.")

    if st.button("Run PPO liquidation optimizer", type="primary"):
        cost = (shk.spread_bps / 2).values
        env = E.LiqEnv(shk.pos, shk.cap, cost, buf1["cash0"], buf1["burn"])
        try:
            with st.spinner("Training PPO agent..."):
                pol = E.train_ppo(env); label = "PPO"
        except Exception as e:
            pol, label = E.greedy, "Greedy (PPO unavailable)"; st.info(f"stable-baselines3 not available ({type(e).__name__}); using greedy.")
        names = list(shk.name); res = {}
        for nm, p in [(label, pol), ("Greedy cheapest-first", E.greedy), ("Pro-rata", E.prorata)]:
            sched, summ = E.rollout(env, p, names); res[nm] = (sched, summ)
        st.subheader("Policy comparison (20-day horizon, $mm)")
        st.dataframe(pd.DataFrame({k: v[1] for k, v in res.items()}).T, use_container_width=True)
        sched = res[label][0]
        st.subheader(f"{label} liquidation schedule ($mm sold per day)")
        st.plotly_chart(px.bar(sched.drop(columns="cash_mm"), barmode="stack"), use_container_width=True)
        top = sched.drop(columns="cash_mm").iloc[0].sort_values(ascending=False)
        st.info("Day-1 action: " + ", ".join(f"sell ${v:.0f}mm {k}" for k, v in top.items() if v > 0) if top.sum() > 0
                else "Day-1 action: hold; buffer covers near-term outflows, keep hedges in place.")
    st.caption("Illustrative model for research/portfolio purposes. Not investment advice; parameters are assumptions, not calibrated to any firm's book.")
