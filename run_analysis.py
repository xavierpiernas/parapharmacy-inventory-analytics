"""
Demand forecasting, replenishment simulation and customer segmentation
for a hypothetical Spanish subsidiary of an online parapharmacy.

Pipeline:
    1. Generate a reproducible synthetic dataset (catalog, weekly demand, campaigns,
       customers and orders).
    2. Forecast weekly demand per SKU with two baselines and Prophet (rolling origin).
    3. Simulate a periodic-review replenishment policy driven by each forecast.
    4. Segment customers with RFM + K-Means.
    5. Export data, metrics and figures to ./results.

Usage:
    python run_analysis.py               # full run (requires Prophet)
    python run_analysis.py --no-prophet  # quick test: Prophet replaced by a Fourier regression
"""
import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
OUTPUT_DIR = "results"
START_DATE = "2024-11-04"     # Monday; 104 weeks up to late October 2026
N_WEEKS = 104
TEST_WEEKS = 26               # last 26 weeks: evaluation and simulation
Z_SERVICE = 1.65              # ~95 % target service level, same for every policy
EXCESS_COVER_WEEKS = 12       # stock above 12 weeks of real demand counts as excess
HOLDING_COST_RATE = 0.20      # annual cost of holding inventory (% of cost value)
N_CUSTOMERS = 15000

rng = np.random.default_rng(SEED)
weeks = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-MON")
week_of_year = weeks.isocalendar().week.astype(int).values

# ---------------------------------------------------------------------------
# 1. Synthetic data
# ---------------------------------------------------------------------------
def build_campaigns():
    """Commercial calendar: (name, affected category, start, end, demand uplift)."""
    rows = []
    for year in (2024, 2025, 2026):
        calendar = [
            ("Black Friday", "all", f"{year}-11-20", f"{year}-11-30", 0.70),
            ("January sales", "all", f"{year}-01-05", f"{year}-01-19", 0.25),
            ("Sun care campaign", "dermocosmetics", f"{year}-05-26", f"{year}-06-09", 0.45),
            ("VAT-free days", "all", f"{year}-07-07", f"{year}-07-14", 0.35),
            ("Back to routine", "supplements", f"{year}-09-01", f"{year}-09-15", 0.40),
            ("Autumn immunity", "supplements", f"{year}-10-13", f"{year}-10-27", 0.30),
        ]
        for name, category, start, end, uplift in calendar:
            for w in weeks[(weeks >= pd.Timestamp(start)) & (weeks <= pd.Timestamp(end))]:
                rows.append({"week": w, "campaign": name, "category": category, "uplift": uplift})
    return pd.DataFrame(rows).reset_index(drop=True)


SUBCATEGORIES = (
    [("dermocosmetics", "sun_care")] * 12 + [("dermocosmetics", "facial")] * 10
    + [("dermocosmetics", "body")] * 8 + [("supplements", "vitamins_immunity")] * 10
    + [("supplements", "collagen_beauty")] * 8 + [("supplements", "probiotics")] * 6
    + [("supplements", "energy")] * 6
)


def build_catalog():
    rows = []
    for i, (category, subcategory) in enumerate(SUBCATEGORIES):
        turnover = rng.choice(["high", "medium", "low"], p=[0.3, 0.45, 0.25])
        # all three draws are made on purpose so the random sequence stays fixed
        base_options = {"high": rng.uniform(60, 140), "medium": rng.uniform(18, 45), "low": rng.uniform(2, 7)}
        price = round(rng.uniform(9, 22) if category == "dermocosmetics" else rng.uniform(12, 30), 2)
        rows.append({
            "sku": f"{'DER' if category == 'dermocosmetics' else 'SUP'}-{i + 1:03d}",
            "category": category,
            "subcategory": subcategory,
            "turnover": turnover,
            "base_demand": round(base_options[turnover], 1),
            "price": price,
            "cost": round(price * rng.uniform(0.55, 0.65), 2),
            "lead_time_weeks": int(rng.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])),
            "annual_trend": round(rng.normal(0.12, 0.08), 3),
        })
    return pd.DataFrame(rows)


def seasonality(subcategory, w):
    """Multiplicative seasonal factor by ISO week of the year."""
    angle = 2 * np.pi * (w - 1) / 52
    if subcategory == "sun_care":
        return np.clip(0.25 + 2.6 * np.exp(-((w - 28) ** 2) / (2 * 6.5 ** 2)), 0.2, None)
    if subcategory == "facial":
        return 1 + 0.15 * np.cos(angle)
    if subcategory == "body":
        return 1 + 0.20 * np.cos(angle - np.pi)
    if subcategory == "vitamins_immunity":
        return 1 + 0.45 * np.cos(angle)
    if subcategory == "collagen_beauty":
        return 1 + 0.15 * np.cos(angle - 2 * np.pi * 20 / 52)
    if subcategory == "probiotics":
        return 1 + 0.10 * np.cos(angle)
    if subcategory == "energy":
        return 1 + 0.25 * np.exp(-((w - 37) ** 2) / (2 * 3 ** 2)) + 0.2 * np.exp(-((w - 2) ** 2) / (2 * 3 ** 2))
    return np.ones_like(w, dtype=float)


def build_demand(catalog, campaigns):
    """Weekly units per SKU: trend x seasonality x campaigns, negative binomial noise."""
    rows = []
    t = np.arange(N_WEEKS) / 52
    for _, p in catalog.iterrows():
        level = p.base_demand * (1 + p.annual_trend) ** t * seasonality(p.subcategory, week_of_year)
        uplift = np.ones(N_WEEKS)
        for _, c in campaigns.iterrows():
            if c.category in ("all", p.category):
                uplift[weeks.get_loc(c.week)] += c.uplift * rng.uniform(0.7, 1.3)
        mu = level * uplift
        k = 8.0 if p.turnover != "low" else 2.0     # lower k = more dispersion
        units = rng.poisson(rng.gamma(k, mu / k))
        for j in range(N_WEEKS):
            rows.append({"week": weeks[j], "sku": p.sku, "units": int(units[j]), "campaign": int(uplift[j] > 1)})
    return pd.DataFrame(rows)


def build_customers_and_orders(catalog, demand):
    """Customers with different loyalty profiles; every unit of demand is assigned to an
    active customer, so orders add up exactly to the weekly demand per SKU."""
    types = rng.choice(["loyal", "recurring", "occasional", "new"], size=N_CUSTOMERS, p=[0.08, 0.22, 0.55, 0.15])
    signup = np.where(types == "new",
                      rng.integers(N_WEEKS - 26, N_WEEKS, N_CUSTOMERS),
                      rng.integers(0, N_WEEKS - 26, N_CUSTOMERS))
    weight = pd.Series(types).map({"loyal": 7, "recurring": 4, "occasional": 1, "new": 3}).values.astype(float)
    churn = np.where((types == "occasional") & (rng.random(N_CUSTOMERS) < 0.45),
                     signup + rng.integers(4, 40, N_CUSTOMERS), 10 ** 6)
    derm_affinity = rng.beta(2, 2, N_CUSTOMERS)
    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(N_CUSTOMERS)],
        "generation_type": types,
        "signup_week": weeks[np.minimum(signup, N_WEEKS - 1)],
        "acquisition_channel": rng.choice(["SEO", "SEM", "Social media", "Affiliates", "Direct"], N_CUSTOMERS,
                                          p=[0.3, 0.25, 0.2, 0.1, 0.15]),
    })

    price = catalog.set_index("sku").price
    category = catalog.set_index("sku").category
    ids = customers.customer_id.values
    lines = []
    for j, w in enumerate(weeks):
        active = np.where((signup <= j) & (churn > j))[0]
        if len(active) == 0:
            continue
        week_demand = demand[demand.week == w]
        for cat in ("dermocosmetics", "supplements"):
            affinity = derm_affinity[active] if cat == "dermocosmetics" else 1 - derm_affinity[active]
            p = weight[active] * affinity
            p = p / p.sum()
            for r in week_demand[week_demand.sku.map(category) == cat].itertuples():
                remaining = r.units
                if remaining == 0:
                    continue
                quantities = []
                while remaining > 0:
                    q = int(min(remaining, rng.choice([1, 1, 1, 2, 2, 3])))
                    quantities.append(q)
                    remaining -= q
                buyers = active[rng.choice(len(active), size=len(quantities), p=p)]
                for q, b in zip(quantities, buyers):
                    lines.append((w, ids[b], r.sku, q, round(q * price[r.sku], 2)))

    orders = pd.DataFrame(lines, columns=["week", "customer_id", "sku", "units", "amount"])
    orders["order_id"] = orders.groupby(["week", "customer_id"]).ngroup().map(lambda x: f"P{x:07d}")
    orders["date"] = orders.week + pd.to_timedelta(rng.integers(0, 7, len(orders)), unit="D")
    return customers, orders[["order_id", "date", "week", "customer_id", "sku", "units", "amount"]]


# ---------------------------------------------------------------------------
# 2. Forecasting (rolling origin, 4-week horizon)
# ---------------------------------------------------------------------------
def campaign_flag(campaigns, category):
    flag = np.zeros(N_WEEKS)
    affected = campaigns[campaigns.category.isin(["all", category])].week
    flag[[weeks.get_loc(w) for w in affected]] = 1
    return flag


def forecast_moving_average(y, origin, h):
    return np.repeat(y[origin - 4:origin].mean(), h)


def forecast_seasonal_naive(y, origin, h):
    return y[origin - 52:origin - 52 + h].astype(float)


def forecast_prophet(y, origin, h, flag):
    """Prophet on log(1 + units). Campaigns enter as a single regressor per category:
    with one or two years of history each campaign appears only once or twice, so
    individual effects cannot be estimated reliably."""
    from prophet import Prophet
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.ERROR)
    train = pd.DataFrame({"ds": weeks[:origin], "y": np.log1p(y[:origin]), "campaign": flag[:origin]})
    model = Prophet(yearly_seasonality=5, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode="additive", changepoint_prior_scale=0.01,
                    seasonality_prior_scale=5, uncertainty_samples=0)
    model.add_regressor("campaign", prior_scale=0.5)
    model.fit(train)
    future = pd.DataFrame({"ds": weeks[origin:origin + h], "campaign": flag[origin:origin + h]})
    return np.clip(np.expm1(model.predict(future).yhat.values), 0, None)


def forecast_fourier_regression(y, origin, h, flag):
    """Test-only stand-in for Prophet (same additive structure, fitted by ridge regression)."""
    from sklearn.linear_model import Ridge
    t = np.arange(N_WEEKS)
    features = [t / 52]
    for k in range(1, 6):
        features += [np.sin(2 * np.pi * k * week_of_year / 52), np.cos(2 * np.pi * k * week_of_year / 52)]
    X = np.column_stack(features + [flag])
    model = Ridge(alpha=1.0).fit(X[:origin], np.log1p(y[:origin]))
    return np.clip(np.expm1(model.predict(X[origin:origin + h])), 0, None)


def run_forecasts(catalog, demand, campaigns, use_prophet):
    """Origins every 4 weeks over calibration (26 weeks) and test (26 weeks) periods."""
    model_forecast = forecast_prophet if use_prophet else forecast_fourier_regression
    rows = []
    origins = range(N_WEEKS - 2 * TEST_WEEKS, N_WEEKS, 4)
    for n, (_, p) in enumerate(catalog.iterrows()):
        if use_prophet:
            print(f"   SKU {n + 1}/{len(catalog)}", end="\r")
        y = demand[demand.sku == p.sku].units.values
        flag = campaign_flag(campaigns, p.category)
        for origin in origins:
            h = min(4, N_WEEKS - origin)
            cap = 2 * y[:origin].max()      # plausibility limit: never above twice the historical peak
            ma = np.minimum(forecast_moving_average(y, origin, h), cap)
            sn = np.minimum(forecast_seasonal_naive(y, origin, h), cap)
            pr = np.minimum(model_forecast(y, origin, h, flag), cap)
            for k in range(h):
                rows.append({"week": weeks[origin + k], "sku": p.sku, "actual": y[origin + k],
                             "moving_average": ma[k], "seasonal_naive": sn[k], "prophet": pr[k]})
    return pd.DataFrame(rows)


def forecast_metrics(forecasts, method):
    test = forecasts[forecasts.week >= weeks[N_WEEKS - TEST_WEEKS]]
    error = (test.actual - test[method]).abs()
    return {"MAE": round(error.mean(), 2), "WAPE_%": round(100 * error.sum() / test.actual.sum(), 1)}


# ---------------------------------------------------------------------------
# 3. Replenishment simulation
# ---------------------------------------------------------------------------
def simulate(method, catalog, demand, forecasts):
    """Weekly periodic review, order-up-to level S = forecast(L + 1) + safety stock,
    with safety stock = z * sigma * sqrt(L + 1). Sigma is a robust estimate of the
    method's own forecast error in the calibration period."""
    rows = []
    start = N_WEEKS - TEST_WEEKS
    for _, p in catalog.iterrows():
        y = demand[demand.sku == p.sku].units.values
        L = p.lead_time_weeks
        f_sku = forecasts[forecasts.sku == p.sku]
        calibration = f_sku[f_sku.week < weeks[start]]
        sigma = max(1.25 * np.mean(np.abs(calibration.actual - calibration[method])), 0.5)
        f = f_sku[f_sku.week >= weeks[start]][method].values
        safety_stock = Z_SERVICE * sigma * np.sqrt(L + 1)
        stock = y[start - 8:start].mean() * (L + 2)
        pipeline = [0.0] * L
        for k in range(TEST_WEEKS):
            if L > 0:
                stock += pipeline.pop(0)
            d = y[start + k]
            served = min(stock, d)
            stock -= served
            horizon = f[k:k + L + 1]
            horizon = np.concatenate([horizon, np.repeat(f[-1], L + 1 - len(horizon))])
            order = max(0.0, horizon.sum() + safety_stock - (stock + sum(pipeline)))
            if L > 0:
                pipeline.append(order)
            else:
                stock += order
            next_weeks = y[start + k + 1:start + k + 1 + EXCESS_COVER_WEEKS]
            demand_12w = next_weeks.sum() * EXCESS_COVER_WEEKS / max(len(next_weeks), 1)
            rows.append({"method": method, "sku": p.sku, "week": weeks[start + k], "demand": d,
                         "served": served, "stockout": int(served < d), "ending_stock": stock,
                         "stock_value": stock * p.cost,
                         "excess_value": max(0.0, stock - demand_12w) * p.cost,
                         "lost_sales": (d - served) * p.price})
    return pd.DataFrame(rows)


def summarize_simulation(sim, n_skus):
    s = sim.groupby("method").agg(sku_weeks=("stockout", "size"), stockouts=("stockout", "sum"),
                                  demand=("demand", "sum"), served=("served", "sum"),
                                  stock_value=("stock_value", "mean"), excess_value=("excess_value", "mean"),
                                  lost_sales=("lost_sales", "sum"))
    s["stockout_rate_%"] = 100 * s.stockouts / s.sku_weeks
    s["fill_rate_%"] = 100 * s.served / s.demand
    s["avg_stock_value_total"] = s.stock_value * n_skus
    s["avg_excess_value_total"] = s.excess_value * n_skus
    return s.round(2)


# ---------------------------------------------------------------------------
# 4. Customer segmentation (RFM + K-Means)
# ---------------------------------------------------------------------------
def segment_customers(orders):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    cutoff = weeks[-1] + pd.Timedelta(days=7)
    last_year = orders[orders.date >= cutoff - pd.Timedelta(days=365)]
    rfm = last_year.groupby("customer_id").agg(last_purchase=("date", "max"), F=("order_id", "nunique"),
                                               M=("amount", "sum"))
    rfm["R"] = (cutoff - rfm.last_purchase).dt.days
    X = StandardScaler().fit_transform(np.column_stack([np.log1p(rfm.R), np.log1p(rfm.F), np.log1p(rfm.M)]))

    tests = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(5000, len(X)), random_state=SEED)
        tests.append({"k": k, "inertia": round(km.inertia_, 1), "silhouette": round(sil, 3)})
    tests = pd.DataFrame(tests)
    # k = 2 usually wins on silhouette but is not actionable; choose the best between 3 and 6
    k = int(tests[(tests.k >= 3) & (tests.k <= 6)].sort_values("silhouette", ascending=False).k.iloc[0])
    rfm["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X).labels_

    profile = rfm.groupby("cluster").agg(customers=("F", "size"), R_mean=("R", "mean"), F_mean=("F", "mean"),
                                         M_mean=("M", "mean"), M_total=("M", "sum")).round(1)
    profile["customers_%"] = (100 * profile.customers / profile.customers.sum()).round(1)
    profile["revenue_%"] = (100 * profile.M_total / profile.M_total.sum()).round(1)

    # name clusters from their centroids
    names, rest = {}, list(profile.index)
    c = profile.M_mean.idxmax(); names[c] = "High-value loyal"; rest.remove(c)
    c = profile.loc[rest].R_mean.idxmax(); names[c] = "At risk of churn"; rest.remove(c)
    for i, c in enumerate(profile.loc[rest].sort_values("F_mean").index):
        names[c] = ["New or infrequent", "Mid-value recurring", "High-value recurring"][min(i, 2)] \
            if len(rest) > 1 else "Mid-value recurring"
    profile["segment"] = profile.index.map(names)
    rfm["segment"] = rfm.cluster.map(names)
    return rfm.reset_index(), profile.reset_index(), tests, k


# ---------------------------------------------------------------------------
# 5. Figures (labels in Spanish: they are used in the thesis document)
# ---------------------------------------------------------------------------
SEGMENT_ES = {"High-value loyal": "Fieles de alto valor", "At risk of churn": "En riesgo de abandono",
              "New or infrequent": "Nuevos o poco frecuentes", "Mid-value recurring": "Recurrentes de valor medio",
              "High-value recurring": "Recurrentes de valor alto"}
CATEGORY_ES = {"dermocosmetics": "Dermocosmética", "supplements": "Complementos"}


def make_figures(catalog, demand, forecasts, metrics, sim, sim_summary, rfm, profile, tests, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    BLUE, GREY, ORANGE = "#1f4e79", "#9aa5b1", "#e07b39"
    fig_dir = f"{OUTPUT_DIR}/figures"
    test_start = weeks[N_WEEKS - TEST_WEEKS]

    # Fig. 1 - weekly demand by category
    d = demand.merge(catalog[["sku", "category"]]).groupby(["week", "category"]).units.sum().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d.index, d["dermocosmetics"], color=ORANGE, label="Dermocosmética")
    ax.plot(d.index, d["supplements"], color=BLUE, label="Complementos alimenticios")
    ax.set_ylabel("Unidades por semana"); ax.legend(frameon=False)
    ax.set_title("Demanda semanal sintética por categoría")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig1_demand_by_category.png", dpi=200); plt.close(fig)

    # Fig. 2 - actual vs forecast for the top sun care SKU
    sku = catalog[catalog.subcategory == "sun_care"].sort_values("base_demand", ascending=False).sku.iloc[0]
    hist = demand[demand.sku == sku]
    fc = forecasts[(forecasts.sku == sku) & (forecasts.week >= test_start)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(hist.week, hist.units, color=GREY, label="Demanda real")
    ax.plot(fc.week, fc.moving_average, color=BLUE, ls="--", label="Media móvil 4 semanas")
    ax.plot(fc.week, fc.prophet, color=ORANGE, label="Prophet")
    ax.axvline(test_start, color="k", lw=0.8)
    ax.text(test_start, ax.get_ylim()[1] * 0.92, "  periodo de test", fontsize=8)
    ax.set_title(f"Previsión frente a demanda real · {sku} (protección solar)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig2_forecast_example.png", dpi=200); plt.close(fig)

    # Fig. 3 - WAPE by method
    labels = {"moving_average": "Media móvil", "seasonal_naive": "Naive estacional", "prophet": "Prophet"}
    values = [metrics[m]["WAPE_%"] for m in labels]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(list(labels.values()), values, color=[GREY, GREY, ORANGE])
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, f"{v:.1f} %", ha="center")
    ax.set_ylabel("WAPE (%)"); ax.set_title("Error de previsión en el periodo de test")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig3_wape.png", dpi=200); plt.close(fig)

    # Fig. 4 - simulation: stockouts and excess stock
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    methods = ["moving_average", "prophet"]
    names = ["Política actual\n(media móvil)", "Política propuesta\n(Prophet)"]
    stockouts = [sim_summary.loc[m, "stockout_rate_%"] for m in methods]
    excess = [sim_summary.loc[m, "avg_excess_value_total"] for m in methods]
    axs[0].bar(names, stockouts, color=[GREY, ORANGE]); axs[0].set_title("Semanas-SKU con rotura (%)")
    axs[1].bar(names, excess, color=[GREY, ORANGE]); axs[1].set_title("Stock inmovilizado medio (€ a coste)")
    for ax, vals in zip(axs, (stockouts, excess)):
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f"{v:,.1f}".replace(",", "."), ha="center")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig4_simulation.png", dpi=200); plt.close(fig)

    # Fig. 5 - elbow and silhouette
    fig, ax1 = plt.subplots(figsize=(6, 3.2))
    ax1.plot(tests.k, tests.inertia, marker="o", color=BLUE)
    ax1.set_xlabel("Número de segmentos (k)"); ax1.set_ylabel("Inercia", color=BLUE)
    ax2 = ax1.twinx(); ax2.plot(tests.k, tests.silhouette, marker="s", color=ORANGE)
    ax2.set_ylabel("Silueta", color=ORANGE)
    ax1.axvline(k, color="k", lw=0.8, ls=":"); ax1.set_title(f"Selección del número de segmentos (k = {k})")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig5_elbow_silhouette.png", dpi=200); plt.close(fig)

    # Fig. 6 - customer segments
    fig, ax = plt.subplots(figsize=(7, 4))
    for seg, g in rfm.groupby("segment"):
        ax.scatter(g.F + rng.uniform(-0.3, 0.3, len(g)), g.M, s=4, alpha=0.4, label=SEGMENT_ES[seg])
    ax.set_yscale("log"); ax.set_xlabel("Frecuencia (pedidos en 12 meses)")
    ax.set_ylabel("Gasto en 12 meses (€, escala log)")
    ax.legend(markerscale=4, frameon=False, fontsize=8); ax.set_title("Segmentos de clientes (RFM + K-Means)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig6_segments.png", dpi=200); plt.close(fig)

    # Fig. 7 - dashboard mock-up
    fig = plt.figure(figsize=(12, 7)); fig.patch.set_facecolor("#f4f6f8")
    fig.text(0.02, 0.95, "Farmalisto España · Panel de previsión y clientes (maqueta)",
             fontsize=15, weight="bold", color=BLUE)
    fig.text(0.02, 0.915, "Filtros: Categoría [Todas] · Subcategoría [Todas] · Horizonte [4 semanas] · "
                          "Semana de referencia [última]", fontsize=9, color="#555")
    kpis = [("WAPE previsión", f"{metrics['prophet']['WAPE_%']:.1f} %"),
            ("Roturas (semanas-SKU)", f"{sim_summary.loc['prophet', 'stockout_rate_%']:.1f} %"),
            ("Stock inmovilizado", f"{sim_summary.loc['prophet', 'avg_excess_value_total']:,.0f} €".replace(",", ".")),
            ("Clientes activos 12 m", f"{len(rfm):,}".replace(",", "."))]
    for i, (title, value) in enumerate(kpis):
        ax = fig.add_axes([0.02 + i * 0.245, 0.76, 0.23, 0.12]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("white")
        ax.text(0.05, 0.62, title, fontsize=10, color="#555")
        ax.text(0.05, 0.18, value, fontsize=20, weight="bold", color=BLUE)
    ax = fig.add_axes([0.05, 0.33, 0.58, 0.36])
    total = demand.groupby("week").units.sum()
    total_fc = forecasts[forecasts.week >= test_start].groupby("week").prophet.sum()
    ax.plot(total.index[-52:], total.values[-52:], color=GREY, label="Real")
    ax.plot(total_fc.index, total_fc.values, color=ORANGE, label="Previsión")
    ax.set_title("Demanda total y previsión (últimas 52 semanas)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = fig.add_axes([0.80, 0.33, 0.18, 0.36])
    seg = profile.sort_values("revenue_%")
    ax.barh(seg.segment.map(SEGMENT_ES), seg["revenue_%"], color=BLUE)
    ax.set_title("% de ingresos por segmento", fontsize=10)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.24]); ax.axis("off")
    last = sim[(sim.method == "prophet") & (sim.week == sim.week.max())].merge(
        catalog[["sku", "category", "lead_time_weeks"]])
    last = last.merge(forecasts[forecasts.week == forecasts.week.max()][["sku", "prophet"]], on="sku")
    last["cover_weeks"] = (last.ending_stock / last.prophet.clip(lower=0.1)).round(1)
    alerts = last.sort_values("cover_weeks").head(6)
    cells = [[r.sku, CATEGORY_ES[r.category], int(r.ending_stock), round(r.prophet, 0), r.cover_weeks,
              r.lead_time_weeks, "Pedir ya" if r.cover_weeks <= r.lead_time_weeks else "Vigilar"]
             for r in alerts.itertuples()]
    table = ax.table(cellText=cells, colLabels=["SKU", "Categoría", "Stock", "Previsión sem.", "Cobertura (sem.)",
                                                "Plazo prov. (sem.)", "Alerta"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.4)
    ax.set_title("Alertas de reposición (vista Operaciones)", fontsize=10, loc="left")
    fig.savefig(f"{fig_dir}/fig7_dashboard_mockup.png", dpi=200, facecolor=fig.get_facecolor()); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prophet", action="store_true", help="quick test without Prophet")
    use_prophet = not parser.parse_args().no_prophet
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

    print("1/6 Generating synthetic data...")
    campaigns = build_campaigns()
    catalog = build_catalog()
    demand = build_demand(catalog, campaigns)
    customers, orders = build_customers_and_orders(catalog, demand)

    print("2/6 Forecasting demand", "(Prophet)..." if use_prophet else "(test mode, no Prophet)...")
    forecasts = run_forecasts(catalog, demand, campaigns, use_prophet)
    metrics = {m: forecast_metrics(forecasts, m) for m in ["moving_average", "seasonal_naive", "prophet"]}

    print("3/6 Simulating replenishment...")
    sim = pd.concat([simulate(m, catalog, demand, forecasts) for m in ["moving_average", "prophet"]])
    sim_summary = summarize_simulation(sim, len(catalog))
    # what-if scenario: every supplier takes one extra week to deliver
    slow_catalog = catalog.assign(lead_time_weeks=catalog.lead_time_weeks + 1)
    slow_sim = pd.concat([simulate(m, slow_catalog, demand, forecasts) for m in ["moving_average", "prophet"]])
    slow_summary = summarize_simulation(slow_sim, len(catalog))

    print("4/6 Segmenting customers...")
    rfm, profile, tests, k = segment_customers(orders)

    print("5/6 Drawing figures...")
    make_figures(catalog, demand, forecasts, metrics, sim, sim_summary, rfm, profile, tests, k)

    print("6/6 Saving results...")
    base, prop = sim_summary.loc["moving_average"], sim_summary.loc["prophet"]
    best_baseline = min(metrics["moving_average"]["WAPE_%"], metrics["seasonal_naive"]["WAPE_%"])
    summary = {
        "mode": "prophet" if use_prophet else "test_without_prophet",
        "dataset": {"weeks": N_WEEKS, "skus": len(catalog), "customers_generated": N_CUSTOMERS,
                    "customers_with_orders": int(orders.customer_id.nunique()),
                    "orders": int(orders.order_id.nunique()), "order_lines": len(orders),
                    "units": int(demand.units.sum()), "revenue_eur": round(orders.amount.sum(), 0),
                    "period": f"{weeks[0].date()} to {weeks[-1].date()}"},
        "forecast": metrics,
        "wape_improvement_vs_best_baseline_%": round(100 * (best_baseline - metrics["prophet"]["WAPE_%"]) / best_baseline, 1),
        "simulation": sim_summary.reset_index().to_dict(orient="records"),
        "stockout_reduction_%": round(100 * (base["stockout_rate_%"] - prop["stockout_rate_%"]) / base["stockout_rate_%"], 1),
        "excess_stock_reduction_%": round(100 * (base.avg_excess_value_total - prop.avg_excess_value_total)
                                          / base.avg_excess_value_total, 1),
        "annual_holding_cost_saving_eur": round((base.avg_stock_value_total - prop.avg_stock_value_total)
                                                * HOLDING_COST_RATE, 0),
        "lost_sales_avoided_26w_eur": round(base.lost_sales - prop.lost_sales, 0),
        "scenario_lead_time_plus_1_week": slow_summary.reset_index().to_dict(orient="records"),
        "segmentation": {"k": k, "tests": tests.to_dict(orient="records"), "profile": profile.to_dict(orient="records")},
    }
    with open(f"{OUTPUT_DIR}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)

    tables = {"catalog": catalog, "weekly_demand": demand, "campaigns": campaigns, "customers": customers,
              "orders": orders, "forecasts": forecasts, "replenishment_simulation": sim, "rfm_segments": rfm,
              "segment_profile": profile}
    for name, df in tables.items():
        df.to_csv(f"{OUTPUT_DIR}/data/{name}.csv", index=False)
    with pd.ExcelWriter(f"{OUTPUT_DIR}/data/powerbi_dataset.xlsx") as writer:
        for name in ["catalog", "weekly_demand", "campaigns", "forecasts", "replenishment_simulation",
                     "rfm_segments", "segment_profile"]:
            tables[name].to_excel(writer, sheet_name=name[:31], index=False)

    print(json.dumps({key: summary[key] for key in ["forecast", "wape_improvement_vs_best_baseline_%",
                                                    "stockout_reduction_%", "excess_stock_reduction_%"]}, indent=2))
    print(f"Done. Results saved in ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
