# Parapharmacy inventory analytics

Demand forecasting, replenishment simulation and customer segmentation for an online parapharmacy.

This repository contains the code and data behind the Master's thesis *Optimización de inventario y retención de clientes mediante analítica de datos* (Master in Strategic Business Analytics, EAE Business School, 2026). The thesis studies a **hypothetical Spanish subsidiary of Farmalisto**, a Latin American digital pharmacy, selling dermocosmetics and food supplements online. No real company data is used: every dataset here is synthetic.

## What it does

`run_analysis.py` runs the whole pipeline end to end:

1. **Synthetic data.** Builds 104 weeks of activity (November 2024 to October 2026) for 60 SKUs and 15,000 customers: trend, seasonality, commercial campaigns, negative binomial noise and orders that add up exactly to weekly demand. A fixed random seed makes it fully reproducible.
2. **Forecasting.** Predicts weekly demand per SKU four weeks ahead with Prophet and compares it against two baselines (4-week moving average and seasonal naive), using rolling origins.
3. **Replenishment simulation.** Replays the last 26 weeks as a warehouse would live them, with a weekly order-up-to policy driven by each forecast, and measures stockouts, excess stock and lost sales. A what-if scenario repeats the simulation with every supplier delivering one week later.
4. **Customer segmentation.** Computes RFM (recency, frequency, monetary value) over the last 12 months and groups customers with K-Means.
5. **Outputs.** Exports data, metrics and the figures used in the thesis.

## Results

| Metric (26-week test period) | Current policy (moving average) | Proposed (Prophet) |
|---|---|---|
| Forecast error (WAPE) | 39.1 % | 32.9 % |
| SKU-weeks with stockout | 4.36 % | 2.82 % |
| Average excess stock (cost value) | 4,588 € | 3,492 € |
| Fill rate | 97.2 % | 98.8 % |

Prophet cuts forecast error by 15.9 % against the best baseline, which translates into 35 % fewer stockouts and 24 % less excess stock at a higher service level. The advantage holds when supplier lead times grow by one week (3.85 % vs 4.74 % stockouts; 3,746 € vs 5,994 € excess stock). Segmentation yields four groups; high-value loyal customers are 33 % of the base and 72 % of revenue.

These figures come from synthetic data designed to behave plausibly. They show that the method works and its order of magnitude, not what a real deployment would achieve.

## Repository structure

```
├── run_analysis.py        # full pipeline
├── requirements.txt
└── results/
    ├── summary.json       # key metrics
    ├── data/              # synthetic dataset, forecasts, simulation and segments (CSV)
    │   └── powerbi_dataset.xlsx
    └── figures/           # charts used in the thesis (labels in Spanish)
```

## How to run

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python run_analysis.py
```

The full run takes about 5–10 minutes, most of it fitting Prophet 780 times (60 SKUs × 13 origins). For a quick check of the pipeline without Prophet:

```bash
python run_analysis.py --no-prophet
```

On Windows, Prophet may log a few `cmdstanpy ... Operation not permitted` errors. They are harmless: Prophet retries the fit with another optimizer.

## Methodological notes

- **Log transform.** Prophet is fitted on `log(1 + units)`, which stabilises variance in count data and prevents negative forecasts.
- **Campaigns as one regressor.** With two years of history each campaign appears only once or twice, so campaigns enter as a single regressor per category instead of one effect each.
- **Plausibility cap.** No forecast may exceed twice the SKU's historical peak, applied equally to every method.
- **Safety stock.** `z · σ · √(L + 1)` with z = 1.65 for every policy; σ is a robust estimate (1.25 × MAE) of each method's own error over a 26-week calibration period.
- **Number of segments.** Chosen between 3 and 6 by silhouette score; k = 2 is excluded as not actionable.

## Author

Xavier Piernas Andrés
