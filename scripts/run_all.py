"""End-to-end pipeline: `python scripts/run_all.py` (add --force to re-download)."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from volforecast.config import PROJECT_ROOT, load_config
from volforecast.data import load_close, load_universe
from volforecast.evaluation import common_sample, metrics_by_ticker, universe_summary
from volforecast.features import har_features, pooled_dataset
from volforecast.models.gbm import GBM
from volforecast.models.garch import garch_forecast
from volforecast.models.har import HAR
from volforecast.models.naive import naive_forecast
from volforecast.plots import target_vs_vix
from volforecast.targets import build_targets, variance_components
from volforecast.walkforward import make_splits, run_supervised

PROCESSED = PROJECT_ROOT / "data" / "processed"
PREDICTIONS = PROJECT_ROOT / "results" / "predictions"
TABLES = PROJECT_ROOT / "reports" / "tables"
FIGURES = PROJECT_ROOT / "reports" / "figures"


def data_stage(cfg: dict, force: bool) -> pd.DataFrame:
    """Download and clean the data, build the target, write a sanity-check report."""
    panel, cleaning = load_universe(cfg, force)
    calendar = panel[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar, force)
    load_close(cfg["data"]["risk_free"], cfg, calendar, force)  # cached for the backtest

    targets = build_targets(panel, cfg)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(PROCESSED / "targets.parquet")

    ann = cfg["target"]["annualization"]
    rows = []
    for ticker, df in panel.items():
        comp = variance_components(df).dropna()
        tgt = targets.xs(ticker, level="ticker")
        rows.append({
            "ticker": ticker,
            "mean_vol_pct": 100 * np.sqrt(ann * tgt["daily_var"].mean()),
            "overnight_share_pct": 100 * comp["overnight"].sum() / comp.sum().sum(),
            "n_targets": int(tgt["target_rv"].notna().sum()),
        })
    summary = cleaning.join(pd.DataFrame(rows).set_index("ticker")).round(2)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "data_summary.csv")
    print(summary.to_string())
    print(f"\nVIX missing on benchmark calendar: {int(vix.isna().sum())} days")

    spy = targets.xs(cfg["data"]["benchmark"], level="ticker")["target_rv"]
    target_vs_vix(spy, vix, FIGURES / "spy_target_vs_vix.png")
    return panel, targets


def _benchmarks_one_ticker(item, cfg, splits):
    """Naive, HAR, GARCH and GJR forecasts for one ticker (run in a worker process)."""
    ticker, df, tgt = item
    t = cfg["target"]
    X = har_features(tgt["daily_var"], t["annualization"])
    first_test = splits[0].test_dates[0]
    out = pd.DataFrame({
        "naive": naive_forecast(X)[lambda s: s.index >= first_test],
        "har": run_supervised(HAR, X, tgt["target_log_rv"], splits),
    })
    n_failed = {}
    for name, asym in (("garch", False), ("gjr", True)):
        fc, n_failed[name] = garch_forecast(df["close"], splits, asym, t["horizon"], t["annualization"])
        out[name] = fc
    out["ticker"] = ticker
    return out.set_index("ticker", append=True), {"ticker": ticker, **n_failed}


def benchmark_stage(cfg: dict, panel: dict, targets: pd.DataFrame) -> pd.DataFrame:
    calendar = panel[cfg["data"]["benchmark"]].index
    w = cfg["walkforward"]
    splits = make_splits(calendar, w["test_start"], w["refit_every"], cfg["target"]["horizon"])
    items = [(tk, panel[tk], targets.xs(tk, level="ticker")) for tk in panel]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(partial(_benchmarks_one_ticker, cfg=cfg, splits=splits), items))
    forecasts = pd.concat([r[0] for r in results]).sort_index()
    failures = pd.DataFrame([r[1] for r in results]).set_index("ticker")
    print(f"{len(splits)} refits per ticker; non-converged GARCH/GJR fits: "
          f"{int(failures['garch'].sum())}/{int(failures['gjr'].sum())}")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    for model in forecasts.columns:
        forecasts[[model]].to_parquet(PREDICTIONS / f"{model}.parquet")
    return forecasts


def ml_stage(cfg: dict, panel: dict, targets: pd.DataFrame) -> pd.DataFrame:
    """Pooled models: GBM (full features), plus two ablations (GBM on HAR features
    only, and a single HAR estimated on all tickers at once)."""
    calendar = panel[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar)
    w, g, h = cfg["walkforward"], cfg["gbm"], cfg["target"]["horizon"]
    splits = make_splits(calendar, w["test_start"], w["refit_every"], h)
    X, y = pooled_dataset(panel, targets, vix, cfg)

    fitted = []

    def gbm_factory(features, importance):
        def make():
            model = GBM(g["params"], features, g["val_days"], h, g["max_iter"],
                        g["n_iter_no_change"], cfg["seed"], importance)
            if importance:
                fitted.append(model)
            return model
        return make

    X_har = np.exp(X[["log_rv_d", "log_rv_w", "log_rv_m"]]).rename(columns=lambda c: c[4:])
    out = pd.DataFrame({
        "gbm": run_supervised(gbm_factory(g["features"], True), X[g["features"]], y, splits),
        "gbm_har_feats": run_supervised(gbm_factory(g["har_only_features"], False),
                                        X[g["har_only_features"]], y, splits),
        "har_pooled": run_supervised(HAR, X_har, y, splits),
    })

    importances = pd.DataFrame([m.importances_ for m in fitted],
                               index=[s.refit_date for s in splits])
    importances.to_csv(TABLES / "gbm_permutation_importance_by_refit.csv")
    summary = importances.agg(["mean", "std"]).T.sort_values("mean", ascending=False)
    summary.round(5).to_csv(TABLES / "gbm_permutation_importance.csv")
    trees = pd.Series([m.n_trees_ for m in fitted])
    print(f"GBM trees per refit: median {trees.median():.0f}, range {trees.min()}-{trees.max()}")

    for model in out.columns:
        out[[model]].to_parquet(PREDICTIONS / f"{model}.parquet")
    return out


def evaluation_stage(forecasts: pd.DataFrame, targets: pd.DataFrame) -> None:
    models = list(forecasts.columns)
    sample = common_sample(forecasts, targets["target_rv"] ** 2)
    by_ticker = metrics_by_ticker(sample, models)
    summary = universe_summary(by_ticker)
    by_ticker.round(5).to_csv(TABLES / "metrics_by_ticker.csv")
    summary.round(4).to_csv(TABLES / "metrics_universe.csv")
    dates = sample.index.get_level_values("date")
    print(f"\nCommon sample: {dates.min().date()} to {dates.max().date()}, {len(sample)} obs")
    print(summary.round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download raw data")
    args = parser.parse_args()
    cfg = load_config()
    panel, targets = data_stage(cfg, args.force)
    forecasts = benchmark_stage(cfg, panel, targets)
    forecasts = forecasts.join(ml_stage(cfg, panel, targets))
    evaluation_stage(forecasts, targets)


if __name__ == "__main__":
    main()
