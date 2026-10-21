"""End-to-end pipeline: `python scripts/run_all.py`.

--force      re-download the raw data
--use-cache  reload model forecasts already in results/predictions instead of
             recomputing them (development only; the default recomputes everything)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

from volforecast.backtest import performance, run_backtest, target_weights, weekly_rebalance_dates
from volforecast.config import PROJECT_ROOT, load_config
from volforecast.data import load_close, load_universe
from volforecast.evaluation import (common_sample, dm_table, metrics_by_ticker, qlike,
                                   regime_table, to_markdown, universe_summary)
from volforecast.features import har_features, pooled_dataset
from volforecast.models.gbm import GBM
from volforecast.models.garch import garch_forecast
from volforecast.models.har import HAR, LinearLogVol
from volforecast.models.lstm import LSTMConfig, lstm_walkforward
from volforecast.models.naive import naive_forecast
from volforecast.plots import (backtest_nav, cumulative_loss_difference, feature_importance,
                              forecasts_vs_realized, target_vs_vix)
from volforecast.targets import build_targets, variance_components
from volforecast.walkforward import make_splits, run_supervised, yearly_splits

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


def lstm_stage(cfg: dict, panel: dict, targets: pd.DataFrame) -> pd.DataFrame:
    """Pooled LSTM, refitted yearly; the forecast is the average over seeds."""
    calendar = panel[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar)
    X, y = pooled_dataset(panel, targets, vix, cfg)
    splits = yearly_splits(calendar, cfg["walkforward"]["test_start"], cfg["target"]["horizon"])
    by_seed, logs = lstm_walkforward(X[cfg["lstm"]["features"]], y, splits,
                                     LSTMConfig.from_cfg(cfg), cfg["lstm"]["seeds"])
    logs.to_csv(TABLES / "lstm_training_log.csv", index=False)
    by_seed.to_parquet(PREDICTIONS / "lstm_by_seed.parquet")
    print(f"LSTM best epochs: median {logs['best_epoch'].median():.0f}, "
          f"range {logs['best_epoch'].min()}-{logs['best_epoch'].max()}")
    out = by_seed.mean(axis=1).rename("lstm").to_frame()
    out.to_parquet(PREDICTIONS / "lstm.parquet")
    return out


def cached(name: str, compute, use_cache: bool) -> pd.DataFrame:
    """Run a forecasting stage, or reload its saved forecasts when allowed."""
    path = PREDICTIONS / f"{name}.parquet"
    if use_cache and path.exists():
        print(f"[cache] {name}")
        return pd.read_parquet(path)
    out = compute()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def harx_stage(cfg: dict, panel: dict, targets: pd.DataFrame) -> pd.DataFrame:
    """HAR-X control: linear model on the ML feature set, pooled and per ticker."""
    calendar = panel[cfg["data"]["benchmark"]].index
    vix = load_close(cfg["data"]["vix"], cfg, calendar)
    w, h, feats = cfg["walkforward"], cfg["target"]["horizon"], cfg["gbm"]["features"]
    splits = make_splits(calendar, w["test_start"], w["refit_every"], h)
    X, y = pooled_dataset(panel, targets, vix, cfg)
    X = X[feats]
    per_ticker = [run_supervised(lambda: LinearLogVol(feats), X.xs(tk, level="ticker"),
                                 y.xs(tk, level="ticker"), splits).to_frame().assign(ticker=tk)
                  for tk in panel]
    return pd.DataFrame({
        "harx_pooled": run_supervised(lambda: LinearLogVol(feats), X, y, splits),
        "harx_ticker": pd.concat(per_ticker).set_index("ticker", append=True)["forecast_var"],
    })


def evaluation_stage(cfg: dict, forecasts: pd.DataFrame, targets: pd.DataFrame) -> None:
    ev, h = cfg["evaluation"], cfg["target"]["horizon"]
    main, ablation = ev["main_models"], ev["ablation_models"]
    models = list(dict.fromkeys(main + ablation))
    sample = common_sample(forecasts[models], targets["target_rv"] ** 2)
    dates = sample.index.get_level_values("date")
    print(f"\nCommon sample: {dates.min().date()} to {dates.max().date()}, {len(sample)} obs")

    # Point metrics.
    by_ticker = metrics_by_ticker(sample, models)
    by_ticker.round(5).to_csv(TABLES / "metrics_by_ticker.csv")
    summary = universe_summary(by_ticker)
    summary.round(4).to_csv(TABLES / "metrics_universe.csv")

    # Diebold-Mariano (QLIKE and MSE) against HAR and GARCH.
    dm = pd.concat([dm_table(sample, main, ev["dm_references"], h, loss) for loss in ("qlike", "mse")])
    dm.round(4).to_csv(TABLES / "diebold_mariano.csv", index=False)

    # Regimes.
    regimes = regime_table(sample, main, ev["regimes"], h)
    regimes.round(4).to_csv(TABLES / "regimes.csv", index=False)

    # Markdown versions for the README.
    cols = ["qlike", "mse_var", "mae_vol", "mz_beta", "mz_r2", "qlike_vs_har", "pct_tickers_beat_har"]
    def dm_cells(loss):
        d = dm[dm["loss"] == loss]
        cell = d.apply(lambda r: f"p={r.universe_p_value:.3f} ({r.n_tickers_better}+ / "
                                 f"{r.n_tickers_worse}-)", axis=1)
        return d.assign(cell=cell).pivot(index="model", columns="reference", values="cell") \
                .reindex(main).fillna("—").add_prefix("vs ")

    reg = regimes.assign(cell=regimes.apply(
        lambda r: f"{r.qlike_vs_har:.3f}" + ("" if r.model == "har" else f" (p={r.dm_p_value:.3f})"), axis=1))
    md = {
        "Main results (ticker-averaged)": summary.loc[main, cols].sort_values("qlike"),
        "Ablations": summary.loc[ablation, cols].sort_values("qlike"),
        "Diebold-Mariano on QLIKE (universe p-value; tickers significantly better+ / worse-)": dm_cells("qlike"),
        "Diebold-Mariano on MSE": dm_cells("mse"),
        "QLIKE ratio to HAR by regime (universe DM p-value vs HAR)": reg.pivot(
            index="model", columns="regime", values="cell").loc[main, list(ev["regimes"])],
    }
    text = "\n\n".join(f"### {k}\n\n{to_markdown(v)}" for k, v in md.items())
    (TABLES / "results.md").write_text(text + "\n")
    print(text)

    # Figures.
    spy = sample.xs(cfg["data"]["benchmark"], level="ticker")
    forecasts_vs_realized(spy, ["har", "lstm"], "SPY — 5-day realized vol vs forecasts, 2012–2025",
                          FIGURES / "spy_forecasts_full.png")
    forecasts_vs_realized(spy.loc["2020-01-01":"2020-12-31"], ["garch", "har", "harx_ticker", "lstm"],
                          "SPY — 5-day realized vol vs forecasts, 2020",
                          FIGURES / "spy_forecasts_2020.png")
    losses = pd.DataFrame({m: qlike(sample["realized"].to_numpy(), sample[m].to_numpy())
                           for m in main}, index=sample.index).groupby(level="date").mean()
    diff = losses[[m for m in main if m not in ("har", "naive")]].sub(losses["har"], axis=0)
    cumulative_loss_difference(diff, FIGURES / "cumulative_qlike_vs_har.png")
    feature_importance(pd.read_csv(TABLES / "gbm_permutation_importance.csv", index_col=0),
                       FIGURES / "gbm_feature_importance.png")


def backtest_stage(cfg: dict, panel: dict, forecasts: pd.DataFrame, targets: pd.DataFrame) -> None:
    b = cfg["backtest"]
    asset = panel[b["asset"]]
    calendar = asset.index
    ret = asset["close"].pct_change()  # adjusted close: total return
    irx = load_close(cfg["data"]["risk_free"], cfg, calendar).ffill()
    rf = (irx / 100 / 252).shift(1)    # yesterday's T-bill rate accrues today

    fc = forecasts.xs(b["asset"], level="ticker")
    vols = {m: np.sqrt(fc[m]) for m in b["strategies"]}
    # Oracle: the realized vol of the coming week, known in advance (infeasible).
    # Shifted so that the value read on R-1 is the realized vol over R+1..R+5.
    vols["oracle"] = targets.xs(b["asset"], level="ticker")["target_rv"].shift(-1)

    # Same period for every strategy: rebalancing dates where every signal exists.
    rebal = weekly_rebalance_dates(calendar)
    weights = {k: target_weights(v, rebal, calendar, b["target_vol"], b["max_leverage"])
               for k, v in vols.items()}
    common = sorted(set.intersection(*(set(w.index) for w in weights.values())))
    start, end = common[0], common[-1]
    weights = {k: w.loc[start:end] for k, w in weights.items()}
    ret, rf = ret.loc[:end], rf.loc[:end]

    results, navs = {}, {}
    bh = run_backtest(ret, rf, pd.Series(1.0, index=[start]), b["cost_bps"])
    results["buy_hold"], navs["buy_hold"] = performance(bh, rf), bh["nav"]
    for name, w in weights.items():
        bt = run_backtest(ret, rf, w, b["cost_bps"])
        results[name], navs[name] = performance(bt, rf), bt["nav"]
        # Buy-and-hold scaled (ex post) to the same realized vol, rebalanced weekly
        # at a constant weight with the same costs: a fair, if hindsight, benchmark.
        k = results[name]["ann_vol"] / results["buy_hold"]["ann_vol"]
        matched = run_backtest(ret, rf, pd.Series(k, index=w.index), b["cost_bps"])
        perf = performance(matched, rf)
        if name == "har":
            navs["bh_matched"] = matched["nav"]
        results[name].update({"bh_matched_weight": k, "bh_matched_return": perf["ann_return"],
                              "bh_matched_sharpe": perf["sharpe"],
                              "bh_matched_max_dd": perf["max_drawdown"]})

    table = pd.DataFrame(results).T
    table.round(4).to_csv(TABLES / "backtest.csv")
    print(f"\nBacktest {start.date()} to {end.date()}")
    print(table.round(3).to_string())
    cols = ["ann_return", "ann_vol", "sharpe", "sharpe_se", "max_drawdown", "turnover_per_year",
            "avg_weight", "bh_matched_return", "bh_matched_sharpe", "bh_matched_max_dd"]
    order = ["buy_hold"] + b["strategies"] + ["oracle"]
    with open(TABLES / "results.md", "a") as f:
        f.write(f"\n### Volatility targeting on SPY ({start.date()} to {end.date()})\n\n"
                f"{to_markdown(table.loc[order, cols])}\n")
    backtest_nav(pd.DataFrame(navs)[["bh_matched", "har", "garch", "harx_ticker", "lstm"]],
                 FIGURES / "backtest_nav.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-download raw data")
    parser.add_argument("--use-cache", action="store_true", help="reload saved forecasts")
    args = parser.parse_args()
    cfg = load_config()
    panel, targets = data_stage(cfg, args.force)
    forecasts = cached("stage_benchmarks", lambda: benchmark_stage(cfg, panel, targets), args.use_cache)
    forecasts = forecasts.join(cached("stage_ml", lambda: ml_stage(cfg, panel, targets), args.use_cache))
    forecasts = forecasts.join(cached("stage_lstm", lambda: lstm_stage(cfg, panel, targets), args.use_cache))
    forecasts = forecasts.join(cached("stage_harx", lambda: harx_stage(cfg, panel, targets), args.use_cache))
    evaluation_stage(cfg, forecasts, targets)
    backtest_stage(cfg, panel, forecasts, targets)


if __name__ == "__main__":
    main()
