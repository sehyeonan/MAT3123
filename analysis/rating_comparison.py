# -*- coding: utf-8 -*-
"""
rating_comparison.py

Port of rating_comparison.R:
- Load IMDB and Naver review CSVs
- Standardize columns and merge
- Provide plotting helpers:
    * rating density (KDE)
    * histogram
    * box / violin
    * mean rating trend over time (day/week/month)
    * early vs late density split (by quantile or fixed cutoff)

Defaults mimic the R script's 'output' folder usage.

Usage examples:
  python rating_comparison.py --output-dir output --demo
  python rating_comparison.py --output-dir output --trend --time-unit month
  python rating_comparison.py --output-dir output --early-late --split-prop 0.5
"""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except Exception:
    HAVE_KDE = False

def make_reviews_data(imdb: pd.DataFrame, naver: pd.DataFrame) -> pd.DataFrame:
    imdb_std = imdb.copy()
    imdb_std["platform"] = "imdb"
    imdb_std["date"] = pd.to_datetime(imdb_std["date"], errors="coerce").dt.date
    imdb_std["rating"] = pd.to_numeric(imdb_std.get("star_rating"), errors="coerce")
    imdb_std["text"] = imdb_std.get("content", np.nan)
    imdb_std["upvote"] = np.nan
    imdb_std["downvote"] = np.nan
    imdb_std = imdb_std[["id", "platform", "date", "rating", "text", "upvote", "downvote"]]

    naver_std = naver.copy()
    naver_std["platform"] = "naver"
    naver_std["date"] = pd.to_datetime(naver_std["writing_date"], errors="coerce").dt.date
    naver_std["rating"] = pd.to_numeric(naver_std.get("star_rating"), errors="coerce")
    naver_std["text"] = naver_std.get("comment", np.nan)
    if "upvote" not in naver_std.columns:
        naver_std["upvote"] = np.nan
    if "downvote" not in naver_std.columns:
        naver_std["downvote"] = np.nan
    naver_std = naver_std[["id", "platform", "date", "rating", "text", "upvote", "downvote"]]

    out = pd.concat([imdb_std, naver_std], ignore_index=True).sort_values("date")
    return out

def filter_by_date_platform(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    platform_sel: str = "both",
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if start_date:
        out = out[out["date"] >= pd.to_datetime(start_date)]
    if end_date:
        out = out[out["date"] <= pd.to_datetime(end_date)]
    if platform_sel != "both":
        out = out[out["platform"] == platform_sel]
    return out

def plot_rating_density(df: pd.DataFrame, which_platform: str = "both") -> plt.Figure:
    d = filter_by_date_platform(df, platform_sel=which_platform).dropna(subset=["rating"])
    fig = plt.figure()
    if d.empty:
        plt.title("No data after filtering.")
        return fig

    x = np.linspace(1, 10, 400)
    if HAVE_KDE:
        for platform, sub in d.groupby("platform"):
            vals = sub["rating"].astype(float).to_numpy()
            vals = vals[(vals >= 1) & (vals <= 10)]
            if len(vals) < 2:
                continue
            y = gaussian_kde(vals)(x)
            plt.plot(x, y, label=platform)
            plt.fill_between(x, 0, y, alpha=0.25)
        plt.ylabel("Density")
    else:
        warnings.warn("SciPy KDE not available; using histogram density instead.")
        bins = np.arange(0.5, 10.6, 1.0)
        for platform, sub in d.groupby("platform"):
            plt.hist(sub["rating"], bins=bins, density=True, alpha=0.35, label=platform)
        plt.ylabel("Density (hist)")

    plt.xlabel("Rating")
    plt.title("Rating density by platform")
    plt.xlim(1, 10)
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    return fig

def plot_rating_hist(df: pd.DataFrame, which_platform: str = "both", binwidth: int = 1) -> plt.Figure:
    d = filter_by_date_platform(df, platform_sel=which_platform).dropna(subset=["rating"])
    fig = plt.figure()
    bins = np.arange(0.5, 10.6, binwidth)
    if which_platform == "both":
        for platform, sub in d.groupby("platform"):
            plt.hist(sub["rating"], bins=bins, alpha=0.45, label=platform)
        plt.legend()
    else:
        plt.hist(d["rating"], bins=bins, alpha=0.8)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.title("Rating histogram")
    plt.xlim(0.5, 10.5)
    plt.xticks(range(1, 11))
    plt.tight_layout()
    return fig

def plot_rating_box(df: pd.DataFrame) -> plt.Figure:
    d = df.dropna(subset=["rating"]).copy()
    fig = plt.figure()
    data = [sub["rating"].astype(float).to_numpy() for _, sub in d.groupby("platform")]
    labels = [p for p, _ in d.groupby("platform")]
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylim(1, 10)
    plt.yticks(range(1, 11))
    plt.title("Rating distribution (boxplot)")
    plt.xlabel("Platform")
    plt.ylabel("Rating")
    plt.tight_layout()
    return fig

def plot_rating_violin(df: pd.DataFrame) -> plt.Figure:
    d = df.dropna(subset=["rating"]).copy()
    fig = plt.figure()
    data = [sub["rating"].astype(float).to_numpy() for _, sub in d.groupby("platform")]
    labels = [p for p, _ in d.groupby("platform")]
    plt.violinplot(data, showmeans=False, showmedians=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylim(1, 10)
    plt.yticks(range(1, 11))
    plt.title("Rating distribution (violin)")
    plt.xlabel("Platform")
    plt.ylabel("Rating")
    plt.tight_layout()
    return fig

def summarise_ratings(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["rating"]).copy()
    out = []
    for platform, sub in d.groupby("platform"):
        r = sub["rating"].astype(float)
        out.append({
            "platform": platform,
            "n": len(r),
            "mean": float(r.mean()),
            "sd": float(r.std(ddof=1)),
            "median": float(r.median()),
            "p25": float(r.quantile(0.25)),
            "p75": float(r.quantile(0.75)),
            "prop_10": float((r == 10).mean()),
            "prop_7_10": float((r >= 7).mean()),
            "prop_1_3": float((r <= 3).mean()),
        })
    return pd.DataFrame(out)

def summarise_rating_trend(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    platform_sel: str = "both",
    time_unit: str = "month",
    min_n: int = 5,
) -> pd.DataFrame:
    d = filter_by_date_platform(df, start_date, end_date, platform_sel).dropna(subset=["rating"]).copy()
    if d.empty:
        return d.iloc[0:0].copy()

    d["date"] = pd.to_datetime(d["date"])
    freq = {"day": "D", "week": "W-MON", "month": "MS"}[time_unit]

    out = []
    for platform, sub in d.groupby("platform"):
        sub = sub.set_index("date").sort_index()
        g = sub["rating"].resample(freq)
        agg = pd.DataFrame({"mean_rating": g.mean(), "sd_rating": g.std(ddof=1), "n": g.size()}).dropna()
        agg = agg[agg["n"] >= min_n].reset_index().rename(columns={"date": "period"})
        agg["platform"] = platform
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else d.iloc[0:0].copy()

def plot_rating_trend(df: pd.DataFrame, **kwargs) -> plt.Figure:
    trend = summarise_rating_trend(df, **kwargs)
    fig = plt.figure()
    if trend.empty:
        plt.title("No data after filtering.")
        return fig
    for platform, sub in trend.groupby("platform"):
        plt.plot(sub["period"], sub["mean_rating"], marker="o", label=platform)
    plt.ylim(1, 10)
    plt.yticks(range(1, 11))
    plt.title("시간에 따른 평균 평점 추이")
    plt.xlabel("Time")
    plt.ylabel("Mean rating")
    plt.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def add_early_late_group(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    platform_sel: str = "both",
    split_prop: float = 0.5,
) -> pd.DataFrame:
    d = filter_by_date_platform(df, start_date, end_date, platform_sel).copy()
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    cutoff = d["date"].quantile(split_prop, interpolation="lower")
    d["period_group"] = np.where(d["date"] <= cutoff, "early", "late")
    d["cutoff_date"] = cutoff.date()
    return d

def plot_early_late_density(df: pd.DataFrame, split_prop: float = 0.5, platform_sel: str = "both") -> plt.Figure:
    d = add_early_late_group(df, platform_sel=platform_sel, split_prop=split_prop).dropna(subset=["rating"])
    fig = plt.figure()
    if d.empty:
        plt.title("No data after filtering.")
        return fig

    x = np.linspace(1, 10, 400)
    if HAVE_KDE:
        for platform, subp in d.groupby("platform"):
            for grp, sub in subp.groupby("period_group"):
                vals = sub["rating"].astype(float).to_numpy()
                vals = vals[(vals >= 1) & (vals <= 10)]
                if len(vals) < 2:
                    continue
                y = gaussian_kde(vals)(x)
                plt.plot(x, y, label=f"{platform}-{grp}")
        plt.ylabel("Density")
    else:
        bins = np.arange(0.5, 10.6, 1.0)
        for platform, subp in d.groupby("platform"):
            for grp, sub in subp.groupby("period_group"):
                plt.hist(sub["rating"], bins=bins, density=True, alpha=0.25, label=f"{platform}-{grp}")
        plt.ylabel("Density (hist)")

    cutoff = d["cutoff_date"].iloc[0] if "cutoff_date" in d.columns and len(d) else None
    plt.title(f"초기 vs 후기 평점 분포 (cutoff: {cutoff})")
    plt.xlabel("Rating")
    plt.xlim(1, 10)
    plt.xticks(range(1, 11))
    plt.legend()
    plt.tight_layout()
    return fig

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output", help="Directory containing CSVs.")
    ap.add_argument("--imdb-file", default="reviews_imdb.csv")
    ap.add_argument("--naver-file", default="reviews_naver.csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--trend", action="store_true")
    ap.add_argument("--time-unit", default="month", choices=["month", "week", "day"])
    ap.add_argument("--early-late", action="store_true")
    ap.add_argument("--split-prop", type=float, default=0.5)
    ap.add_argument("--save", default=None, help="If set, save the last plot to this PNG path.")
    args = ap.parse_args()

    od = Path(args.output_dir)
    imdb_path = od / args.imdb_file
    naver_path = od / args.naver_file
    if not imdb_path.exists():
        raise FileNotFoundError(f"Missing IMDB CSV: {imdb_path.resolve()}")
    if not naver_path.exists():
        raise FileNotFoundError(f"Missing Naver CSV: {naver_path.resolve()}")

    imdb = pd.read_csv(imdb_path)
    naver = pd.read_csv(naver_path)
    reviews_all = make_reviews_data(imdb, naver)

    fig = None
    if args.demo:
        print(summarise_ratings(reviews_all).to_string(index=False))
        plot_rating_density(reviews_all, "both")
        plot_rating_hist(reviews_all, "both")
        plot_rating_box(reviews_all)
        plot_rating_violin(reviews_all)
        plt.show()
        return
    if args.trend:
        fig = plot_rating_trend(reviews_all, time_unit=args.time_unit, min_n=1)
    elif args.early_late:
        fig = plot_early_late_density(reviews_all, split_prop=args.split_prop)
    else:
        fig = plot_rating_density(reviews_all, "both")

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
