# -*- coding: utf-8 -*-
"""
rating_phase_analysis.py

Port of rating_phase_analysis.R:
- Read imdb_clean_nostop.csv and reviews_naver_merged.csv (from output/)
- Standardize and merge
- Plot:
    1) IMDB vs Naver rating distribution (ratio within platform)
    2) Early vs late normalized per rating (cut by a given cut date)

Usage:
  python rating_phase_analysis.py --output-dir output --cut-date 2020-04-01 --save output/phase_norm.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_reviews_data(imdb: pd.DataFrame, naver: pd.DataFrame) -> pd.DataFrame:
    imdb_std = imdb.copy()
    imdb_std["platform"] = "imdb"
    imdb_std["date"] = pd.to_datetime(imdb_std["date"], errors="coerce")
    imdb_std["rating"] = pd.to_numeric(imdb_std.get("star_rating"), errors="coerce")
    imdb_std["text"] = imdb_std.get("content_clean_nostop", np.nan)
    imdb_std["upvote"] = np.nan
    imdb_std["downvote"] = np.nan
    imdb_std = imdb_std[["id", "platform", "date", "rating", "text", "upvote", "downvote"]]

    naver_std = naver.copy()
    naver_std["platform"] = "naver"
    naver_std["date"] = pd.to_datetime(naver_std["writing_date"], errors="coerce")
    naver_std["rating"] = pd.to_numeric(naver_std.get("star_rating"), errors="coerce")
    naver_std["text"] = naver_std.get("comment", np.nan)
    if "upvote" not in naver_std.columns:
        naver_std["upvote"] = np.nan
    if "downvote" not in naver_std.columns:
        naver_std["downvote"] = np.nan
    naver_std = naver_std[["id", "platform", "date", "rating", "text", "upvote", "downvote"]]

    return pd.concat([imdb_std, naver_std], ignore_index=True).sort_values("date")

def rating_distribution_ratio(reviews_all: pd.DataFrame) -> pd.DataFrame:
    d = reviews_all.dropna(subset=["rating"]).copy()
    d["rating"] = d["rating"].astype(int)
    d = d[(d["rating"] >= 1) & (d["rating"] <= 10)]
    dist = d.groupby(["platform", "rating"]).size().reset_index(name="n")
    dist["prop"] = dist.groupby("platform")["n"].transform(lambda s: s / s.sum())
    return dist

def plot_rating_dist_ratio(dist: pd.DataFrame, ylim: float = 0.6) -> plt.Figure:
    fig = plt.figure()
    platforms = list(dist["platform"].unique())
    x = np.arange(1, 11)
    width = 0.35

    for i, p in enumerate(platforms):
        sub = dist[dist["platform"] == p].set_index("rating").reindex(x).fillna(0)
        offset = (i - (len(platforms) - 1) / 2) * width
        plt.bar(x + offset, sub["prop"].values, width=width, label=p)

    plt.xticks(x, [str(i) for i in x])
    plt.ylim(0, ylim)
    plt.xlabel("rating")
    plt.ylabel("ratio within platform")
    plt.title("IMDB vs Naver rating distribution (ratio per platform)")
    plt.legend()
    plt.tight_layout()
    return fig

def plot_rating_early_late_norm(reviews_all: pd.DataFrame, cut_date: str, platform_filter: str | None = None) -> plt.Figure:
    df = reviews_all.dropna(subset=["rating"]).copy()
    df["period_3m"] = df["date"].dt.to_period("Q").dt.start_time
    cut = pd.to_datetime(cut_date)
    df["phase"] = np.where(df["period_3m"] < cut, "early", "late")
    df["rating"] = df["rating"].astype(int)
    df = df[(df["rating"] >= 1) & (df["rating"] <= 10)]
    if platform_filter:
        df = df[df["platform"] == platform_filter]
    if df.empty:
        fig = plt.figure()
        plt.title("No data after filtering.")
        return fig

    dens_phase = df.groupby(["platform", "phase", "rating"]).size().reset_index(name="n")
    dens_phase["prop_phase"] = dens_phase.groupby(["platform", "phase"])["n"].transform(lambda s: s / s.sum())
    dens_norm = dens_phase.copy()
    dens_norm["prop_norm"] = dens_norm.groupby(["platform", "rating"])["prop_phase"].transform(lambda s: s / s.sum())

    platforms = list(dens_norm["platform"].unique())
    fig, axes = plt.subplots(nrows=len(platforms), ncols=1, figsize=(10, 4 * len(platforms)))
    if len(platforms) == 1:
        axes = [axes]

    for ax, p in zip(axes, platforms):
        subp = dens_norm[dens_norm["platform"] == p]
        for phase, offset in [("early", -0.2), ("late", 0.2)]:
            s = subp[subp["phase"] == phase].set_index("rating").reindex(range(1, 11)).fillna(0)
            ax.bar(np.arange(1, 11) + offset, s["prop_norm"].values, width=0.4, label=phase)
        ax.set_title(f"{p}: Early vs late normalized per rating (cut = {cut_date})")
        ax.set_xlabel("rating")
        ax.set_ylabel("normalized share (early vs late)")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, 11))
        ax.legend()

    plt.tight_layout()
    return fig

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--imdb-file", default="imdb_clean_nostop.csv")
    ap.add_argument("--naver-file", default="reviews_naver_merged.csv")
    ap.add_argument("--cut-date", default="2020-04-01")
    ap.add_argument("--plot", default="phase_norm", choices=["platform_ratio", "phase_norm"])
    ap.add_argument("--platform-filter", default=None, choices=[None, "imdb", "naver"])
    ap.add_argument("--save", default=None, help="If set, save the plot to this PNG path.")
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

    if args.plot == "platform_ratio":
        dist = rating_distribution_ratio(reviews_all)
        fig = plot_rating_dist_ratio(dist)
    else:
        fig = plot_rating_early_late_norm(reviews_all, cut_date=args.cut_date, platform_filter=args.platform_filter)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
