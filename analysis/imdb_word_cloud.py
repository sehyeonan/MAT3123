# -*- coding: utf-8 -*-
"""
imdb_word_cloud.py

Port of imdb_word_cloud.R:
- Build word-frequency table from IMDB 'nostop' text
- Plot a basic wordcloud
- Plot wordclouds by rating range
- Compute rating distribution for a selected keyword and plot it

Inputs:
  - output/imdb_clean_nostop.csv (default), with columns:
      - id, star_rating, content_clean_nostop

Usage examples:
  python imdb_word_cloud.py --mode basic
  python imdb_word_cloud.py --mode pos --min-rating 8
  python imdb_word_cloud.py --mode neg --max-rating 3
  python imdb_word_cloud.py --mode keyword-dist --keyword boring
  python imdb_word_cloud.py --mode basic --out-png output/imdb_wordcloud.png
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_en(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())

def make_word_freq(df: pd.DataFrame, text_col: str = "content_clean_nostop", min_chars: int = 2) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for t in df[text_col].fillna("").astype(str):
        for w in tokenize_en(t):
            if len(w) >= min_chars:
                counts[w] = counts.get(w, 0) + 1
    out = pd.DataFrame({"word": list(counts.keys()), "n": list(counts.values())})
    return out.sort_values("n", ascending=False).reset_index(drop=True)

def get_word_freq_by_rating(
    df: pd.DataFrame,
    min_rating: float | None = None,
    max_rating: float | None = None,
    text_col: str = "content_clean_nostop",
    rating_col: str = "star_rating",
    min_chars: int = 2,
) -> pd.DataFrame:
    data = df.copy()
    if min_rating is not None:
        data = data[data[rating_col] >= min_rating]
    if max_rating is not None:
        data = data[data[rating_col] <= max_rating]
    if data.empty:
        return pd.DataFrame({"word": [], "n": []})
    return make_word_freq(data, text_col=text_col, min_chars=min_chars)

def plot_wordcloud_from_freq(
    freq_tbl: pd.DataFrame,
    max_words: int = 200,
    min_freq: int = 3,
    width: int = 1200,
    height: int = 800,
    random_state: int = 123,
) -> plt.Figure:
    if freq_tbl.empty:
        raise ValueError("No words to plot. Check rating filters or input data.")
    freq_tbl = freq_tbl[freq_tbl["n"] >= min_freq].head(max_words)
    wc = WordCloud(width=width, height=height, background_color="white", random_state=random_state)
    wc.generate_from_frequencies(dict(zip(freq_tbl["word"], freq_tbl["n"])))
    fig = plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig

def get_top_keywords(
    df: pd.DataFrame,
    top_n: int = 100,
    text_col: str = "content_clean_nostop",
    min_chars: int = 3,
    min_freq: int = 10,
) -> pd.DataFrame:
    freq = make_word_freq(df, text_col=text_col, min_chars=min_chars)
    freq = freq[freq["n"] >= min_freq]
    return freq.head(top_n).reset_index(drop=True)

def rating_dist_for_keyword(
    df: pd.DataFrame,
    keyword: str,
    rating_col: str = "star_rating",
    text_col: str = "content_clean_nostop",
) -> pd.DataFrame:
    keyword = (keyword or "").lower().strip()
    if not keyword:
        return pd.DataFrame({"star_rating": [], "n": [], "prop": []})

    def has_kw(text: str) -> bool:
        return keyword in set(tokenize_en(text))

    tmp = df[["id", rating_col, text_col]].copy()
    tmp["has_kw"] = tmp[text_col].fillna("").astype(str).map(has_kw)
    tmp = tmp[tmp["has_kw"]].dropna(subset=[rating_col])
    if tmp.empty:
        return pd.DataFrame({"star_rating": [], "n": [], "prop": []})

    tmp = tmp.drop_duplicates(subset=["id", rating_col])
    dist = tmp.groupby(rating_col).size().reset_index(name="n").sort_values(rating_col)
    dist["prop"] = dist["n"] / dist["n"].sum()
    dist.rename(columns={rating_col: "star_rating"}, inplace=True)
    return dist

def plot_rating_dist_for_keyword(df: pd.DataFrame, keyword: str) -> plt.Figure:
    dist = rating_dist_for_keyword(df, keyword)
    if dist.empty:
        raise ValueError(f"No reviews contain keyword='{keyword}'.")
    fig = plt.figure()
    plt.bar(dist["star_rating"].astype(str), dist["n"])
    plt.title(f"Rating distribution for keyword: '{keyword}'")
    plt.xlabel("Star rating")
    plt.ylabel("Number of reviews")
    return fig

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/imdb_clean_nostop.csv", help="Input CSV (imdb_clean_nostop).")
    ap.add_argument(
        "--mode",
        default="basic",
        choices=["basic", "pos", "neg", "keyword-dist", "top-keywords"],
        help="What to run.",
    )
    ap.add_argument("--min-rating", type=float, default=None, help="Min rating for pos mode.")
    ap.add_argument("--max-rating", type=float, default=None, help="Max rating for neg mode.")
    ap.add_argument("--keyword", default="boring", help="Keyword for keyword-dist mode.")
    ap.add_argument("--out-png", default=None, help="If set, save the plot to this PNG path.")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=200)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    df = pd.read_csv(in_path)
    if "star_rating" in df.columns:
        df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")

    fig = None

    if args.mode == "basic":
        freq = make_word_freq(df)
        fig = plot_wordcloud_from_freq(freq, max_words=args.max_words, min_freq=args.min_freq)
    elif args.mode == "pos":
        freq = get_word_freq_by_rating(df, min_rating=args.min_rating if args.min_rating is not None else 8)
        fig = plot_wordcloud_from_freq(freq, max_words=args.max_words, min_freq=args.min_freq)
    elif args.mode == "neg":
        freq = get_word_freq_by_rating(df, max_rating=args.max_rating if args.max_rating is not None else 3)
        fig = plot_wordcloud_from_freq(freq, max_words=args.max_words, min_freq=args.min_freq)
    elif args.mode == "keyword-dist":
        fig = plot_rating_dist_for_keyword(df, args.keyword)
    elif args.mode == "top-keywords":
        top = get_top_keywords(df)
        print(top.head(20).to_string(index=False))
        return

    if fig is None:
        return

    if args.out_png:
        out_path = Path(args.out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
