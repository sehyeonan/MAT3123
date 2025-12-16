# -*- coding: utf-8 -*-
"""
naver_word_cloud.py

Port of naver_word_cloud.R (Korean wordcloud + rating-conditional word freq + keyword rating distribution).

Input:
  - output/reviews_naver_merged.csv (default), with columns:
      - id, star_rating, normalized_text

Important:
  - For proper Korean rendering in WordCloud, you usually need a Korean TTF/OTF font.
    Provide --font-path if auto-detection fails.

Usage examples:
  python naver_word_cloud.py --mode pos --min-rating 9 --out-png output/naver_pos.png --font-path /path/to/NanumGothic.ttf
  python naver_word_cloud.py --mode neg --max-rating 3 --out-png output/naver_neg.png --font-path /path/to/NanumGothic.ttf
  python naver_word_cloud.py --mode keyword-dist --keyword 연기
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from matplotlib import font_manager

DEFAULT_STOPWORDS_KR = ["영화", "보다", "있다", "없다", "너무", "같다"]

_KOREAN_RE = re.compile(r"[가-힣]+")

def find_korean_font_path() -> str | None:
    """Try to locate a Korean font on the system. Returns a font file path or None."""
    candidates: list[str] = []
    try:
        candidates.extend(font_manager.findSystemFonts(fontext="ttf"))
        candidates.extend(font_manager.findSystemFonts(fontext="otf"))
    except Exception:
        return None

    preferred = ["NanumGothic", "Malgun", "AppleGothic", "NotoSansCJK", "NotoSansKR"]
    for p in candidates:
        name = Path(p).stem.lower()
        if any(pref.lower() in name for pref in preferred):
            return p
    return candidates[0] if candidates else None

def tokenize_kr(text: str, keep_korean_only: bool = True) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    toks = re.split(r"\s+", text)
    if keep_korean_only:
        toks = [t for t in toks if _KOREAN_RE.search(t)]
    return [t for t in toks if t]

def get_word_freq_by_rating(
    df: pd.DataFrame,
    min_rating: float | None = None,
    max_rating: float | None = None,
    min_chars: int = 2,
    keep_korean_only: bool = True,
    stopwords: list[str] | None = None,
    text_col: str = "normalized_text",
    rating_col: str = "star_rating",
) -> pd.DataFrame:
    data = df.copy()
    if min_rating is not None:
        data = data[data[rating_col] >= min_rating]
    if max_rating is not None:
        data = data[data[rating_col] <= max_rating]
    if data.empty:
        return pd.DataFrame({"word": [], "n": []})

    stopset = set(stopwords or [])
    counts: dict[str, int] = {}
    for t in data[text_col].fillna("").astype(str):
        for w in tokenize_kr(t, keep_korean_only=keep_korean_only):
            if len(w) < min_chars:
                continue
            if w in stopset:
                continue
            counts[w] = counts.get(w, 0) + 1

    out = pd.DataFrame({"word": list(counts.keys()), "n": list(counts.values())})
    return out.sort_values("n", ascending=False).reset_index(drop=True)

def plot_wordcloud_from_freq(
    freq_tbl: pd.DataFrame,
    font_path: str | None = None,
    max_words: int = 200,
    min_freq: int = 3,
    width: int = 1400,
    height: int = 900,
    random_state: int = 123,
) -> plt.Figure:
    if freq_tbl.empty:
        raise ValueError("No words to plot. Check rating filters or input data.")
    freq_tbl = freq_tbl[freq_tbl["n"] >= min_freq].head(max_words)
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        random_state=random_state,
        font_path=font_path,
    )
    wc.generate_from_frequencies(dict(zip(freq_tbl["word"], freq_tbl["n"])))
    fig = plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig

def rating_dist_for_keyword(
    df: pd.DataFrame,
    keyword: str,
    stopwords: list[str] | None = None,
    min_chars: int = 1,
    text_col: str = "normalized_text",
    rating_col: str = "star_rating",
) -> pd.DataFrame:
    keyword = (keyword or "").strip()
    if not keyword:
        return pd.DataFrame({"star_rating": [], "n": [], "prop": []})

    stopset = set(stopwords or [])

    def has_kw(text: str) -> bool:
        toks = set(tokenize_kr(text, keep_korean_only=False))
        toks = {t for t in toks if len(t) >= min_chars and t not in stopset}
        return keyword in toks

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

def plot_rating_dist_for_keyword(df: pd.DataFrame, keyword: str, stopwords: list[str] | None = None) -> plt.Figure:
    dist = rating_dist_for_keyword(df, keyword, stopwords=stopwords)
    if dist.empty:
        raise ValueError(f"No reviews contain keyword='{keyword}'.")
    fig = plt.figure()
    plt.bar(dist["star_rating"].astype(str), dist["prop"])
    plt.title(f"Rating distribution for keyword: '{keyword}'")
    plt.xlabel("Star rating")
    plt.ylabel("Proportion")
    plt.ylim(0, 1)
    return fig

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/reviews_naver_merged.csv", help="Input CSV path.")
    ap.add_argument("--mode", default="pos", choices=["pos", "neg", "basic", "keyword-dist"])
    ap.add_argument("--min-rating", type=float, default=9)
    ap.add_argument("--max-rating", type=float, default=3)
    ap.add_argument("--keyword", default="연기")
    ap.add_argument("--font-path", default=None, help="Korean font file path (ttf/otf).")
    ap.add_argument("--out-png", default=None, help="If set, save the plot to this PNG path.")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=200)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    df = pd.read_csv(in_path, encoding="utf-8")
    if "star_rating" in df.columns:
        df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")

    font_path = args.font_path or find_korean_font_path()
    if font_path is None and args.mode in ("basic", "pos", "neg"):
        print("WARNING: Could not auto-detect a Korean font. WordCloud may render squares. "
              "Pass --font-path to fix.")

    if args.mode == "keyword-dist":
        fig = plot_rating_dist_for_keyword(df, args.keyword, stopwords=DEFAULT_STOPWORDS_KR)
    else:
        if args.mode == "basic":
            freq = get_word_freq_by_rating(df, stopwords=DEFAULT_STOPWORDS_KR)
        elif args.mode == "pos":
            freq = get_word_freq_by_rating(df, min_rating=args.min_rating, stopwords=DEFAULT_STOPWORDS_KR)
        else:  # neg
            freq = get_word_freq_by_rating(df, max_rating=args.max_rating, stopwords=DEFAULT_STOPWORDS_KR)
        fig = plot_wordcloud_from_freq(freq, font_path=font_path, max_words=args.max_words, min_freq=args.min_freq)

    if args.out_png:
        out_path = Path(args.out_png)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {out_path.resolve()}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
