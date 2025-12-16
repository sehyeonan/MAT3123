# -*- coding: utf-8 -*-
"""
imdb_stopwords.py

Port of imdb_stopwords.R:
- Build an English stopword list (base + movie-specific)
- Remove stopwords from IMDB cleaned review text
- Save to output/imdb_clean_nostop.csv

Expected input columns (defaults can be changed via CLI):
  - id, star_rating, title, date, site, content_clean

Usage:
  python imdb_stopwords.py --input output/imdb_clean.csv
  python imdb_stopwords.py --input output/imdb_clean.csv --text-col content_clean

Notes:
  - Uses scikit-learn's built-in English stopwords (no NLTK downloads needed).
  - Groups by (id, star_rating, title, date, site) like the R script and concatenates text within-group.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MOVIE_STOPWORDS = [
    # 영화 자체를 가리키는 말
    "movie", "movies", "film", "films", "cinema",
    "series", "season", "seasons", "episode", "episodes", "parasite",

    # 리뷰/시청 관련 표현
    "watch", "watched", "watching", "rewatch", "rewatched",
    "see", "seeing", "saw", "seen",
    "view", "viewed", "viewer", "viewers",

    # 작품 메타 정보
    "director", "directors", "directed",
    "actor", "actors", "actress", "actresses",
    "cast", "crew", "screenplay",

    # 영화 구성 요소
    "scene", "scenes", "shot", "shots",
    "character", "characters",
    "story", "stories", "plot", "subplot",
    "ending", "endings", "opening",

    # 평가용 흔한 단어
    "really", "very", "quite", "lot", "lots",

    # 플랫폼/리뷰 사이트
    "imdb", "netflix", "hbo", "disney", "marvel",
    "dvd", "blu", "bluray",

    # 스포일러 관련
    "spoiler", "spoilers",

    # 기타 자주 나오지만 의미 약한 표현
    "thing", "things", "kind", "sort",
]

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")

def tokenize_en(text: str) -> list[str]:
    text = (text or "").lower()
    return _WORD_RE.findall(text)

def remove_stopwords(tokens: list[str], stopset: set[str]) -> list[str]:
    return [t for t in tokens if t not in stopset]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="output/imdb_clean.csv", help="Input CSV path (imdb_clean).")
    ap.add_argument("--output", default="output/imdb_clean_nostop.csv", help="Output CSV path.")
    ap.add_argument("--text-col", default="content_clean", help="Text column to clean.")
    ap.add_argument(
        "--group-cols",
        default="id,star_rating,title,date,site",
        help="Comma-separated columns to define a 'review unit' (R script uses id, star_rating, title, date, site).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path.resolve()}")

    df = pd.read_csv(in_path)

    text_col = args.text_col
    if text_col not in df.columns:
        raise KeyError(f"Text column '{text_col}' not found. Available columns: {list(df.columns)}")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in df.columns:
            raise KeyError(f"Group column '{c}' not found. Available columns: {list(df.columns)}")

    stopset = set(ENGLISH_STOP_WORDS)
    stopset.update(w.lower() for w in MOVIE_STOPWORDS)

    grouped = (
        df.assign(**{text_col: df[text_col].fillna("").astype(str)})
          .groupby(group_cols, dropna=False, as_index=False)[text_col]
          .agg(lambda s: " ".join(s))
    )

    def _clean_row(t: str) -> str:
        toks = tokenize_en(t)
        toks = remove_stopwords(toks, stopset)
        return " ".join(toks)

    grouped["content_clean_nostop"] = grouped[text_col].map(_clean_row)

    out_df = grouped[group_cols + ["content_clean_nostop"]]
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Saved: {out_path.resolve()} (rows={len(out_df):,})")

if __name__ == "__main__":
    main()
