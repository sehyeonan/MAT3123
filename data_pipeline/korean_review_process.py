#!/usr/bin/env python3
"""
한국어 리뷰를 Kiwi 형태소 분석으로 정제하는 스크립트.
- CSV에서 id와 텍스트를 읽어 명사/동사/형용사/부사만 뽑아 기본형으로 정렬
- 불용어 제거 (기본 내장 + 선택적 data_pipeline/stopwords.txt)
- 결과는 처리 즉시 id, original_text, normalized_text 컬럼으로 저장

필수:
- 입력 CSV에 id 컬럼 필수, 텍스트 컬럼은 comment/content/review/text 중 하나
- pip install kiwipiepy
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

from kiwipiepy import Kiwi

# 보수적인 기본 불용어 (필요 시 data_pipeline/stopwords.txt로 확장)
DEFAULT_STOPWORDS = {
    "하다",
    "되다",
    "이다",
}

# Nouns/verbs/adjectives/adverbs
KEEP_TAG_PREFIXES = ("NN", "VV", "VA", "VCP", "VCN", "MA")

STOPWORDS_PATH = Path(__file__).resolve().parent / "stopwords.txt"


def load_stopwords() -> set:
  stopwords = set(DEFAULT_STOPWORDS)
  if STOPWORDS_PATH.exists():
    with STOPWORDS_PATH.open(encoding="utf-8") as f:
      for line in f:
        word = line.strip()
        if word:
          stopwords.add(word)
  return stopwords


def count_rows(path: str) -> int:
  with open(path, newline="", encoding="utf-8") as f:
    return max(0, sum(1 for _ in f) - 1)  # minus header


def iter_rows(path: str) -> Iterable[Tuple[str, str]]:
  with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if "id" not in reader.fieldnames:
      raise ValueError(f"{path} 에 id 컬럼이 없습니다.")

    text_cols = ["comment", "content", "review", "text"]
    text_col = next((c for c in text_cols if c in reader.fieldnames), None)
    if not text_col:
      raise ValueError(f"{path} 에서 텍스트 컬럼을 찾지 못했습니다. 후보: {', '.join(text_cols)}")

    for row in reader:
      yield row["id"], row[text_col]


def normalize_text(kiwi: Kiwi, text: str, stopwords: set) -> str:
  if text is None:
    text = ""

  analyses = kiwi.analyze(text, top_n=1)
  if not analyses:
    return ""

  tokens = analyses[0][0]
  kept = []

  for token in tokens:
    tag = token.tag or ""
    if not tag.startswith(KEEP_TAG_PREFIXES):
      continue

    lemma = getattr(token, "lemma", None) or token.form
    lemma = lemma.strip()
    if not lemma:
      continue
    if lemma in stopwords:
      continue

    kept.append(lemma)

  return " ".join(kept)


def process_file(input_path: str, output_path: str) -> None:
  kiwi = Kiwi()
  stopwords = load_stopwords()

  total = count_rows(input_path)
  written = 0

  with open(output_path, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["id", "original_text", "normalized_text"])

    for idx, (row_id, text) in enumerate(iter_rows(input_path), 1):
      normalized = normalize_text(kiwi, text, stopwords)
      writer.writerow([row_id, text, normalized])
      out_f.flush()
      written += 1

      pct = (idx / total) * 100 if total else 0
      print(f"[진행상황] {idx}/{total} ({pct:0.1f}%) 처리", file=sys.stderr, flush=True)

  print(f"완료: {written}행을 {output_path} 에 저장했습니다.")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Kiwi 기반 한국어 리뷰 정제 (명사/동사 기본형 추출).")
  parser.add_argument("--input", default=os.path.join("output", "reviews_naver.csv"), help="입력 CSV 경로")
  parser.add_argument("--output", default=os.path.join("output", "reviews_naver_normalized.csv"), help="결과 CSV 경로")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  process_file(args.input, args.output)


if __name__ == "__main__":
  main()
