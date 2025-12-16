import argparse
import os

from .imdb_crawler import ImdbReviewCrawler
from .naver_crawler import NaverReviewCrawler

DEFAULT_IMDB_URL = (
    os.getenv("IMDB_REVIEW_URL")
    or "https://www.imdb.com/title/tt0332280/reviews/?ref_=tt_ururv_genai_sm"
)
DEFAULT_NAVER_URL = (
    os.getenv("NAVER_REVIEW_URL")
    or "https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&ssc=tab.nx.all"
    "&query=%EC%98%81%ED%99%94+%EB%85%B8%ED%8A%B8%EB%B6%81&oquery=%EC%98%81%ED%95%98+%EB%85%B8%ED%8A%B8%EB%B6%81"
    "&tqi=jfww9wqVW9hssk8Q31R-271403&ackey=rwxorky9"
)
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR") or "output"
DEFAULT_DRIVER_PATH = os.getenv("CHROMEDRIVER_PATH") or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IMDb/Naver review crawler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="CSV 저장 폴더",
    )

    subparsers = parser.add_subparsers(dest="site", required=True)

    imdb_parser = subparsers.add_parser("imdb", help="IMDb 리뷰 크롤링")
    imdb_parser.add_argument("--url", default=DEFAULT_IMDB_URL, help="IMDb 리뷰 페이지 URL")
    imdb_parser.add_argument(
        "--max-reviews", type=int, default=500, help="최대 크롤링 리뷰 개수"
    )

    naver_parser = subparsers.add_parser("naver", help="네이버 관람평 크롤링")
    naver_parser.add_argument("--url", default=DEFAULT_NAVER_URL, help="네이버 검색 URL")
    naver_parser.add_argument(
        "--scroll-rounds", type=int, default=30, help="내부 스크롤 반복 횟수"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.site == "imdb":
        crawler = ImdbReviewCrawler(
            review_url=args.url,
            output_dir=args.output_dir,
            max_reviews=args.max_reviews,
        )
    else:
        crawler = NaverReviewCrawler(
            review_url=args.url,
            output_dir=args.output_dir,
            scroll_rounds=args.scroll_rounds,
        )

    try:
        crawler.start_browser()
        crawler.scrape_reviews()
        csv_path = crawler.save_to_database()
        print(f"CSV 저장 완료 → {csv_path}")
    finally:
        crawler.close()


if __name__ == "__main__":
    main()
