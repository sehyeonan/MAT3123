import csv
import re
import time
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from .base import BaseCrawler


class ImdbReviewCrawler(BaseCrawler):
    def __init__(
        self,
        review_url: str,
        output_dir: str,
        max_reviews: int = 500,
        driver_path: Optional[str] = None,
        wait_timeout: int = 15,
    ):
        super().__init__(output_dir)
        self.review_url = review_url
        self.max_reviews = max_reviews
        self.driver_path = driver_path
        self.wait_timeout = wait_timeout

        self.driver: Optional[webdriver.Chrome] = None
        self.wait: Optional[WebDriverWait] = None
        self.rows: List[List[str]] = []
        self.filename: str = ""

    def start_browser(self):
        if self.driver:
            return self.driver

        service = Service(self.driver_path) if self.driver_path else Service()
        self.driver = webdriver.Chrome(service=service)
        self.wait = WebDriverWait(self.driver, self.wait_timeout)
        return self.driver

    def scrape_reviews(self):
        if not self.driver or not self.wait:
            self.start_browser()

        assert self.driver is not None  # for type checkers
        assert self.wait is not None

        self.driver.get(self.review_url)
        time.sleep(3)

        self._click_see_all()
        title = self._extract_movie_title()
        self.filename = self._make_filename(title)

        self._scroll_until_limit()
        self.rows = self._collect_reviews()
        return self.rows

    def save_to_database(self):
        if not self.rows:
            raise ValueError("No reviews scraped yet; run scrape_reviews() first.")

        target_dir = Path(self.output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / (self.filename or "imdb_reviews.csv")

        with target_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["작성자", "제목", "별점", "내용"])
            writer.writerows(self.rows)

        return target_path

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None
            self.wait = None

    def _click_see_all(self):
        try:
            see_all_btn = self.wait.until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        "//span[contains(@class,'ipc-see-more__text') "
                        "and normalize-space()='See all']/ancestor::button",
                    )
                )
            )
            self.driver.execute_script("arguments[0].click();", see_all_btn)
            time.sleep(3)
        except Exception:
            # 버튼이 없거나 이미 전체 보기 상태일 수 있음
            pass

    def _extract_movie_title(self) -> str:
        title_text = ""
        try:
            subtitle_el = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'h2[data-testid="subtitle"]')
                )
            )
            title_text = subtitle_el.text.strip()
        except Exception:
            try:
                h1_el = self.driver.find_element(
                    By.CSS_SELECTOR, 'h1[data-testid="hero-title-block__title"]'
                )
                title_text = h1_el.text.strip()
            except Exception:
                title_text = self.driver.title.split("-")[0].strip()
        return title_text

    def _make_filename(self, movie_title: str) -> str:
        safe_title = re.sub(r'[\\/*?:"<>|]', "", movie_title).replace(" ", "_")
        return f"{safe_title}_imdb_reviews.csv"

    def _scroll_until_limit(self):
        prev_count = -1
        same_rounds = 0

        while True:
            cards = self.driver.find_elements(By.CSS_SELECTOR, "article.user-review-item")
            cur_count = len(cards)

            if cur_count >= self.max_reviews:
                break

            if cur_count == prev_count:
                same_rounds += 1
                if same_rounds >= 3:
                    break
            else:
                same_rounds = 0
                prev_count = cur_count

            self.driver.execute_script(
                "window.scrollBy(0, document.body.scrollHeight * 0.7);"
            )
            time.sleep(2)

    def _collect_reviews(self) -> List[List[str]]:
        html = self.driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        review_articles = soup.select("article.user-review-item")

        rows: List[List[str]] = []
        for art in review_articles[: self.max_reviews]:
            rating = ""
            rating_span = art.select_one("span[aria-label*='rating:']")
            if rating_span and rating_span.has_attr("aria-label"):
                aria = rating_span["aria-label"]
                rating_match = re.match(r"^(.*?)'s rating:\s*([0-9]+)", aria)
                if rating_match:
                    rating = rating_match.group(2).strip()

            author = ""
            author_link = art.select_one(
                'div[data-testid="reviews-author"] a[data-testid="author-link"]'
            )
            if author_link:
                author = author_link.get_text(strip=True)
            elif rating_span and rating_span.has_attr("aria-label"):
                aria = rating_span["aria-label"]
                author_match = re.match(r"^(.*?)'s rating:\s*([0-9]+)", aria)
                if author_match:
                    author = author_match.group(1).strip()

            title = ""
            title_el = art.select_one('[data-testid="review-summary"] h3') or art.select_one(
                "h3"
            )
            if title_el:
                title = title_el.get_text(strip=True)

            text = ""
            text_el = art.select_one('[data-testid="review-overflow"]')
            if text_el:
                text = " ".join(text_el.get_text(separator=" ").split())

            if not text:
                continue

            rows.append([author, title, rating, text])

        return rows
