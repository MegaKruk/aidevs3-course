"""
SoftoNavigator – LLM-guided mini-agent for navigating https://softo.ag3nts.org
and extracting concise answers to given questions.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ai_agents_framework import LLMClient, HttpClient


class SoftoNavigator:
    """A reusable, site-specific navigation helper."""

    def __init__(
        self,
        llm_client: LLMClient,
        base_site: str = "https://softo.ag3nts.org",
        max_steps: int = 10,
    ) -> None:
        self.llm = llm_client
        self.base_site = base_site.rstrip("/")
        self.max_steps = max_steps
        self.http = HttpClient()
        self.cache: Dict[str, Tuple[str, List[str]]] = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_html(html: str, base: str) -> Tuple[str, List[str]]:
        """
        Convert HTML to plaintext and collect on-site links.

        • Visible anchor text is preserved.
        • Each anchor gets its absolute href appended in parentheses, so
          the LLM can see real URLs inside the excerpt.
        """
        soup = BeautifulSoup(html, "html.parser")

        # expand <a> tags:  text  →  "text (url)"
        for a in soup.find_all("a", href=True):
            href = urljoin(base, a["href"].strip())
            if href:
                a.append(f" ({href})")

        # drop scripts, styles …
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()

        links: List[str] = []
        for a in soup.find_all("a", href=True):
            abs_url = urljoin(base, a["href"].strip())
            if urlparse(abs_url).netloc.endswith("softo.ag3nts.org"):
                links.append(abs_url)

        # de-duplicate, keep order
        seen: set[str] = set()
        uniq = [u for u in links if not (u in seen or seen.add(u))]
        return text, uniq

    def _fetch(self, url: str) -> Tuple[str, List[str]]:
        """Download + parse page with caching."""
        if url in self.cache:
            return self.cache[url]
        resp = self.http.get(url, timeout=15)
        resp.raise_for_status()
        txt, links = self._strip_html(resp.text, url)
        self.cache[url] = (txt, links)
        return txt, links

    # ------------------------------------------------------------------
    def _decision(self, question: str, page: str, links: List[str]) -> Tuple[str, str]:
        """
        Ask the LLM whether *page* answers *question*.
        Returns (answer, next_link) – exactly one of them is non-empty.
        """
        system_prompt = (
            "You are a web-navigation assistant. Your job is to answer ONE user "
            "question by analysing the given page excerpt and, if necessary, "
            "selecting exactly one link that is most likely to contain the answer.\n\n"
            "If the answer is present already, reply:\n"
            "ANSWER: <concise answer>\n"
            "Otherwise reply:\n"
            "LINK: <one url from the provided list>\n\n"
            "Always reply with ONE line; do not add explanations."
        )

        user_prompt = (
            f"QUESTION:\n{question}\n\n"
            f"PAGE EXCERPT:\n{page[:3500]}\n\n"
            f"LINKS JSON: {json.dumps(links[:15], ensure_ascii=False)}"
        )

        raw = self.llm.answer_with_context(
            question=user_prompt,
            context=system_prompt,
            model="gpt-4.1-mini",
            max_tokens=100,
        ).strip()

        # ----------------------------------------
        # strict formats
        if raw.upper().startswith("ANSWER:"):
            return raw[7:].strip(), ""
        if raw.upper().startswith("LINK:"):
            return "", raw[5:].strip()

        # ----------------------------------------
        # heuristics: plain answer?
        # very short email / url / 2-item ISO list etc.
        if "@" in raw or raw.startswith("http"):
            return raw.split()[0].strip(), ""

        return "", ""

    # ------------------------------------------------------------------
    def solve(self, question: str) -> str:
        """
        Try to answer *question* by guided navigation.
        Returns empty string if the answer was not found within *max_steps* hops.
        """
        queue: List[str] = [self.base_site]
        visited: set[str] = set()

        for _ in range(self.max_steps):
            if not queue:
                break
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                txt, links = self._fetch(url)
            except Exception:
                continue

            answer, next_link = self._decision(question, txt, links)
            if answer:
                return answer
            if next_link and next_link not in visited:
                queue.append(next_link)

        return ""
