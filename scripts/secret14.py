# secret14.py  – v3: normalised prefix so we hit /portfolio_5_…
#
# Szuka brakujących numerów w sekwencjach   /prefix_N_<md5(N)>
# i odwiedza wygenerowane strony, aby znaleźć {{FLG:…}}.

from __future__ import annotations
import hashlib, re, sys, requests
from urllib.parse import urljoin, urlparse
from collections import defaultdict, deque
from bs4 import BeautifulSoup

BASE   = "https://softo.ag3nts.org"
SESS   = requests.Session()
SESS.headers["User-Agent"] = "secret14/3.0"
FLAG_RE = re.compile(r"\{\{FLG:([^}]+)\}\}", re.I)

#              /prefix_      N     _  32-hex
PAT = re.compile(r"(?P<prefix>/.+?_)(?P<num>\d+)_(?P<hash>[0-9a-f]{32})", re.I)

def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def dbg(msg: str): print("[DBG]", msg)

def get(url: str) -> str | None:
    try:
        dbg(f"GET {url}")
        r = SESS.get(url, timeout=10)
        r.raise_for_status()
        return r.text
    except Exception as e:
        dbg(f"  !! {e}")
        return None

def scan_flag(txt: str, ctx: str) -> bool:
    m = FLAG_RE.search(txt)
    if m:
        print(f"\n>>> FLAG FOUND in {ctx}: {m.group(0)}\n")
        return True
    return False

def extract_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [
        urljoin(base, a["href"])
        for a in soup.find_all("a", href=True)
        if urlparse(urljoin(base, a["href"])).netloc.endswith("softo.ag3nts.org")
    ]

# -- NEW -----------------------------------------------------------------
def normalise_prefix(raw: str) -> str:
    """
    Zwraca samą ścieżkę (z wiodącym /), niezależnie od tego, czy
    w HTML występowało  '/portfolio_'  czy  '//softo.ag3nts.org/portfolio_'.
    """
    if raw.startswith("//"):
        raw = "http:" + raw            # urlparse wymaga schematu
    parsed = urlparse(raw)
    return parsed.path                # zawsze zaczyna się od '/'

# -----------------------------------------------------------------------
def crawl(start: str):
    seen, q = set(), deque([start])
    groups: defaultdict[str, set[int]] = defaultdict(set)

    while q:
        url = q.popleft()
        if url in seen:
            continue
        seen.add(url)

        html = get(url)
        if not html:
            continue
        if scan_flag(html, url):
            return

        # zbieramy numery
        for m in PAT.finditer(html):
            pref = normalise_prefix(m["prefix"])
            groups[pref].add(int(m["num"]))

        # kolejne linki
        q.extend(l for l in extract_links(html, url) if l not in seen)

    # -- generujemy brakujące -------------------------------------------------
    for path, nums in groups.items():
        missing = sorted(set(range(1, max(nums) + 1)) - nums)
        for n in missing:
            cand = f"{BASE}{path}{n}_{md5(str(n))}"
            html = get(cand)
            if html and scan_flag(html, cand):
                return
    print("\nScan finished – flag not found.")


if __name__ == "__main__":
    crawl(sys.argv[1] if len(sys.argv) > 1 else BASE)
