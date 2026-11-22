"""Download the four 10-Ks from SEC EDGAR, extract Items 1A, 7 and 8, and
split them into passages with stable IDs.

Output (committed):
    scenario/filings/{TICKER}/item1a.txt, item7.txt, item8.txt
    scenario/filings/passages.jsonl     one passage per line

Raw downloads go to scenario/raw/ (gitignored).

Usage:
    uv run python scenario/extract.py              # download if needed, then extract
    uv run python scenario/extract.py --refresh    # force re-download

Needs SEC_USER_AGENT in .env ("Your Name you@example.com"). SEC requires it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
OUT = HERE / "filings"

# Must match scenario/TASK.md §2.
FILINGS = {
    "MSFT": {"cik": "789019", "accession": "0001193125-26-323660", "doc": "msft-20260630.htm", "fy": "FY2026"},
    "NKE": {"cik": "320187", "accession": "0000320187-26-000088", "doc": "nke-20260531.htm", "fy": "FY2026"},
    "DUK": {"cik": "1326160", "accession": "0001326160-26-000014", "doc": "duk-20251231.htm", "fy": "FY2025"},
    "F": {"cik": "37996", "accession": "0000037996-26-000015", "doc": "f-20251231.htm", "fy": "FY2025"},
}

# section id -> (start heading, end heading)
SECTIONS = {
    "item1a": (r"ITEM\s*1A\.?", r"ITEM\s*1B\.?"),
    "item7": (r"ITEM\s*7\.?(?!A)", r"ITEM\s*7A\.?"),
    "item8": (r"ITEM\s*8\.?", r"ITEM\s*9\.?(?!A|B|C)"),
}
# Ford's Item 8 only points to the statements, which follow the signature
# pages at the end of the filing. Take the last auditor's report heading
# through the end of the document.
OVERRIDES = {
    ("F", "item8"): r"REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM",
}
MIN_SECTION_CHARS = 5000      # shorter matches are table-of-contents entries
PASSAGE_CHARS = 1500          # target passage size
SEC_DELAY_S = 0.2             # stays well under SEC's 10 requests/second


def download(ticker: str, meta: dict, user_agent: str, refresh: bool) -> Path:
    path = RAW / f"{ticker}.htm"
    if path.exists() and not refresh:
        return path
    acc = meta["accession"].replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{meta['cik']}/{acc}/{meta['doc']}"
    r = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    r.raise_for_status()
    RAW.mkdir(exist_ok=True)
    path.write_bytes(r.content)
    time.sleep(SEC_DELAY_S)
    return path


def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:header")):
        tag.decompose()  # hidden XBRL metadata
    # Tables become one line per row, cells separated by " | ", so the
    # financial statements stay readable as text.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in {"$", ")", "%"}]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def find_section(text: str, start_pat: str, end_pat: str) -> str:
    starts = [m.start() for m in re.finditer(rf"^\s*{start_pat}", text, re.I | re.M)]
    for s in starts:
        end = re.search(rf"^\s*{end_pat}", text[s + 1 :], re.I | re.M)
        if end and end.start() >= MIN_SECTION_CHARS:
            return text[s : s + 1 + end.start()].strip()
    raise ValueError(f"section not found: {start_pat} → {end_pat}")


def split_passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Greedy split on line boundaries, so table rows and paragraphs stay whole."""
    passages, current = [], ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if current and len(current) + len(line) + 1 > size:
            passages.append(current)
            current = ""
        current = f"{current}\n{line}" if current else line
    if current:
        passages.append(current)
    return passages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download from EDGAR")
    args = parser.parse_args()

    load_dotenv(HERE.parent / ".env")
    user_agent = os.getenv("SEC_USER_AGENT", "").strip().strip('"')
    if not user_agent:
        print('Missing SEC_USER_AGENT in .env, e.g. SEC_USER_AGENT="Your Name you@example.com"')
        return 2

    OUT.mkdir(exist_ok=True)
    all_passages = []
    for ticker, meta in FILINGS.items():
        text = html_to_text(download(ticker, meta, user_agent, args.refresh))
        (OUT / ticker).mkdir(exist_ok=True)
        summary = []
        for sec_id, (start, end) in SECTIONS.items():
            if (ticker, sec_id) in OVERRIDES:
                hits = list(re.finditer(rf"^\s*{OVERRIDES[(ticker, sec_id)]}", text, re.I | re.M))
                section = text[hits[-1].start():].strip()
            else:
                section = find_section(text, start, end)
            (OUT / ticker / f"{sec_id}.txt").write_text(section + "\n", encoding="utf-8")
            chunks = split_passages(section)
            for i, chunk in enumerate(chunks, 1):
                all_passages.append({
                    "id": f"{ticker}-{meta['fy']}-{sec_id}-{i:03d}",
                    "ticker": ticker,
                    "fiscal_year": meta["fy"],
                    "section": sec_id,
                    "accession": meta["accession"],
                    "text": chunk,
                })
            summary.append(f"{sec_id} {len(section) // 1000}k chars / {len(chunks)} passages")
        print(f"{ticker:<5} " + " · ".join(summary))

    with open(OUT / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n{len(all_passages)} passages → {OUT / 'passages.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
