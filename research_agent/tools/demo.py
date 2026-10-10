"""Offline demo toolkit: fake search results + synthetic downloads (no internet needed)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .toolkit import Toolkit

LOREM = (
    "The adoption of this technology has accelerated over the last decade, driven by falling costs, supportive policy "
    "and improving performance. Studies report annual growth of around 12 percent, with notable regional differences. "
    "Barriers include financing constraints, skills shortages and infrastructure gaps. Evidence from multiple countries "
    "suggests that coordinated public and private investment is the most reliable accelerator. "
)


def _make_pdf(path: Path, title: str) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    st = getSampleStyleSheet()
    story = [Paragraph(title, st["Title"]), Spacer(1, 12)]
    for i in range(8):
        story.append(Paragraph(f"Section {i+1}. " + LOREM * 3, st["Normal"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


class DemoToolkit(Toolkit):
    def _item(self, i: int, kind: str, origin: str, q: str) -> dict:
        base = dict(title=f"{q.title()} - {kind} source {i}", snippet=LOREM[:300], kind=kind, origin=origin,
                    pdf_url="", download_url="", year=str(2015 + i % 10), authors="A. Researcher, B. Scholar")
        slug = q.replace(" ", "-")[:30]
        if kind == "paper":
            base.update(url=f"demo://paper/{slug}/{i}", pdf_url=f"demo://pdf/{slug}/{i}.pdf")
        elif kind == "dataset":
            base.update(url=f"demo://dataset/{slug}/{i}", download_url=f"demo://data/{slug}/{i}.csv")
        else:
            base.update(url=f"demo://web/{slug}/{i}", year="")
        return base

    def web_search(self, query, n=8, filetype=None):
        if filetype in ("csv", "xlsx"):
            return [self._item(i, "dataset", "demo:" + filetype, query) for i in range(1, 3)]
        out = [self._item(i, "web", "demo", query) for i in range(1, 5)]
        out.append({**self._item(9, "web", "demo", query), "url": f"demo://pdf/{query[:20]}/report.pdf",
                    "pdf_url": f"demo://pdf/{query[:20]}/report.pdf"})
        return out

    def arxiv(self, query, n=6):
        return [self._item(i, "paper", "arxiv", query) for i in range(1, 4)]

    def openalex(self, query, n=6):
        return [self._item(i + 10, "paper", "openalex", query) for i in range(1, 4)]

    def wikipedia(self, query, n=3):
        return [{**self._item(1, "wiki", "wikipedia", query), "title": f"Wikipedia: {query.title()}"}]

    def ckan(self, query, n=6, base=""):
        return [self._item(i + 20, "dataset", "data.gov", query) for i in range(1, 3)]

    def zenodo(self, query, n=5):
        return [self._item(i + 30, "dataset", "zenodo", query) for i in range(1, 2)]

    def worldbank(self, indicator, countries="all", date="2000:2023"):
        rng = np.random.default_rng(7)
        rows = []
        for c, base in [("United States", 3.3e8), ("China", 1.3e9), ("India", 1.1e9), ("Germany", 8.0e7), ("Brazil", 2.0e8), ("Ethiopia", 8.0e7)]:
            for y in range(2000, 2024):
                rows.append({"country": c, "iso3": c[:3].upper(), "year": y, "indicator": "Population, total",
                             "value": base * (1 + 0.012 * (y - 2000)) * (1 + rng.normal(0, 0.003))})
        return pd.DataFrame(rows)

    def download(self, url, dest_dir, max_mb=None, prefix=""):
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        rnd = random.Random(url)
        name = url.rstrip("/").split("/")[-1] or "file"
        if url.startswith("demo://pdf/") or name.endswith(".pdf"):
            p = dest / (prefix + (name if name.endswith(".pdf") else name + ".pdf"))
            _make_pdf(p, "Demo research document: " + name)
            return {"path": str(p), "content_type": "application/pdf", "size": p.stat().st_size, "kind": "pdf"}
        if url.startswith("demo://data/"):
            p = dest / (prefix + (name if name.endswith(".csv") else name + ".csv"))
            rng = np.random.default_rng(abs(hash(url)) % 2**32)
            rows = []
            for region in ["North", "South", "East", "West", "Central"]:
                level = rnd.uniform(20, 80)
                for y in range(2010, 2024):
                    cost = max(5, 100 - 4.5 * (y - 2010) + rng.normal(0, 3))
                    rows.append({"year": y, "region": region, "capacity_gw": round(level * (1.18 ** (y - 2010)) / 10 + rng.normal(0, 1), 2),
                                 "unit_cost": round(cost, 2), "investment_musd": round(rng.uniform(50, 500) * (1 + 0.05 * (y - 2010)), 1)})
            pd.DataFrame(rows).to_csv(p, index=False)
            return {"path": str(p), "content_type": "text/csv", "size": p.stat().st_size, "kind": "csv"}
        # generic web page
        p = dest / (prefix + _safe(name) + ".html")
        body = "".join(f"<p>{LOREM}</p>" for _ in range(8))
        p.write_text(f"<html><head><title>Demo page {name}</title></head><body><article><h1>Demo page</h1>{body}</article></body></html>", encoding="utf-8")
        return {"path": str(p), "content_type": "text/html", "size": p.stat().st_size, "kind": "html"}


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)[:60] or "page"
