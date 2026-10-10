"""Shared state passed between agents (the 'blackboard')."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


def normalize_url(url: str) -> str:
    try:
        p = urlparse(url.strip())
        q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(("utm_", "fbclid", "gclid"))]
        path = p.path.rstrip("/") or "/"
        return urlunparse((p.scheme.lower(), p.netloc.lower().removeprefix("www."), path, "", urlencode(q), ""))
    except Exception:
        return url.strip().lower()


def slugify(text: str, n: int = 50) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n].strip("-") or "research"


@dataclass
class Source:
    id: int
    title: str
    url: str
    kind: str = "web"  # web | paper | wiki | dataset
    origin: str = ""  # search engine / API that found it
    snippet: str = ""
    pdf_url: str = ""
    download_url: str = ""
    year: str = ""
    authors: str = ""
    heuristic: float = 0.0
    llm_score: float = 0.0
    score: float = 0.0
    reason: str = ""
    selected: bool = False
    status: str = "found"  # found | downloaded | read | abstract_only | summarized | failed
    error: str = ""
    local_path: str = ""
    file_kind: str = ""
    text: str = ""
    pages: int = 0
    summary: str = ""
    key_points: list = field(default_factory=list)
    key_numbers: list = field(default_factory=list)
    relevance: float = 0.0
    quality_notes: str = ""

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc.removeprefix("www.")


@dataclass
class ChartInfo:
    id: str
    path: str
    title: str
    caption: str = ""
    dataset: str = ""


@dataclass
class TableInfo:
    id: str
    title: str
    columns: list
    rows: list
    caption: str = ""


class ResearchState:
    def __init__(self, topic: str, run_dir: Path, on_event: Callable[[dict], None] | None = None, verbose: bool = True):
        self.topic = topic
        self.run_dir = Path(run_dir)
        self.on_event = on_event
        self.verbose = verbose
        for sub in ("sources", "datasets", "charts", "tables"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.plan: dict = {}
        self.sources: list[Source] = []
        self._by_url: dict[str, Source] = {}
        self._next_id = 1
        self.datasets: list[dict] = []
        self.analyses: list[dict] = []
        self.charts: list[ChartInfo] = []
        self.tables: list[TableInfo] = []
        self.findings: dict = {}
        self.critique: dict = {}
        self.sections: list[dict] = []  # {title, content}
        self.title = ""
        self.subtitle = ""
        self.exec_summary = ""
        self.takeaways: list[str] = []
        self.queries_run: list[str] = []
        self.events: list[dict] = []
        self.stage = "init"
        self.stats: dict = {}

    # -- sources ---------------------------------------------------------
    def add_source(self, **kw) -> Source | None:
        url = kw.get("url") or kw.get("download_url") or kw.get("pdf_url") or ""
        if not url:
            return None
        key = normalize_url(url)
        if key in self._by_url:
            return None
        s = Source(id=self._next_id, **kw)
        self._next_id += 1
        self.sources.append(s)
        self._by_url[key] = s
        return s

    def get(self, sid: int) -> Source | None:
        return next((s for s in self.sources if s.id == sid), None)

    # -- logging ---------------------------------------------------------
    def log(self, agent: str, msg: str) -> None:
        ev = {"t": round(time.time() - self.started, 1), "stage": self.stage, "agent": agent, "msg": msg}
        self.events.append(ev)
        if self.verbose:
            print(f"[{ev['t']:>6.1f}s] {agent:<16} {msg}", flush=True)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.log("Orchestrator", f"== {stage} ==")

    # -- persistence -----------------------------------------------------
    def save(self) -> None:
        def src_dict(s: Source) -> dict:
            d = asdict(s)
            d["text"] = s.text[:1500]
            return d

        data = {
            "topic": self.topic,
            "title": self.title,
            "plan": self.plan,
            "queries_run": self.queries_run,
            "sources": [src_dict(s) for s in self.sources],
            "datasets": self.datasets,
            "analyses": self.analyses,
            "charts": [asdict(c) for c in self.charts],
            "tables": [asdict(t) for t in self.tables],
            "findings": self.findings,
            "critique": self.critique,
            "sections": self.sections,
            "executive_summary": self.exec_summary,
            "key_takeaways": self.takeaways,
            "stats": self.stats,
            "events": self.events,
        }
        (self.run_dir / "state.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
