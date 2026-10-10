from __future__ import annotations

import math
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..state import Source
from ..tools.doc_tools import html_to_text
from .base import BaseAgent


class DownloaderAgent(BaseAgent):
    name = "Downloader"
    role = "You download PDFs, web pages and datasets safely (size limits, type sniffing)."

    # -- World Bank indicators (clean, reliable time series) --------------
    def fetch_worldbank(self) -> int:
        plan = self.state.plan
        inds = [i for i in plan.get("worldbank_indicators", []) if isinstance(i, dict) and i.get("code")]
        limit = max(0, math.ceil(self.cfg.max_datasets / 2))
        countries = ";".join(plan.get("countries") or []) or "all"
        got = 0
        for ind in inds[:limit]:
            code = str(ind["code"]).strip()
            try:
                df = self.tk.worldbank(code, countries=countries)
            except Exception as e:  # noqa: BLE001
                self.log(f"World Bank {code} failed: {str(e)[:100]}")
                continue
            if df is None or len(df) < 10:
                self.log(f"World Bank {code}: no data")
                continue
            path = self.state.run_dir / "datasets" / f"worldbank_{code}.csv"
            df.to_csv(path, index=False)
            name = ind.get("name") or df["indicator"].iloc[0]
            src = self.state.add_source(title=f"World Bank: {name} ({code})", url=f"https://data.worldbank.org/indicator/{code}",
                                        kind="dataset", origin="worldbank", snippet=name, year=str(df["year"].max()),
                                        authors="World Bank", selected=True, status="downloaded",
                                        local_path=str(path), file_kind="csv", score=9.0)
            if src:
                self.state.datasets.append({"source_id": src.id, "name": path.name, "path": str(path), "title": src.title})
                got += 1
                self.log(f"World Bank {code}: {len(df):,} rows saved")
        return got

    # -- generic downloads -------------------------------------------------
    def _fetch(self, s: Source) -> None:
        dest = self.state.run_dir / ("datasets" if s.kind == "dataset" else "sources")
        urls = [u for u in (s.download_url, s.pdf_url, s.url) if u]
        seen, err = set(), ""
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            try:
                info = self.tk.download(url, dest, prefix=f"{s.id:02d}_")
            except Exception as e:  # noqa: BLE001
                err = str(e)[:140]
                continue
            kind, path = info["kind"], info["path"]
            if kind == "zip":
                try:
                    with zipfile.ZipFile(path) as z:
                        names = [n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx", ".json", ".pdf"))]
                        if names:
                            out = Path(path).with_name(f"{s.id:02d}_" + Path(names[0]).name)
                            out.write_bytes(z.read(names[0]))
                            path, kind = str(out), out.suffix.lower().lstrip(".")
                except Exception as e:  # noqa: BLE001
                    err = f"bad zip: {e}"
                    continue
            if s.kind == "dataset":
                if kind not in ("csv", "xlsx", "xls", "json", "parquet"):
                    err = f"not tabular ({kind})"
                    continue
            elif kind == "html":
                text, _ = html_to_text(Path(path).read_text(encoding="utf-8", errors="ignore"))
                if len(text) < 300:
                    err = "page has too little text"
                    continue
                s.text = text
            elif kind not in ("pdf", "html", "text"):
                err = f"unsupported type ({kind})"
                continue
            s.local_path, s.file_kind, s.status, s.error = path, kind, "downloaded", ""
            if s.kind == "dataset":
                self.state.datasets.append({"source_id": s.id, "name": Path(path).name, "path": path, "title": s.title})
            return
        s.status, s.error = "failed", err or "no usable URL"

    def run(self, sources: list[Source]) -> None:
        todo = [s for s in sources if s.selected and s.status == "found"]
        with ThreadPoolExecutor(max_workers=self.cfg.download_workers) as ex:
            list(ex.map(self._fetch, todo))
        ok = [s for s in todo if s.status == "downloaded"]
        for s in todo:
            if s.status == "failed":
                self.log(f"x {s.title[:60]} - {s.error}")
        self.log(f"downloaded {len(ok)}/{len(todo)} "
                 f"({sum(1 for s in ok if s.file_kind == 'pdf')} PDFs, {sum(1 for s in ok if s.kind == 'dataset')} datasets, "
                 f"{sum(1 for s in ok if s.file_kind in ('html', 'text'))} web pages)")
