"""Real-world tools: web/academic/dataset search + file download."""
from __future__ import annotations

import html as htmlmod
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd
import requests

from ..config import Config
from .doc_tools import sniff_file

DATA_EXT = ("csv", "xlsx", "xls", "json", "zip", "parquet", "tsv")


def _safe_name(name: str, n: int = 90) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._")
    return name[:n] or "file"


def _strip_tags(s: str) -> str:
    return htmlmod.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))[:700]


class Toolkit:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent, "Accept-Language": "en"})

    # -- http ------------------------------------------------------------
    def _get(self, url: str, params: dict | None = None, stream: bool = False, timeout: float | None = None):
        last: Exception | None = None
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, stream=stream, timeout=timeout or self.cfg.request_timeout,
                                     allow_redirects=True)
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    last = requests.HTTPError(f"HTTP {r.status_code}")
                    continue
                r.raise_for_status()
                return r
            except requests.HTTPError as e:
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise
                last = e
            except requests.RequestException as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise last or RuntimeError("request failed")

    # -- search ----------------------------------------------------------
    def web_search(self, query: str, n: int = 8, filetype: str | None = None) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:  # older package name
            from duckduckgo_search import DDGS
        q = f"{query} filetype:{filetype}" if filetype else query
        with DDGS() as d:
            rows = list(d.text(q, max_results=n))
        out = []
        for r in rows:
            url = r.get("href") or r.get("url") or ""
            if not url.startswith("http"):
                continue
            is_pdf = urlparse(url).path.lower().endswith(".pdf")
            item = dict(title=r.get("title", url), url=url, snippet=r.get("body", ""), kind="web",
                        origin="duckduckgo", pdf_url=url if is_pdf else "", download_url="", year="", authors="")
            if filetype and filetype.lower() in DATA_EXT:
                item.update(kind="dataset", download_url=url, origin=f"duckduckgo:{filetype}")
            out.append(item)
        return out

    def arxiv(self, query: str, n: int = 6) -> list[dict]:
        r = self._get("https://export.arxiv.org/api/query",
                      params={"search_query": f"all:{query}", "start": 0, "max_results": n, "sortBy": "relevance"})
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        out = []
        for e in root.findall("a:entry", ns):
            title = re.sub(r"\s+", " ", (e.findtext("a:title", "", ns) or "")).strip()
            url = e.findtext("a:id", "", ns) or ""
            pdf = ""
            for l in e.findall("a:link", ns):
                if l.get("title") == "pdf":
                    pdf = l.get("href", "")
            authors = ", ".join(a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)[:4])
            out.append(dict(title=title, url=url, snippet=re.sub(r"\s+", " ", e.findtext("a:summary", "", ns) or "").strip()[:700],
                            kind="paper", origin="arxiv", pdf_url=pdf or url.replace("/abs/", "/pdf/"), download_url="",
                            year=(e.findtext("a:published", "", ns) or "")[:4], authors=authors))
        return out

    def openalex(self, query: str, n: int = 6) -> list[dict]:
        params = {"search": query, "per-page": n, "filter": "is_oa:true,has_fulltext:true"}
        if self.cfg.contact_email:
            params["mailto"] = self.cfg.contact_email
        r = self._get("https://api.openalex.org/works", params=params)
        out = []
        for w in r.json().get("results", []):
            loc = w.get("best_oa_location") or {}
            pdf = loc.get("pdf_url") or (w.get("open_access") or {}).get("oa_url") or ""
            url = w.get("doi") or w.get("id") or ""
            authors = ", ".join(a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])[:4])
            out.append(dict(title=w.get("display_name") or "Untitled", url=url, snippet=_abstract(w.get("abstract_inverted_index")),
                            kind="paper", origin="openalex", pdf_url=pdf, download_url="",
                            year=str(w.get("publication_year") or ""), authors=authors))
        return [o for o in out if o["url"]]

    def wikipedia(self, query: str, n: int = 3) -> list[dict]:
        r = self._get("https://en.wikipedia.org/w/api.php",
                      params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n})
        out = []
        for h in r.json().get("query", {}).get("search", []):
            t = h["title"]
            out.append(dict(title=f"Wikipedia: {t}", url="https://en.wikipedia.org/wiki/" + t.replace(" ", "_"),
                            snippet=_strip_tags(h.get("snippet", "")), kind="wiki", origin="wikipedia",
                            pdf_url="", download_url="", year="", authors=""))
        return out

    def ckan(self, query: str, n: int = 6, base: str = "https://catalog.data.gov") -> list[dict]:
        r = self._get(f"{base}/api/3/action/package_search", params={"q": query, "rows": n})
        out = []
        for p in r.json().get("result", {}).get("results", []):
            res = [x for x in p.get("resources", []) if (x.get("format") or "").lower() in ("csv", "xlsx", "xls", "json")]
            if not res:
                continue
            res.sort(key=lambda x: 0 if (x.get("format") or "").lower() == "csv" else 1)
            out.append(dict(title=p.get("title", "Dataset"), url=f"{base}/dataset/{p.get('name')}",
                            snippet=_strip_tags(p.get("notes", ""))[:500], kind="dataset", origin="data.gov",
                            pdf_url="", download_url=res[0].get("url", ""), year=(p.get("metadata_modified") or "")[:4],
                            authors=(p.get("organization") or {}).get("title", "")))
        return out

    def zenodo(self, query: str, n: int = 5) -> list[dict]:
        r = self._get("https://zenodo.org/api/records", params={"q": query, "size": n, "type": "dataset", "sort": "mostviewed"})
        out = []
        limit = self.cfg.max_file_mb * 1024 * 1024
        for h in r.json().get("hits", {}).get("hits", []):
            files = [f for f in h.get("files", []) if f.get("key", "").lower().endswith(DATA_EXT) and f.get("size", 0) < limit]
            if not files:
                continue
            md = h.get("metadata", {})
            out.append(dict(title=md.get("title", "Zenodo dataset"), url=h.get("links", {}).get("self_html") or h.get("links", {}).get("html", ""),
                            snippet=_strip_tags(md.get("description", ""))[:500], kind="dataset", origin="zenodo",
                            pdf_url="", download_url=files[0]["links"]["self"], year=(md.get("publication_date") or "")[:4],
                            authors=", ".join(c.get("name", "") for c in md.get("creators", [])[:3])))
        return [o for o in out if o["url"]]

    def worldbank(self, indicator: str, countries: str = "all", date: str = "2000:2023") -> pd.DataFrame | None:
        url = f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
        r = self._get(url, params={"format": "json", "per_page": 20000, "date": date})
        data = r.json()
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            return None
        rows = [{"country": x["country"]["value"], "iso3": x.get("countryiso3code", ""), "year": int(x["date"]),
                 "indicator": x["indicator"]["value"], "value": x["value"]} for x in data[1] if x.get("value") is not None]
        return pd.DataFrame(rows) if rows else None

    # -- download --------------------------------------------------------
    def download(self, url: str, dest_dir: str | Path, max_mb: float | None = None, prefix: str = "") -> dict:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        max_bytes = int((max_mb or self.cfg.max_file_mb) * 1024 * 1024)
        r = self._get(url, stream=True)
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        cl = r.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > max_bytes:
            r.close()
            raise ValueError(f"file too large ({int(cl)/1e6:.1f} MB)")
        fname = ""
        cd = r.headers.get("Content-Disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
        if m:
            fname = unquote(m.group(1))
        if not fname:
            fname = unquote(os.path.basename(urlparse(r.url).path)) or "download"
        path = dest / (prefix + _safe_name(fname))
        size = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    path.unlink(missing_ok=True)
                    raise ValueError(f"file exceeds {max_bytes/1e6:.0f} MB limit")
                f.write(chunk)
        kind = sniff_file(str(path), ctype)
        want = {"pdf": ".pdf", "html": ".html", "csv": ".csv", "xlsx": ".xlsx", "xls": ".xls", "json": ".json",
                "zip": ".zip", "parquet": ".parquet", "text": ".txt"}.get(kind)
        if want and path.suffix.lower() != want:
            new = path.with_name(path.stem + want) if path.suffix.lower() in (".php", ".aspx", ".asp", ".htm", ".html", "") else path.with_name(path.name + want)
            path.rename(new)
            path = new
        return {"path": str(path), "content_type": ctype, "size": size, "kind": kind}
