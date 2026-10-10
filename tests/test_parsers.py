"""Check API response parsers against realistic sample payloads (no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from research_agent.config import Config  # noqa: E402
from research_agent.llm import LLM, LLMError  # noqa: E402
from research_agent.tools.toolkit import Toolkit  # noqa: E402


class FakeResp:
    def __init__(self, text="", js=None):
        self.text, self._js = text, js

    def json(self):
        return self._js


def tk_with(resp):
    tk = Toolkit(Config())
    tk._get = lambda *a, **k: resp  # type: ignore
    return tk


ARXIV = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.00001v1</id>
<published>2024-01-02T00:00:00Z</published><title>Deep  Batteries
 for Grids</title><summary>We study storage.</summary><author><name>A. One</name></author>
<link href="http://arxiv.org/abs/2401.00001v1" rel="alternate"/><link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" rel="related"/></entry></feed>"""


def test_arxiv():
    r = tk_with(FakeResp(text=ARXIV)).arxiv("x")
    assert r[0]["title"] == "Deep Batteries for Grids" and r[0]["pdf_url"].endswith("2401.00001v1") and r[0]["year"] == "2024"


def test_openalex():
    js = {"results": [{"display_name": "Paper", "doi": "https://doi.org/10.1/x", "publication_year": 2022,
                       "best_oa_location": {"pdf_url": "https://x.org/p.pdf"}, "abstract_inverted_index": {"Hello": [0], "world": [1]},
                       "authorships": [{"author": {"display_name": "Z"}}]}]}
    r = tk_with(FakeResp(js=js)).openalex("x")
    assert r[0]["snippet"] == "Hello world" and r[0]["pdf_url"] == "https://x.org/p.pdf"


def test_ckan_zenodo_wiki_worldbank():
    ck = {"result": {"results": [{"title": "T", "name": "t", "notes": "<b>n</b>", "resources": [{"format": "CSV", "url": "https://d/x.csv"}]},
                                 {"title": "NoRes", "name": "n", "resources": [{"format": "PDF", "url": "u"}]}]}}
    r = tk_with(FakeResp(js=ck)).ckan("x")
    assert len(r) == 1 and r[0]["download_url"].endswith(".csv")
    zen = {"hits": {"hits": [{"links": {"self_html": "https://zenodo.org/r/1"}, "metadata": {"title": "Z", "description": "d", "creators": [{"name": "A"}]},
                              "files": [{"key": "a.csv", "size": 10, "links": {"self": "https://zenodo.org/f/a.csv"}}]}]}}
    assert tk_with(FakeResp(js=zen)).zenodo("x")[0]["download_url"].endswith("a.csv")
    wiki = {"query": {"search": [{"title": "Solar power", "snippet": "a <span>b</span>"}]}}
    assert tk_with(FakeResp(js=wiki)).wikipedia("x")[0]["url"].endswith("Solar_power")
    wb = [{"page": 1}, [{"indicator": {"id": "I", "value": "Ind"}, "country": {"value": "Kenya"}, "countryiso3code": "KEN", "date": "2020", "value": 5.0},
                        {"indicator": {"id": "I", "value": "Ind"}, "country": {"value": "Kenya"}, "countryiso3code": "KEN", "date": "2019", "value": None}]]
    df = tk_with(FakeResp(js=wb)).worldbank("I")
    assert len(df) == 1 and df.iloc[0]["year"] == 2020


def test_llm_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = Config()
    cfg.llm_provider, cfg.gemini_api_key = "gemini", ""
    with pytest.raises(LLMError):
        LLM(cfg)


def test_server(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    import time

    import server

    c = TestClient(server.app)
    jid = c.post("/research", json={"topic": "test topic", "depth": "quick", "demo": True}).json()["job_id"]
    for _ in range(60):
        st = c.get(f"/jobs/{jid}").json()
        if st["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert st["status"] == "done", st
    assert c.get(f"/jobs/{jid}/report").headers["content-type"] == "application/pdf"
