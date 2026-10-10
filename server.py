"""Optional web UI + REST API.   uvicorn server:app --port 8000"""
from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from research_agent.config import Config
from research_agent.orchestrator import ResearchOrchestrator

app = FastAPI(title="Deep Research Agent")
JOBS: dict[str, dict] = {}


class Req(BaseModel):
    topic: str
    depth: str = "standard"
    demo: bool = False


def _worker(job_id: str, req: Req) -> None:
    job = JOBS[job_id]
    try:
        cfg = Config().apply_depth(req.depth)
        cfg.verbose = False
        if req.demo:
            cfg.offline_demo, cfg.llm_provider = True, "mock"
        orch = ResearchOrchestrator(req.topic, cfg, on_event=lambda ev: job["events"].append(ev))
        job["status"] = "running"
        pdf = orch.run()
        job.update(status="done", pdf=str(pdf), run_dir=str(pdf.parent))
    except Exception as e:  # noqa: BLE001
        job.update(status="error", error=str(e))


@app.post("/research")
def start(req: Req):
    if not req.topic.strip():
        raise HTTPException(400, "topic is required")
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "queued", "events": [], "topic": req.topic}
    threading.Thread(target=_worker, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {"status": job["status"], "error": job.get("error"), "events": job["events"][-40:]}


@app.get("/jobs/{job_id}/report")
def report(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "report not ready")
    return FileResponse(job["pdf"], media_type="application/pdf", filename=f"{Path(job['run_dir']).name}.pdf")


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Deep Research Agent</title>
<style>body{font-family:system-ui;max-width:760px;margin:30px auto;padding:0 16px;color:#222}input,select,button{font-size:16px;padding:10px;margin:4px 0}
input[type=text]{width:100%;box-sizing:border-box}button{background:#1F4E79;color:#fff;border:0;border-radius:6px;cursor:pointer}
pre{background:#f4f6f8;padding:12px;border-radius:6px;max-height:340px;overflow:auto;font-size:12px}a.btn{display:inline-block;background:#E07B39;color:#fff;padding:10px 16px;border-radius:6px;text-decoration:none}</style>
<h2>Deep Research Agent</h2>
<input id=t type=text placeholder="Enter an idea or topic to research...">
<select id=d><option>quick</option><option selected>standard</option><option>deep</option></select>
<label><input type=checkbox id=demo> offline demo</label><br><button onclick=go()>Start research</button>
<pre id=log>Idle.</pre><div id=out></div>
<script>
async function go(){const r=await fetch('/research',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({topic:t.value,depth:d.value,demo:demo.checked})});const {job_id}=await r.json();out.innerHTML='';poll(job_id)}
async function poll(id){const s=await (await fetch('/jobs/'+id)).json();
log.textContent=s.events.map(e=>`[${e.t}s] ${e.agent}: ${e.msg}`).join('\\n')||s.status;log.scrollTop=log.scrollHeight;
if(s.status=='done'){out.innerHTML=`<a class=btn href=/jobs/${id}/report>Download PDF report</a>`;return}
if(s.status=='error'){log.textContent+='\\nERROR: '+s.error;return}setTimeout(()=>poll(id),2000)}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
