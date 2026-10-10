from __future__ import annotations

import numpy as np
import pandas as pd

from ..tools.data_tools import clean_df, col_kind, fmt_num, load_table, profile_df
from .base import BaseAgent


def _col(df: pd.DataFrame, name) -> str | None:
    if not name or not isinstance(name, str):
        return None
    if name in df.columns:
        return name
    low = {c.lower(): c for c in df.columns}
    return low.get(name.lower())


def _years(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.year + (s.dt.dayofyear - 1) / 366
    return pd.to_numeric(s, errors="coerce")


class AnalystAgent(BaseAgent):
    name = "DataAnalyst"
    role = "You analyse datasets quantitatively. You choose analyses; numbers are computed by code, never guessed."

    # ------------------------------------------------------------------
    def run(self) -> list[dict]:
        """Returns chart_requests (list of dicts with DataFrames). Fills state.analyses and state.tables."""
        requests: list[dict] = []
        n = 0
        for d in list(self.state.datasets)[: self.cfg.max_datasets]:
            try:
                df = clean_df(load_table(d["path"]))
            except Exception as e:  # noqa: BLE001
                self.log(f"cannot load {d['name']}: {str(e)[:100]}")
                d["error"] = str(e)[:100]
                continue
            if df.empty or df.shape[1] < 2:
                self.log(f"{d['name']}: empty or single column, skipped")
                continue
            prof = profile_df(df)
            d.update(rows=prof["rows"], cols=prof["cols"], columns=[c["name"] for c in prof["columns"]][:15])
            time_cols = [c["name"] for c in prof["columns"] if c["kind"] == "time"]
            if time_cols:
                tc = time_cols[0]
                d["coverage"] = f"{tc}: {df[tc].min()} to {df[tc].max()}".replace(" 00:00:00", "")
            plan = self._plan(d, prof)
            d["description"] = plan.get("description", "")
            specs = [sp for sp in plan.get("analyses", []) if isinstance(sp, dict)][:5]
            results = []
            for sp in specs:
                try:
                    r = self._execute(df, sp)
                except Exception as e:  # noqa: BLE001
                    self.log(f"  analysis '{sp.get('type')}' failed: {str(e)[:80]}")
                    r = None
                if r:
                    results.append(r)
            if not results:
                for sp in self._fallback_specs(df, prof):
                    try:
                        r = self._execute(df, sp)
                        if r:
                            results.append(r)
                    except Exception:  # noqa: BLE001
                        continue
            for r in results:
                n += 1
                aid = f"a{n}"
                entry = {"id": aid, "dataset": d["name"], "source_id": d["source_id"], "type": r["type"], "title": r["title"],
                         "why": r.get("why", ""), "insights": r["insights"], "chart_id": None, "table_id": None}
                if r.get("chart"):
                    req = dict(r["chart"], analysis_id=aid, dataset=d["name"])
                    requests.append(req)
                if r.get("table"):
                    tid = f"tab{len(self.state.tables) + 1}"
                    from ..state import TableInfo

                    self.state.tables.append(TableInfo(id=tid, title=r["title"], columns=r["table"]["columns"], rows=r["table"]["rows"],
                                                       caption=f"{r['title']} ({d['name']})"))
                    entry["table_id"] = tid
                self.state.analyses.append(entry)
            self.log(f"{d['name']}: {prof['rows']:,} rows x {prof['cols']} cols -> {len(results)} analyses")
        return requests

    # ------------------------------------------------------------------
    def _plan(self, d: dict, prof: dict) -> dict:
        cols = prof["columns"]
        by = lambda k: ", ".join(c["name"] for c in cols if c["kind"] == k)  # noqa: E731
        desc = "\n".join(
            f"- {c['name']} [{c['kind']}] unique={c['unique']} null%={c['null_pct']}"
            + (f" range={c.get('min'):.4g}..{c.get('max'):.4g}" if c["kind"] == "numeric" and "min" in c else "")
            + (f" top={c.get('top')}" if c.get("top") else "") for c in cols[:30])
        prompt = f"""TOPIC: {self.state.topic}
DATASET: {d.get('title', d['name'])}  ({prof['rows']:,} rows)
NUMERIC: {by('numeric')}
CATEGORICAL: {by('categorical')}
TIME: {by('time')}
COLUMNS:
{desc}
SAMPLE ROWS: {prof['head'][:3]}

Choose 3-5 analyses that best illuminate the topic. Allowed types and fields (use EXACT column names):
- {{"type":"trend","x":<time/numeric col>,"y":<numeric col>,"group":<categorical col or null>,"title":"..","why":".."}}
- {{"type":"distribution","column":<numeric col>,"title":"..","why":".."}}
- {{"type":"correlation","columns":[<numeric cols, 3-8>],"title":"..","why":".."}}
- {{"type":"groupby_bar","category":<categorical col>,"value":<numeric col or null for counts>,"agg":"mean|sum|median|count","top":10,"title":"..","why":".."}}
- {{"type":"scatter","x":<numeric>,"y":<numeric>,"title":"..","why":".."}}
- {{"type":"summary_table","title":"..","why":".."}}
Return JSON: {{"description": "1-2 sentences on what this dataset contains", "analyses": [ ... ]}}"""
        res = self.llm.complete_json(prompt, system=self.system, task="analyst", default={})
        return res if isinstance(res, dict) else {}

    def _fallback_specs(self, df: pd.DataFrame, prof: dict) -> list[dict]:
        kinds = {c["name"]: c["kind"] for c in prof["columns"]}
        nums = [c for c, k in kinds.items() if k == "numeric"]
        times = [c for c, k in kinds.items() if k == "time"]
        cats = [c for c, k in kinds.items() if k == "categorical"]
        specs = [{"type": "summary_table", "title": "Summary statistics"}]
        if nums:
            specs.append({"type": "distribution", "column": nums[0], "title": f"Distribution of {nums[0]}"})
        if times and nums:
            specs.append({"type": "trend", "x": times[0], "y": nums[0], "group": cats[0] if cats else None, "title": f"{nums[0]} over time"})
        if len(nums) >= 3:
            specs.append({"type": "correlation", "columns": nums[:8], "title": "Correlation between numeric variables"})
        if cats and nums:
            specs.append({"type": "groupby_bar", "category": cats[0], "value": nums[0], "agg": "mean", "top": 10, "title": f"Mean {nums[0]} by {cats[0]}"})
        return specs

    # ------------------------------------------------------------------
    def _execute(self, df: pd.DataFrame, sp: dict) -> dict | None:
        t = sp.get("type")
        title = str(sp.get("title") or t)
        why = str(sp.get("why") or "")
        fn = {"trend": self._trend, "distribution": self._dist, "correlation": self._corr, "groupby_bar": self._groupby,
              "scatter": self._scatter, "summary_table": self._summary}.get(t)
        if not fn:
            return None
        out = fn(df, sp, title)
        if out:
            out.update(type=t, title=title, why=why)
        return out

    def _trend(self, df, sp, title):
        x, y = _col(df, sp.get("x")), _col(df, sp.get("y"))
        g = _col(df, sp.get("group"))
        if not x or not y or col_kind(df, y) != "numeric" or col_kind(df, x) not in ("time", "numeric"):
            return None
        cols = [x, y] + ([g] if g else [])
        d = df[cols].dropna().copy()
        if len(d) < 3:
            return None
        if g:
            keep = d.groupby(g)[y].mean().abs().nlargest(6).index
            d = d[d[g].isin(keep)]
            agg = d.groupby([x, g])[y].mean().reset_index()
        else:
            agg = d.groupby(x)[y].mean().reset_index()
        insights = []
        groups = [(None, agg)] if not g else list(agg.groupby(g))
        for name, sub in groups:
            sub = sub.sort_values(x)
            if len(sub) < 2:
                continue
            first, last = sub[y].iloc[0], sub[y].iloc[-1]
            xs = _years(sub[x])
            slope = np.polyfit(xs, sub[y], 1)[0] if xs.nunique() > 1 else float("nan")
            chg = f"{(last / first - 1) * 100:+.1f}%" if first not in (0, np.nan) and not pd.isna(first) else "n/a"
            peak = sub.loc[sub[y].idxmax()]
            label = f"{name}: " if name is not None else ""
            insights.append(f"{label}{y} moved from {fmt_num(first)} to {fmt_num(last)} ({chg}) between {str(sub[x].iloc[0])[:10]} and "
                            f"{str(sub[x].iloc[-1])[:10]}; peak {fmt_num(peak[y])} at {str(peak[x])[:10]}; linear slope {fmt_num(slope)} per year.")
        return {"insights": insights[:8], "chart": {"kind": "line", "df": agg, "x": x, "y": y, "group": g, "title": title},
                "table": None}

    def _dist(self, df, sp, title):
        c = _col(df, sp.get("column"))
        if not c or col_kind(df, c) not in ("numeric",):
            return None
        s = df[c].dropna()
        if len(s) < 5:
            return None
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        ins = [f"{c}: n={len(s):,}, mean {fmt_num(s.mean())}, median {fmt_num(s.median())}, std {fmt_num(s.std())}, "
               f"5th-95th percentile {fmt_num(s.quantile(.05))} to {fmt_num(s.quantile(.95))}; skewness {s.skew():.2f}; {out} IQR outliers ({out / len(s) * 100:.1f}%)."]
        return {"insights": ins, "chart": {"kind": "hist", "series": s, "title": title, "xlabel": c}, "table": None}

    def _corr(self, df, sp, title):
        cols = [_col(df, c) for c in (sp.get("columns") or [])]
        cols = [c for c in cols if c and col_kind(df, c) == "numeric"]
        if len(cols) < 3:
            cols = [c for c in df.columns if col_kind(df, c) == "numeric"][:8]
        if len(cols) < 2:
            return None
        corr = df[cols].corr(numeric_only=True)
        pairs = []
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                v = corr.loc[a, b]
                if not pd.isna(v):
                    pairs.append((abs(v), v, a, b))
        pairs.sort(reverse=True)
        if not pairs:
            return None
        ins = [f"{a} vs {b}: Pearson r = {v:.2f} ({'strong' if abs(v) >= .7 else 'moderate' if abs(v) >= .4 else 'weak'} {'positive' if v > 0 else 'negative'})."
               for _, v, a, b in pairs[:5]]
        ins.append("Correlation does not imply causation.")
        table = {"columns": ["Variable A", "Variable B", "Pearson r"], "rows": [[a, b, f"{v:.2f}"] for _, v, a, b in pairs[:8]]}
        return {"insights": ins, "chart": {"kind": "heatmap", "corr": corr, "title": title}, "table": table}

    def _groupby(self, df, sp, title):
        cat = _col(df, sp.get("category"))
        val = _col(df, sp.get("value"))
        agg = sp.get("agg") if sp.get("agg") in ("mean", "sum", "median", "count") else "mean"
        top = int(sp.get("top") or 10)
        if not cat or col_kind(df, cat) not in ("categorical", "text"):
            return None
        if agg == "count" or not val or col_kind(df, val) != "numeric":
            g = df[cat].value_counts().head(top)
            agg, label = "count", "count"
        else:
            g = df.groupby(cat)[val].agg(agg).sort_values(ascending=False).head(top)
            label = f"{agg} of {val}"
        if g.empty:
            return None
        share = ""
        if agg in ("sum", "count") and g.sum():
            share = f" The top group accounts for {g.iloc[0] / g.sum() * 100:.0f}% of the shown total."
        ins = [f"By {cat} ({label}): highest is {g.index[0]} at {fmt_num(g.iloc[0])}, lowest shown is {g.index[-1]} at {fmt_num(g.iloc[-1])} "
               f"({g.iloc[0] / g.iloc[-1]:.1f}x gap)." + share if g.iloc[-1] else f"By {cat}: top is {g.index[0]} at {fmt_num(g.iloc[0])}."]
        table = {"columns": [cat, label], "rows": [[str(i), fmt_num(v)] for i, v in g.items()]}
        return {"insights": ins, "chart": {"kind": "bar", "labels": [str(i) for i in g.index], "values": [float(v) for v in g.values],
                                           "title": title, "xlabel": label}, "table": table}

    def _scatter(self, df, sp, title):
        x, y = _col(df, sp.get("x")), _col(df, sp.get("y"))
        if not x or not y or x == y or col_kind(df, x) != "numeric" or col_kind(df, y) != "numeric":
            return None
        d = df[[x, y]].dropna()
        if len(d) < 8:
            return None
        r = d[x].corr(d[y])
        rho = d[x].corr(d[y], method="spearman")
        slope = np.polyfit(d[x], d[y], 1)[0] if d[x].nunique() > 1 else float("nan")
        ins = [f"{x} vs {y} (n={len(d):,}): Pearson r = {r:.2f}, Spearman rho = {rho:.2f}; each +1 in {x} is associated with {fmt_num(slope)} change in {y}."]
        return {"insights": ins, "chart": {"kind": "scatter", "df": d, "x": x, "y": y, "title": title}, "table": None}

    def _summary(self, df, sp, title):
        nums = [c for c in df.columns if col_kind(df, c) == "numeric"][:8]
        if not nums:
            return None
        desc = df[nums].describe().T
        rows = [[c, fmt_num(r["count"]), fmt_num(r["mean"]), fmt_num(r["std"]), fmt_num(r["min"]), fmt_num(r["50%"]), fmt_num(r["max"])]
                for c, r in desc.iterrows()]
        ins = [f"Dataset has {len(df):,} rows and {df.shape[1]} columns; {len(nums)} numeric variables summarised."]
        return {"insights": ins, "chart": None,
                "table": {"columns": ["Variable", "Count", "Mean", "Std", "Min", "Median", "Max"], "rows": rows}}
