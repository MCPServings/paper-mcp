"""Medical literature retrieval layer for paper-mcp.

Adds a PubMed-backed, evidence-graded search on top of the generic corpora.
The generic ``search_all`` (arXiv + Semantic Scholar + OpenAlex, RRF over
relevance) is great for CS/ML but fails on clinical questions: it has no
research-type filter, so high-cited reviews / guidelines / diagnostic-criteria
papers bury the actual RCTs, and arXiv injects ML noise. ``search_medical``
fixes that with the two things the generic path lacks:

  * **Research-type filtering** via PubMed Publication Type tags
    (``Randomized Controlled Trial[pt]`` etc.) — NLM's human-curated labels.
  * **Evidence-level re-ranking** (the evidence pyramid: meta-analysis /
    systematic review > RCT > cohort > case-control > case report / other),
    instead of ranking purely by citation count.

Open-access full text is pulled from Europe PMC by the same PMID. This module
is pure retrieval — no LLM. Natural-language / multilingual query understanding
and any LLM re-rank live in the product layer that calls this tool, so the
``query`` here is expected to be English keyword/boolean text PubMed can map.

All endpoints are public and free:
  * NCBI E-utilities  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
  * Europe PMC REST   https://www.ebi.ac.uk/europepmc/webservices/rest/
"""
from __future__ import annotations

import math
import os
import re

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
NCBI_TOOL = os.getenv("NCBI_TOOL", "paper-mcp")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "").strip()

# Friendly study-type name -> PubMed [pt] filter string.
_PT_FILTER = {
    "rct": "Randomized Controlled Trial[pt]",
    "meta-analysis": "Meta-Analysis[pt]",
    "systematic-review": "Systematic Review[pt]",
    "guideline": "Guideline[pt]",
    "review": "Review[pt]",
    "observational": "Observational Study[pt]",
}

# User aliases -> canonical key in _PT_FILTER.
_STUDY_ALIASES = {
    "rct": "rct",
    "randomized controlled trial": "rct",
    "randomised controlled trial": "rct",
    "meta-analysis": "meta-analysis",
    "meta analysis": "meta-analysis",
    "metaanalysis": "meta-analysis",
    "systematic review": "systematic-review",
    "systematic-review": "systematic-review",
    "sr": "systematic-review",
    "guideline": "guideline",
    "review": "review",
    "observational": "observational",
    "observational study": "observational",
}

# Evidence pyramid: PubMed publication type (lowercased) -> level (higher = stronger).
_EVIDENCE_LEVEL = {
    "meta-analysis": 5,
    "systematic review": 5,
    "practice guideline": 5,
    "guideline": 4,
    "randomized controlled trial": 4,
    "controlled clinical trial": 3,
    "clinical trial, phase iii": 3,
    "clinical trial, phase iv": 3,
    "multicenter study": 2,
    "clinical trial": 2,
    "observational study": 2,
    "comparative study": 2,
    "case reports": 1,
    "review": 1,
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"(\d{4})")


def parse_study_types(raw: str) -> list[str]:
    """Comma/semicolon separated study-type names -> canonical keys."""
    out: list[str] = []
    for part in re.split(r"[,;]+", raw or ""):
        key = _STUDY_ALIASES.get(part.strip().lower())
        if key and key not in out:
            out.append(key)
    return out


def _evidence_level(pubtypes: list[str]) -> tuple[int, str]:
    """Strongest evidence level among a paper's publication types."""
    best, label = 0, ""
    for pt in pubtypes or []:
        lvl = _EVIDENCE_LEVEL.get(pt.strip().lower())
        if lvl and lvl > best:
            best, label = lvl, pt
    if best == 0:
        return 1, (pubtypes[0] if pubtypes else "Journal Article")
    return best, label


def _ncbi_params(**kw) -> dict:
    p = {"tool": NCBI_TOOL, **kw}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    return p


def build_pubmed_term(query: str, study_keys: list[str], year_from: int | None) -> str:
    """Compose the PubMed query: user text AND (pt filters) AND year range."""
    q = (query or "").strip()
    if not q:
        raise ValueError("query must not be empty")
    pts = [_PT_FILTER[k] for k in study_keys if k in _PT_FILTER]
    term = q
    if pts:
        term = f"({q}) AND (" + " OR ".join(dict.fromkeys(pts)) + ")"
    if year_from:
        term += f' AND ("{int(year_from)}"[dp] : "3000"[dp])'
    return term


async def _esearch(client: httpx.AsyncClient, term: str, retmax: int):
    r = await client.get(f"{EUTILS}/esearch.fcgi", params=_ncbi_params(
        db="pubmed", term=term, retmax=retmax, retmode="json", sort="relevance"))
    r.raise_for_status()
    res = r.json().get("esearchresult", {})
    return res.get("idlist", []), int(res.get("count", 0) or 0)


async def _esummary(client: httpx.AsyncClient, pmids: list[str]) -> dict:
    if not pmids:
        return {}
    try:
        r = await client.get(f"{EUTILS}/esummary.fcgi", params=_ncbi_params(
            db="pubmed", id=",".join(pmids), retmode="json"))
        r.raise_for_status()
        return r.json().get("result", {})
    except httpx.HTTPError:
        return {}  # No summaries -> rows fall back to whatever Europe PMC has.


async def _epmc_meta(client: httpx.AsyncClient, pmids: list[str]) -> dict:
    """One batch Europe PMC query: PMID -> citations / OA / pmcid / doi / abstract."""
    if not pmids:
        return {}
    q = " OR ".join(f"(EXT_ID:{p} AND SRC:MED)" for p in pmids)
    r = await client.get(f"{EPMC}/search", params={
        "query": q, "format": "json", "resultType": "core",
        "pageSize": min(len(pmids), 100)})
    r.raise_for_status()
    out: dict[str, dict] = {}
    for it in r.json().get("resultList", {}).get("result", []):
        pmid = it.get("pmid")
        if not pmid:
            continue
        out[pmid] = {
            "citations": it.get("citedByCount"),
            "is_open_access": it.get("isOpenAccess") == "Y",
            "pmcid": it.get("pmcid"),
            "doi": it.get("doi"),
            "abstract": it.get("abstractText"),
        }
    return out


async def _epmc_fulltext(client: httpx.AsyncClient, pmcid: str) -> str:
    if not pmcid:
        return ""
    r = await client.get(f"{EPMC}/{pmcid}/fullTextXML")
    if r.status_code != 200:
        return ""
    xml = r.text
    m = re.search(r"<body[^>]*>(.*?)</body>", xml, re.S)
    raw = m.group(1) if m else xml
    raw = re.sub(r"<ref-list.*?</ref-list>", " ", raw, flags=re.S)
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()


def _rank_score(rel_rank: int, level: int, citations, year, this_year: int) -> float:
    """Quality re-rank: evidence level dominates, then relevance, citations, recency.

    The whole point: evidence level is the primary key so real RCTs/SRs rise
    above high-cited reviews/guidelines that pure-citation ranking floats up.
    """
    rel = 1.0 / (1 + rel_rank)                  # PubMed's own relevance order
    cit = math.log1p(citations or 0)
    rec = 0.0
    if year:
        rec = max(0.0, 1.0 - max(0, this_year - year) / 20.0)
    return (3.0 * level) + (2.0 * rel) + (0.5 * cit) + (1.0 * rec)


def _doi_from_summary(s: dict) -> str | None:
    for aid in s.get("articleids") or []:
        if aid.get("idtype") == "doi" and aid.get("value"):
            return aid["value"]
    return None


async def search_medical_impl(
    query: str,
    study_keys: list[str],
    year_from: int | None,
    max_results: int,
    fetch_fulltext: bool,
    fulltext_max_chars: int,
    this_year: int,
) -> dict:
    term = build_pubmed_term(query, study_keys, year_from)
    retmax = max(max_results * 3, 20)
    async with httpx.AsyncClient(timeout=40) as client:
        pmids, total = await _esearch(client, term, retmax)
        # Auto-relax: if the research-type filter yields nothing, drop it and
        # retry on the bare query so a niche question still returns evidence
        # (flagged, so the caller knows the [pt] filter was lifted).
        filter_relaxed = False
        if not pmids and study_keys:
            term = build_pubmed_term(query, [], year_from)
            pmids, total = await _esearch(client, term, retmax)
            filter_relaxed = True
        if not pmids:
            return {"query": query, "pubmed_term": term, "filter_relaxed": filter_relaxed,
                    "total_found": 0, "count": 0, "results": []}
        summ = await _esummary(client, pmids)
        try:
            meta = await _epmc_meta(client, pmids)
        except httpx.HTTPError:
            meta = {}  # Europe PMC is enrichment only; degrade gracefully.

        rows: list[dict] = []
        for rank, pmid in enumerate(pmids):
            s = summ.get(pmid) or {}
            if s.get("error"):
                continue
            m = meta.get(pmid, {})
            # Need at least one source of metadata; skip totally-empty records.
            if not s and not m:
                continue
            level, label = _evidence_level(s.get("pubtype") or [])
            ym = _YEAR_RE.match(s.get("sortpubdate") or s.get("pubdate") or "")
            year = int(ym.group(1)) if ym else None
            authors = [a.get("name") for a in (s.get("authors") or []) if a.get("name")]
            title = (s.get("title") or m.get("title") or "").rstrip(".")
            rows.append({
                "pmid": pmid,
                "doi": m.get("doi") or _doi_from_summary(s),
                "title": title,
                "authors": authors[:8] + (["et al."] if len(authors) > 8 else []),
                "year": year,
                "journal": s.get("fulljournalname") or s.get("source"),
                "study_type": label,
                "evidence_level": level,
                "citations": m.get("citations"),
                "is_open_access": m.get("is_open_access", False),
                "pmcid": m.get("pmcid"),
                "abstract": m.get("abstract"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "_rel_rank": rank,
            })

        for row in rows:
            row["rank_score"] = round(_rank_score(
                row["_rel_rank"], row["evidence_level"], row["citations"],
                row["year"], this_year), 4)
        rows.sort(key=lambda r: r["rank_score"], reverse=True)
        top = rows[:max_results]

        if fetch_fulltext:
            for row in top:
                if row.get("is_open_access") and row.get("pmcid"):
                    try:
                        ft = await _epmc_fulltext(client, row["pmcid"])
                    except httpx.HTTPError:
                        ft = ""
                    if ft:
                        row["fulltext_chars"] = len(ft)
                        row["fulltext"] = ft[:fulltext_max_chars]

    for row in top:
        row.pop("_rel_rank", None)
    return {
        "query": query,
        "pubmed_term": term,
        "filter_relaxed": filter_relaxed,
        "total_found": total,
        "count": len(top),
        "results": top,
    }
