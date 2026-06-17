# paper-mcp

<!-- mcp-name: io.github.MCPServings/paper-mcp -->

Remotely-callable **MCP server for academic paper search, full-text retrieval & image→LaTeX**, served at `https://latex-tools.online/mcp`.

Three corpora behind one normalized interface:
- **`arxiv`** (default) — search, metadata, and **full-text** (HTML / markdown / LaTeX source)
- **`semanticscholar`** (alias `s2`) — the full S2 API surface: citation graph, authors, recommendations, full-text snippets, bulk datasets
- **`openalex`** (alias `oa`) — 316M all-field works: citation graph, authors with h-index, institutions, topics, influence metrics

Plus a **unified `search_all`** that fuses all three corpora, **image→LaTeX** OCR, and **LaTeX lint + PDF→text** tooling.

---

## Tools (41)

### Generic / source-agnostic (8)
| Tool | Purpose |
|---|---|
| **`search_all(query, max_results=10, sources='arxiv,semanticscholar,openalex')`** | **Unified search.** Fans out to all three corpora concurrently, de-duplicates the same work (by DOI/title) and re-ranks with Reciprocal Rank Fusion. Each hit carries `sources` (who found it) + an `ids` map for follow-up calls. Prefer this for broad lookups. |
| `search_papers(query, source='arxiv', max_results=10, sort_by='relevance')` | Single-corpus search. arXiv `query` accepts plain text or field syntax (`ti:` `au:` `cat:cs.CL` `abs:` + AND/OR). |
| `get_paper(paper_id, source='arxiv')` | One paper's full record. S2 id accepts S2 id / `DOI:` / `ARXIV:` / `CorpusId:`. |
| `search_by_author(author, source='arxiv')` | Papers by author, newest first. |
| `list_recent(category, source='arxiv')` | Latest in a category (arXiv code or S2 field of study). |
| `list_categories(source='arxiv')` | Common category codes. |
| **`read_paper(paper_id, format='markdown')`** | **FULL text (arXiv).** `markdown` = body with formulas as `$LaTeX$`; `html` = raw LaTeXML page; `latex` = original manuscript `.tex` source. |
| `list_paper_sources()` | Available corpora. |

`read_paper` fetch chain: `arxiv.org/html/{id}` → `ar5iv` fallback (markdown/html), or `arxiv.org/e-print/{id}` tarball main `.tex` (latex). Formulas are recovered from the LaTeXML `alttext` invariant.

### Medical / evidence-graded (1)
| Tool | Purpose |
|---|---|
| **`search_medical(query, study_types='rct,meta-analysis,systematic-review', year_from=0, max_results=10, fetch_fulltext=True)`** | **Clinical literature search.** Queries PubMed, filters by research type via Publication-Type tags and re-ranks by the **evidence pyramid** (meta-analysis / systematic review > RCT > cohort > ...), so real trials surface above high-cited reviews/guidelines that pure-citation ranking floats up. Open-access full text is attached from Europe PMC by PMID. If the type filter yields nothing it auto-relaxes (flagged `filter_relaxed`). `query` is English keyword/boolean text — do NL/multilingual query understanding upstream. Backed by NCBI E-utilities + Europe PMC (both free, no key required). |

### Image → LaTeX (3)
Turn a formula or table image back into LaTeX (e.g. a figure cropped from a paper) without needing your own vision model. Backed by the co-located recognize service (PaddleOCR-VL / DeepSeek-OCR / texify).
| Tool | Purpose |
|---|---|
| `recognize_formula(image_url=... or image_base64=..., model='deepseek-ocr')` | Formula image → LaTeX. `image_url` is downloaded server-side (with SSRF guards). Returns `{latex, model, elapsed_ms}`. |
| `recognize_table(image_url=... or image_base64=..., model='deepseek-ocr')` | Table image → LaTeX `tabular`. |
| `list_ocr_models()` | Available OCR models (`deepseek-ocr`, `paddleocr-vl`, `texify`). |

### LaTeX tooling (3)
Companions to the LaTeX/PDF web tools at `latex-tools.online` — same backends, exposed over MCP.
| Tool | Purpose |
|---|---|
| `lint_latex(code)` | Check a LaTeX snippet for errors and return an auto-fixed version. Returns `{errors, fixed_code, summary_en, summary_zh, elapsed_ms}`. |
| `extract_pdf(pdf_url=... or pdf_base64=..., formula=True, table=True)` | PDF → clean Markdown/LaTeX text via MinerU (useful for papers with no open-access full text). `pdf_url` is downloaded server-side (SSRF-guarded). Content-addressed + cached: a recently-seen or small PDF returns `content` in one call; a fresh PDF (MinerU is GPU-heavy, minutes) returns `status='running'` + a `task_id`. |
| `extract_pdf_result(task_id)` | Fetch an `extract_pdf` job by `task_id`. Returns `content` once `status='done'`; while `'running'`, `content` is null — call again shortly. |

### OpenAlex (8)
- **Works:** `get_openalex_work` · `get_openalex_citations` · `get_openalex_references` · `search_openalex_works` (filters: year range, open-access, min-citations, institution)
- **Authors/Institutions:** `search_openalex_authors` · `search_openalex_institutions`
- **Analytics:** `get_openalex_trends` · `list_openalex_topics`

### Semantic Scholar (18)
- **Graph:** `get_paper_citations` · `get_paper_references` · `get_paper_authors`
- **Lookup:** `match_paper_title` · `autocomplete_papers`
- **Bulk:** `search_papers_bulk` (≤1000, sortable, token paging) · `get_papers_batch`
- **Authors:** `search_authors` · `get_author` · `get_author_papers` · `get_authors_batch`
- **Full-text:** `search_snippets` (search inside paper body)
- **Recommend:** `recommend_papers_for_paper` · `recommend_papers_from_examples`
- **Datasets:** `list_dataset_releases` · `get_dataset_release` · `get_dataset_download_links` · `get_dataset_diffs`

---

## Layout
```
paper_mcp/
  server.py            FastMCP server (tool registrations + instructions)
  models.py            normalized Paper model
  aggregate.py         cross-source fusion (dedup + Reciprocal Rank Fusion)
  sources/
    base.py            source registry (get_source / list_sources)
    arxiv.py           arXiv Atom API + read_paper (HTML/markdown/latex)
    semanticscholar.py Semantic Scholar full API surface
    openalex.py        OpenAlex REST API (works/authors/institutions/topics)
    recognize.py       image→LaTeX client over the co-located recognize service
    latextools.py      lint + PDF-extract clients over the latex-tools services
pyproject.toml
```

## Run locally
```bash
cd paper-mcp
python -m venv .venv && . .venv/bin/activate
pip install -e .
PAPER_MCP_PORT=9400 python -m paper_mcp.server
# MCP endpoint at http://127.0.0.1:9400/mcp (JSON-RPC; a plain GET returns 406)
```

### Env
| Var | Default | Notes |
|---|---|---|
| `PAPER_MCP_HOST` | `127.0.0.1` | |
| `PAPER_MCP_PORT` | `9400` | |
| `PAPER_MCP_PATH` | `/mcp` | |
| `SEMANTIC_SCHOLAR_API_KEY` | — | optional; raises S2 rate limit. Set via `/etc/paper-mcp.env` in prod. |

---

## Deployment (latex-tools.online)
- Runs as `paper-mcp.service` on the **latex-tools** server, WorkingDirectory `/opt/paper-mcp`, port 9400.
- nginx reverse-proxies `https://latex-tools.online/mcp` → `127.0.0.1:9400/mcp`.
- Secrets in `/etc/paper-mcp.env` (`SEMANTIC_SCHOLAR_API_KEY`).
- systemd unit + env are backed up under `../deploy/` in this repo.

### Update flow
This repo is the source of truth. The server runs an **independent copy** under `/opt/paper-mcp` (not auto-synced):
```bash
# edit here → push → deploy
scp -r paper_mcp/* latex-tools:/opt/paper-mcp/paper_mcp/
ssh latex-tools 'systemctl restart paper-mcp'
ssh latex-tools 'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:9400/mcp'  # 406 = healthy (needs JSON-RPC handshake)
```

## Notes
- arXiv calls are politely rate-limited + retried (`_USER_AGENT`, backoff).
- `read_paper` covers ~80%+ of papers via official HTML; older scan-only papers may have no full text.
- Moved here from the `docs` repo on 2026-06-07; that copy is gone.

---

## License

MIT © MCPServings. See [LICENSE](LICENSE).
