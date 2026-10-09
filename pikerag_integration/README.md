# PIKE-RAG Integration

Wires Microsoft's **PIKE-RAG** decomposition approach (knowledge-aware task
decomposition - sPecIalized KnowledgE and Rationale Augmented Generation,
[arXiv:2501.11551](https://arxiv.org/abs/2501.11551)) to Claude and to this
project's existing retrieval stack (`mcp_server/`), instead of depending on
PIKE-RAG's own codebase live.

## Why this isn't just `pip install pikerag; import pikerag`

Two real constraints, not a style choice:

1. **It was never pip-installable.** The upstream repo
   (`github.com/microsoft/PIKE-RAG`) has no `setup.py`/`pyproject.toml` - it's
   a clone-and-run research codebase built around YAML-config-driven
   benchmark workflows.
2. **It was archived (read-only) by Microsoft on September 16, 2026.** No
   further updates will come from upstream.

So the specific files actually needed - the decomposition prompt protocols
(the real engineered IP: "analyse the context, propose sub-questions,
retrieve, select the most useful one, repeat") and the LLM client base
interface - are **vendored** (copied, with attribution) into
`vendor/pikerag/` as a frozen snapshot, rather than depended on live. See
`vendor/pikerag/NOTICE.md` for exactly what was copied verbatim, what was
trimmed, and the one real bug fixed along the way.

**What wasn't vendored, and why:** PIKE-RAG's own retrieval backend
(`ChunkAtomRetriever`) is built around two purpose-built Chroma vector
stores (one of chunks, one of offline-pre-generated "atomic questions" per
chunk) and an offline preprocessing step this project's data has never gone
through. Rather than rebuild that preprocessing pipeline, `retrieval/
mcp_retrieval.py` retrieves through this project's **existing**
`vector_search` tool (`mcp_server/db/vector_db.py`, backed by Qdrant,
already indexed with every document's chunks) and adapts the result shape
to match what PIKE-RAG's prompts expect. See that file's docstring for
exactly how the adaptation works and what's honestly different from
upstream's intent.

## Architecture

```
cli.py
  -> orchestrator.DecompositionOrchestrator.answer(question)
       -> vendor/pikerag/prompts/decomposition/atom_based.py's 4 protocols
            (question_decompose_protocol, atom_question_selection_protocol,
             final_qa_protocol - all PIKE-RAG's own prompts, unchanged)
       -> retrieval/mcp_retrieval.py
            -> mcp_server/db/vector_db.py's vector_search (Qdrant)
       -> llm_clients/anthropic_client.AnthropicClient
            -> Claude via the Anthropic Messages API
            (implements vendor/pikerag/llm_client/base.py's BaseLLMClient -
             the one interface PIKE-RAG never shipped a Claude client for)
```

The loop itself (propose sub-questions -> retrieve -> select -> repeat ->
answer) mirrors upstream's `QaDecompositionWorkflow.answer()` method, just
rewritten in `orchestrator.py` instead of reused directly - that class's
`__init__` hard-asserts `isinstance(self._retriever, ChunkAtomRetriever)`,
so reusing it as-is wasn't possible without building that retriever too.

## Setup

```bash
cd pikerag_integration
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY or GEMINI_API_KEY (see below)
```

Requires `mcp_server/.env` already set up and Qdrant reachable (`docker
compose up -d` from `mcp_server/`), with at least one document already
loaded via `mcp_server/scripts/load_canonical_to_qdrant.py` (or the
`upload_service/` UI) - this component only reads from Qdrant, it doesn't
load anything itself.

### LLM provider: Anthropic (default) or Gemini (free tier)

The decomposition loop talks to whatever implements `vendor/pikerag/
llm_client/base.py`'s `BaseLLMClient` interface - two implementations ship
here:

- `llm_clients/anthropic_client.py` (Claude, via the Anthropic Messages API)
- `llm_clients/gemini_client.py` (Gemini, via Google's `google-genai` SDK)

Anthropic's API has no free tier; Gemini's Flash models do (get a key at
[aistudio.google.com](https://aistudio.google.com) - check current
free-tier model availability there, it changes). Pick one with `--provider`
or `LLM_PROVIDER` in `.env`:

```bash
python cli.py "..." --provider gemini   # or set LLM_PROVIDER=gemini in .env
```

Only the chosen provider's API key needs to be set; the other can stay
empty in `.env`.

## Run

```bash
python cli.py "What is PFC's Series III coupon?"
python cli.py "Compare PFC's and IIFL's debenture trustee" --show-trace
python cli.py "What is PFC's Series III coupon?" --provider gemini
```

`--show-trace` prints the full decompose/retrieve/select trace for every
sub-question round, not just the final answer - useful for seeing the
decomposition actually working rather than taking it on faith.

## Current scope and honest limitations

- **Single retrieval tool.** Sub-questions are only ever routed through
  `vector_search`. `sql_search_bond_terms`/`graph_query` exist and are
  citation-complete, but routing an LLM-generated sub-question string to the
  right *structured* tool (which needs typed filters, not free text) is a
  separate decision, not implemented here yet - see `retrieval/
  mcp_retrieval.py`'s module docstring.
- **No multi-document routing.** A question spanning several issuers (e.g.
  "which issuers has Beacon Trusteeship served") will retrieve whatever
  `vector_search` ranks highest across all loaded documents, not
  deliberately fan out per-issuer - `graph_find_issues_by_participant` would
  answer that specific shape of question far more precisely and is not yet
  wired in.
- **Not load-tested.** Works against the single-question CLI case; no
  concurrency, rate-limit handling beyond `AnthropicClient`'s basic retry, or
  cost/latency benchmarking has been done yet.
