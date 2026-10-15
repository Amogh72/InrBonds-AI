# Provenance

The files in this `vendor/pikerag/` directory are copied (mostly verbatim,
two adapted as noted below) from Microsoft's **PIKE-RAG** project:

- Source: https://github.com/microsoft/PIKE-RAG
- Commit: `94e14c48170d63d90db659a544dd3d7c8287c0f3`
- License: MIT (see `LICENSE` in this directory, copied unchanged)
- Status: the upstream repository was **archived (read-only) by Microsoft on
  September 16, 2026** - no further updates will come from upstream. These
  files are a frozen snapshot, vendored rather than depended on live, so this
  project doesn't break if the archived repo is ever taken down.

## Why vendored, not a dependency

PIKE-RAG was never pip-installable (no `setup.py`/`pyproject.toml` upstream)
- it's a clone-and-run research codebase built around YAML-config-driven
workflows and a benchmark/evaluation harness. This project only needs its
**decomposition prompt protocols** (the actual engineered IP behind
"knowledge-aware task decomposition") and its **LLM client base interface**
- not its own retrieval backend (`ChunkAtomRetriever`, built around two
chromadb vector stores and an offline atomic-question-generation
preprocessing step that doesn't fit this project's data) or its benchmark
runner (YAML configs, `ExactMatch`/`F1` evaluators, JSONL test suites - built
for reproducing published benchmark numbers, not answering live questions).

See `../../README.md` for how this project's own code
(`../../llm_clients/anthropic_client.py`, `../../retrieval/mcp_retrieval.py`,
`../../orchestrator.py`) plugs into these vendored pieces.

## What's here, verbatim

- `prompts/{base_parser,message_template,protocol}.py` - the
  `CommunicationProtocol`/`MessageTemplate`/`BaseContentParser` machinery
  every prompt below is built on.
- `prompts/qa/generation.py` - the final free-text answer-generation prompt
  (`generation_qa_with_reference_protocol`'s template; the decomposition
  module below builds its own parser on top of it).
- `prompts/decomposition/atom_based.py` - the actual decomposition prompts:
  `question_decompose_protocol`, `atom_question_selection_protocol`,
  `chunk_selection_protocol`, `final_qa_protocol`.
- `utils/{json_parser,logger}.py` - small helpers the above depend on.

## What's trimmed or adapted (and why)

- `llm_client/base.py` - `BaseLLMClient`, the abstract client interface
  (`_get_response_with_messages`/`_get_content_from_response`) that
  `../../llm_clients/anthropic_client.py` implements. **One bug fixed**:
  upstream's `close()` calls `self._cache.save()` unconditionally, which
  raises `AttributeError` whenever a client is constructed with
  `location=None` - the class's own documented, legitimate "no cache" mode
  (`__init__`'s docstring literally says "No cache would be created if set
  to None"). Guarded with `if self._cache is not None`.
- `llm_client/__init__.py` - trimmed to export only `BaseLLMClient`. Upstream
  also eagerly imports `AzureMetaLlamaClient`/`AzureOpenAIClient`/
  `HFMetaLlamaClient`/`StandardOpenAIClient`, pulling in `openai`/
  `transformers`/`torch` - unused here, since this project uses Claude.
- `prompts/qa/__init__.py` - trimmed to drop `multiple_choice.py` (unused;
  this project's final answer step is free-text, not multiple choice).
- `prompts/decomposition/atom_based.py` - **one import line changed**: now
  imports `AtomRetrievalInfo` from `../retrieval_types.py` (a standalone
  copy of just that dataclass) instead of
  `pikerag.knowledge_retrievers.chunk_atom_retriever` (the full
  `ChunkAtomRetriever` class, unused - see `retrieval_types.py`'s own
  docstring). The dataclass's fields are unchanged, so every prompt and
  parser in the file is otherwise identical to upstream.
- `retrieval_types.py` - **new file**, not upstream: just the
  `AtomRetrievalInfo` dataclass, extracted so the decomposition prompts can
  be used without their original retriever backend.

Everything not listed above (the knowledge_retrievers package itself,
workflows/, the YAML-config CLI runner, the evaluation harness, every
provider-specific LLM client) was intentionally left out, not merely
forgotten.
