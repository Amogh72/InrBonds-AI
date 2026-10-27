"""
Retrieval backend for the decomposition loop (orchestrator.py), standing in
for PIKE-RAG's own ChunkAtomRetriever (see
vendor/pikerag/NOTICE.md for why that wasn't vendored). Returns the same
AtomRetrievalInfo shape every decomposition prompt protocol in
vendor/pikerag/prompts/decomposition/atom_based.py is written against, but
sourced from this project's existing Qdrant-backed vector_search tool
(mcp_server/db/vector_db.py) instead of a second, purpose-built atom/chunk
vector store pair.

Honest adaptation, not a drop-in: upstream's `.atom` field is a distinct,
offline-pre-generated candidate sub-question per chunk (from an atomic-
question-generation preprocessing step this project doesn't have - see
NOTICE.md). Here, `.atom` is instead the sub-question text plus a label and
preview of whichever chunk matched it (see retrieve_atom_info_candidates
below) - distinguishable per candidate, since
atom_question_selection_protocol's candidate list is built from `.atom`
alone, but not an independently-phrased question the way upstream's is. So
the selection step's job becomes "pick the most useful retrieved chunk for
this sub-question" rather than upstream's "pick the most useful
pre-generated sub-question." Same mechanism (an LLM selection step over
several candidates), same prompts, slightly different meaning of what's
being selected among.

Only vector_search is used for now - sql_search_bond_terms/graph_query exist
and are citation-complete, but routing an LLM-generated sub-question string
to the right *structured* tool (which needs typed filters, not free text) is
a separate decision not made yet. See orchestrator.py's module docstring.
"""

import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent.parent
MCP_SERVER = ROOT / "mcp_server"
if str(MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER))

from config import qdrant_config  # noqa: E402
from db.vector_db import vector_db  # noqa: E402

from pikerag.retrieval_types import AtomRetrievalInfo  # noqa: E402

_embedding_model = None


def _get_embedding_model():
    """
    Self-contained copy of mcp_server/server.py's _get_embedding_model -
    deliberately not importing server.py itself here, since that would pull
    in the `mcp` protocol SDK as a dependency of this retrieval component,
    which has nothing to do with serving MCP tool calls.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(qdrant_config.embedding_model)
        actual_dim = _embedding_model.get_sentence_embedding_dimension()
        if actual_dim != qdrant_config.embedding_dim:
            raise RuntimeError(
                f"EMBEDDING_DIM in mcp_server/.env is {qdrant_config.embedding_dim} but "
                f"'{qdrant_config.embedding_model}' actually produces {actual_dim}-dim vectors."
            )
    return _embedding_model


def _embed_fn(text: str) -> list[float]:
    vector = _get_embedding_model().encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


vector_db.set_embed_fn(_embed_fn)


def _chunk_label(chunk: dict) -> str:
    parts = [chunk.get("document_name")]
    if chunk.get("start_page"):
        parts.append(f"p.{chunk['start_page']}")
    if chunk.get("section_path"):
        parts.append(chunk["section_path"])
    elif chunk.get("series_id") or chunk.get("category_id"):
        parts.append(f"{chunk.get('series_id') or ''} {chunk.get('category_id') or ''}".strip())
    return " — ".join(p for p in parts if p)


def retrieve_atom_info_candidates(
    query: str,
    top_k: int = 6,
    issuer_filter: str | None = None,
    document_name_filter: str | None = None,
) -> List[AtomRetrievalInfo]:
    """
    The one retrieval entry point orchestrator.py calls per sub-question.
    Wraps mcp_server's vector_search tool output into AtomRetrievalInfo
    objects so the vendored decomposition prompts can consume it unchanged.
    """
    result = vector_db.search(
        query_text=query,
        top_k=top_k,
        issuer_filter=issuer_filter,
        document_type_filter=None,
    )
    if not result.ok:
        return []

    candidates: List[AtomRetrievalInfo] = []
    for chunk in result.data:
        if document_name_filter and chunk.get("document_name") != document_name_filter:
            continue
        label = _chunk_label(chunk)
        text = chunk.get("text", "")
        # atom_question_selection_protocol's candidate list (vendor/pikerag/
        # prompts/decomposition/atom_based.py's AtomQuestionSelectionParser)
        # is built from `.atom` alone, not `.source_chunk` - if every
        # candidate for one sub-question shared the same `.atom` (the bare
        # query text), the LLM would see identical entries and have no way
        # to tell candidates apart. Folding in the chunk's own label and a
        # text preview makes each candidate distinguishable.
        preview = text[:150].replace("\n", " ").strip()
        candidates.append(
            AtomRetrievalInfo(
                atom_query=query,
                atom=f"{query} [{label}]: {preview}",
                source_chunk_title=label,
                source_chunk=text,
                source_chunk_id=chunk.get("chunk_id", ""),
                retrieval_score=chunk.get("score") or 0.0,
                atom_embedding=[],
            )
        )
    return candidates
