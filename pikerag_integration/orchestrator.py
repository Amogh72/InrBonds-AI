"""
The decompose -> retrieve -> select -> answer loop: PIKE-RAG's actual
"knowledge-aware task decomposition" mechanism (the thing the company asked
for specifically), reimplemented over this project's own MCP tools instead
of upstream's ChunkAtomRetriever - see vendor/pikerag/NOTICE.md for why that
retrieval backend wasn't vendored, and retrieval/mcp_retrieval.py for the
honest adaptation that was.

This mirrors the control flow of upstream's QaDecompositionWorkflow.answer()
(pikerag/workflows/qa_decompose.py in the source repo - not vendored here,
since it's a method on a class whose __init__ hard-asserts
isinstance(self._retriever, ChunkAtomRetriever), so reusing the class itself
isn't an option without that retriever). The four prompt protocols it calls
(question_decompose_protocol, atom_question_selection_protocol,
chunk_selection_protocol, final_qa_protocol) ARE vendored, unchanged in
behavior from upstream - this file supplies the loop around them, swapping
in this project's retrieval and LLM client.

Each step, concretely:
  1. Proposal:  ask the LLM for sub-questions, given the original question
     and whatever context has been chosen so far.
  2. Retrieval: run each proposed sub-question through vector_search
     (mcp_retrieval.py), scoped to this project's chunks.
  3. Selection: ask the LLM to pick the single most useful retrieved chunk
     from the candidates, given what's already been chosen.
  4. Repeat until the LLM says no more decomposition is needed, no
     candidates are found, or max_num_question is reached.
  5. Answer the original question using every chunk chosen along the way,
     via final_qa_protocol.
"""

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "pikerag_integration" / "vendor") not in sys.path:
    sys.path.insert(0, str(ROOT / "pikerag_integration" / "vendor"))

from pikerag.llm_client.base import BaseLLMClient  # noqa: E402
from pikerag.retrieval_types import AtomRetrievalInfo  # noqa: E402
from pikerag.prompts.decomposition import (  # noqa: E402
    question_decompose_protocol,
    atom_question_selection_protocol,
    final_qa_protocol,
)

from retrieval.mcp_retrieval import retrieve_atom_info_candidates  # noqa: E402

DEFAULT_MAX_SUB_QUESTIONS = 5


class DecompositionOrchestrator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        llm_config: dict,
        max_sub_questions: int = DEFAULT_MAX_SUB_QUESTIONS,
        retrieve_top_k: int = 6,
    ) -> None:
        self._client = llm_client
        self._llm_config = llm_config
        self._max_sub_questions = max_sub_questions
        self._retrieve_top_k = retrieve_top_k

    def _propose_sub_questions(self, question: str, chosen: list[AtomRetrievalInfo]) -> tuple[bool, str, list[str]]:
        messages = question_decompose_protocol.process_input(content=question, chosen_atom_infos=chosen)
        content = self._client.generate_content_with_messages(messages, **self._llm_config)
        return question_decompose_protocol.parse_output(content)

    def _retrieve_candidates(
        self, proposals: list[str], original_question: str, chosen: list[AtomRetrievalInfo],
    ) -> list[AtomRetrievalInfo]:
        chosen_chunk_ids = {info.source_chunk_id for info in chosen}

        candidates: list[AtomRetrievalInfo] = []
        seen_chunk_ids: set[str] = set()
        for proposal in proposals:
            for candidate in retrieve_atom_info_candidates(proposal, top_k=self._retrieve_top_k):
                if candidate.source_chunk_id in chosen_chunk_ids or candidate.source_chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(candidate.source_chunk_id)
                candidates.append(candidate)

        # Backup: if the proposed sub-questions found nothing new, fall back
        # to searching with the original question directly.
        if not candidates:
            for candidate in retrieve_atom_info_candidates(original_question, top_k=self._retrieve_top_k):
                if candidate.source_chunk_id in chosen_chunk_ids or candidate.source_chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(candidate.source_chunk_id)
                candidates.append(candidate)

        return candidates

    def _select_candidate(
        self, question: str, candidates: list[AtomRetrievalInfo], chosen: list[AtomRetrievalInfo],
    ) -> tuple[bool, str, AtomRetrievalInfo | None]:
        messages = atom_question_selection_protocol.process_input(
            content=question, atom_info_candidates=candidates, chosen_atom_infos=chosen,
        )
        content = self._client.generate_content_with_messages(messages, **self._llm_config)
        return atom_question_selection_protocol.parse_output(content)

    def _answer_with_context(self, question: str, chosen: list[AtomRetrievalInfo]) -> dict:
        messages = final_qa_protocol.process_input(content=question, chosen_atom_infos=chosen)
        content = self._client.generate_content_with_messages(messages, **self._llm_config)
        output = final_qa_protocol.parse_output(content)
        output.setdefault("response", content)
        return output

    def answer(self, question: str) -> dict[str, Any]:
        trace: dict[str, dict] = {}
        chosen: list[AtomRetrievalInfo] = []

        while len(chosen) < self._max_sub_questions:
            step_id = f"Sub{len(chosen) + 1}"
            trace[step_id] = {}

            should_decompose, thinking, proposals = self._propose_sub_questions(question, chosen)
            trace[step_id]["proposal"] = {"decompose": should_decompose, "thinking": thinking, "sub_questions": proposals}
            if not should_decompose:
                break

            candidates = self._retrieve_candidates(proposals, question, chosen)
            trace[step_id]["retrieval"] = [
                {"source": info.source_chunk_title, "chunk_id": info.source_chunk_id, "score": info.retrieval_score}
                for info in candidates
            ]
            if not candidates:
                break

            selected, thinking, chosen_info = self._select_candidate(question, candidates, chosen)
            trace[step_id]["selection"] = {"selected": selected, "thinking": thinking}
            if not selected or chosen_info is None:
                break

            chosen.append(chosen_info)
            trace[step_id]["selection"]["chosen"] = chosen_info.source_chunk_title

        output = self._answer_with_context(question, chosen)
        output["decomposition_trace"] = trace
        output["citations"] = [
            {"source": info.source_chunk_title, "chunk_id": info.source_chunk_id}
            for info in chosen
        ]
        return output
