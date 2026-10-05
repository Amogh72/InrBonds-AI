"""
Fast, network-free tests for the pieces that don't need a live Qdrant or
Anthropic API: the Claude<->OpenAI-message-shape translation, and that the
vendored PIKE-RAG decomposition prompt protocols actually round-trip
correctly against this project's AtomRetrievalInfo objects (process_input
produces well-formed messages, parse_output correctly decodes a realistic
LLM JSON response) - this is what proves vendoring the prompts (rather than
the whole upstream repo) didn't silently break anything.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from llm_clients.anthropic_client import AnthropicClient  # noqa: E402
from pikerag.retrieval_types import AtomRetrievalInfo  # noqa: E402
from pikerag.prompts.decomposition import (  # noqa: E402
    question_decompose_protocol,
    atom_question_selection_protocol,
    final_qa_protocol,
)

from retrieval.mcp_retrieval import _chunk_label, retrieve_atom_info_candidates  # noqa: E402
from models import ToolSuccess  # noqa: E402 (mcp_server/models.py, on sys.path via mcp_retrieval's own import)


class TestAnthropicMessageTranslation:

    def test_system_message_extracted_separately(self):
        messages = [{"role": "system", "content": "sys prompt"}, {"role": "user", "content": "hello"}]
        system_prompt, chat = AnthropicClient._split_system_and_messages(messages)
        assert system_prompt == "sys prompt"
        assert chat == [{"role": "user", "content": "hello"}]

    def test_no_system_message_gives_empty_string(self):
        messages = [{"role": "user", "content": "hello"}]
        system_prompt, chat = AnthropicClient._split_system_and_messages(messages)
        assert system_prompt == ""
        assert chat == [{"role": "user", "content": "hello"}]


class TestChunkLabel:

    def test_includes_document_page_and_section(self):
        chunk = {"document_name": "pfc_ncd_2026.pdf", "start_page": 78, "section_path": "Series III"}
        assert _chunk_label(chunk) == "pfc_ncd_2026.pdf — p.78 — Series III"

    def test_falls_back_to_series_and_category_when_no_section_path(self):
        chunk = {"document_name": "pfc_ncd_2026.pdf", "start_page": 78, "series_id": "pfc_ncd_2026_series_III", "category_id": "III"}
        label = _chunk_label(chunk)
        assert "pfc_ncd_2026_series_III" in label and "III" in label


class TestRetrieveAtomInfoCandidates:
    """
    retrieve_atom_info_candidates() builds each candidate's `.atom` from the
    query plus a per-chunk label/preview specifically so that two chunks
    matching the same sub-question don't show up as identical, indistinguishable
    entries in atom_question_selection_protocol's candidate list - this is
    the fix that test_atom_question_selection_protocol_roundtrip's docstring
    refers to. No live Qdrant needed: db.vector_db.vector_db.search is
    monkeypatched directly.
    """

    def test_two_chunks_for_same_query_get_distinguishable_atoms(self, monkeypatch):
        import retrieval.mcp_retrieval as mcp_retrieval

        fake_result = ToolSuccess(
            tool="vector_search",
            data=[
                {"chunk_id": "c1", "document_name": "pfc_ncd_2026.pdf", "start_page": 78,
                 "section_path": "Series III", "text": "Series III is zero coupon.", "score": 0.9},
                {"chunk_id": "c2", "document_name": "pfc_ncd_2026.pdf", "start_page": 79,
                 "section_path": "Series IV", "text": "Series IV coupon is 7.05%.", "score": 0.8},
            ],
            citations=[],
            row_count=2,
        )
        monkeypatch.setattr(mcp_retrieval.vector_db, "search", lambda **kwargs: fake_result)

        candidates = retrieve_atom_info_candidates("What is the coupon?")

        assert len(candidates) == 2
        assert candidates[0].atom != candidates[1].atom
        assert "Series III" in candidates[0].atom
        assert "Series IV" in candidates[1].atom


class TestVendoredDecompositionProtocols:
    """
    Regression guard for vendor/pikerag/prompts/decomposition/atom_based.py's
    one adapted import (AtomRetrievalInfo now comes from
    pikerag.retrieval_types, not pikerag.knowledge_retrievers.chunk_atom_retriever -
    see vendor/pikerag/NOTICE.md) - these confirm every protocol still
    process_input/parse_output round-trips correctly against that dataclass.
    """

    def _sample_info(self, chunk_id="c1"):
        return AtomRetrievalInfo(
            atom_query="q", atom="q", source_chunk_title="pfc_ncd_2026.pdf p.78",
            source_chunk="Series III coupon is 6.9%", source_chunk_id=chunk_id,
            retrieval_score=0.9, atom_embedding=[],
        )

    def test_question_decompose_protocol_roundtrip(self):
        messages = question_decompose_protocol.process_input(content="What is the coupon?", chosen_atom_infos=[])
        assert messages[0]["role"] == "system"
        assert "What is the coupon?" in messages[1]["content"]

        fake_response = '{"thinking": "need coupon info", "sub_questions": ["series III coupon"]}'
        should_decompose, thinking, sub_questions = question_decompose_protocol.parse_output(fake_response)
        assert should_decompose is True
        assert sub_questions == ["series III coupon"]

    def test_question_decompose_protocol_stops_when_no_sub_questions(self):
        fake_response = '{"thinking": "have enough", "sub_questions": []}'
        should_decompose, _, sub_questions = question_decompose_protocol.parse_output(fake_response)
        assert should_decompose is False
        assert sub_questions == []

    def test_atom_question_selection_protocol_roundtrip(self):
        # The candidate list shown to the LLM is built from `.atom`, not
        # `.source_chunk` (see AtomQuestionSelectionParser.encode) - a
        # dedicated candidate here, distinct from _sample_info's default, so
        # this test actually exercises that the right field is used.
        info = AtomRetrievalInfo(
            atom_query="q", atom="distinguishable atom text", source_chunk_title="pfc_ncd_2026.pdf p.78",
            source_chunk="Series III coupon is 6.9%", source_chunk_id="c1", retrieval_score=0.9, atom_embedding=[],
        )
        messages = atom_question_selection_protocol.process_input(
            content="orig q", atom_info_candidates=[info], chosen_atom_infos=[],
        )
        assert "distinguishable atom text" in messages[1]["content"]

        fake_response = '{"thinking": "pick it", "question_idx": 1}'
        selected, _, chosen = atom_question_selection_protocol.parse_output(fake_response)
        assert selected is True
        assert chosen.source_chunk_id == "c1"

    def test_atom_question_selection_protocol_handles_out_of_range_index(self):
        info = self._sample_info()
        atom_question_selection_protocol.process_input(
            content="orig q", atom_info_candidates=[info], chosen_atom_infos=[],
        )
        fake_response = '{"thinking": "none useful", "question_idx": 99}'
        selected, _, chosen = atom_question_selection_protocol.parse_output(fake_response)
        assert selected is False
        assert chosen is None

    def test_final_qa_protocol_roundtrip(self):
        info = self._sample_info()
        messages = final_qa_protocol.process_input(content="orig q", chosen_atom_infos=[info])
        assert "Series III coupon is 6.9%" in messages[1]["content"]

        fake_response = '{"answer": "6.9%", "rationale": "from pfc_ncd_2026.pdf p.78"}'
        output = final_qa_protocol.parse_output(fake_response)
        assert output["answer"] == "6.9%"
