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

from google.genai.errors import ClientError  # noqa: E402

from llm_clients.anthropic_client import AnthropicClient  # noqa: E402
from llm_clients.gemini_client import GeminiClient, GeminiDailyQuotaExhausted, _is_daily_quota_exhaustion  # noqa: E402
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


class TestGeminiMessageTranslation:

    def test_system_message_extracted_separately(self):
        messages = [{"role": "system", "content": "sys prompt"}, {"role": "user", "content": "hello"}]
        system_prompt, contents = GeminiClient._split_system_and_messages(messages)
        assert system_prompt == "sys prompt"
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "hello"

    def test_assistant_role_becomes_model(self):
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello back"}]
        _, contents = GeminiClient._split_system_and_messages(messages)
        assert [c.role for c in contents] == ["user", "model"]

    def test_no_system_message_gives_empty_string(self):
        messages = [{"role": "user", "content": "hello"}]
        system_prompt, contents = GeminiClient._split_system_and_messages(messages)
        assert system_prompt == ""
        assert len(contents) == 1


class TestGeminiMarkdownFenceStripping:
    """
    Gemini routinely wraps JSON responses in a ```json fence even when not
    asked to, unlike Claude - vendor/pikerag/utils/json_parser.py's
    rfind-based extraction tolerates a fence around otherwise-valid JSON,
    but a fence paired with any other hiccup turns into a confusing
    stray-brace failure. _get_content_from_response strips the fence before
    content ever reaches the vendored parser.
    """

    def _client(self):
        return GeminiClient(api_key="fake-key-for-offline-test")

    class _FakeResponse:
        def __init__(self, text):
            self.text = text

    def test_strips_json_fence(self):
        content = self._client()._get_content_from_response(self._FakeResponse('```json\n{"a": 1}\n```'))
        assert content == '{"a": 1}'

    def test_strips_bare_fence_without_json_tag(self):
        content = self._client()._get_content_from_response(self._FakeResponse('```\n{"a": 1}\n```'))
        assert content == '{"a": 1}'

    def test_leaves_unfenced_content_unchanged(self):
        content = self._client()._get_content_from_response(self._FakeResponse('{"a": 1}'))
        assert content == '{"a": 1}'


class TestGeminiDailyQuotaDetection:
    """
    Seen live: a daily free-tier quota exhaustion (quotaId
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier") was retried 5
    times with increasing backoff (up to ~15 minutes total) by the generic
    429-is-worth-retrying path, when it can't possibly recover within that
    window - only a full daily reset (hours away) fixes it. These confirm
    the daily-vs-per-minute distinction is read correctly from a realistic
    error shape, since that's what decides whether retrying even happens.
    """

    def _quota_error(self, quota_id: str) -> ClientError:
        return ClientError(code=429, response_json={
            "error": {
                "code": 429, "message": "You exceeded your current quota...", "status": "RESOURCE_EXHAUSTED",
                "details": [{
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": quota_id}],
                }],
            },
        })

    def test_per_day_quota_is_detected(self):
        exc = self._quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        assert _is_daily_quota_exhaustion(exc) is True

    def test_per_minute_quota_is_not_flagged_as_daily(self):
        exc = self._quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
        assert _is_daily_quota_exhaustion(exc) is False

    def test_daily_quota_exhaustion_raises_immediately_without_retrying(self, monkeypatch):
        import time as time_module

        client = GeminiClient(api_key="fake-key-for-offline-test")
        monkeypatch.setattr(
            client._client.models, "generate_content",
            lambda **kwargs: (_ for _ in ()).throw(self._quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")),
        )
        # If this ever retried, it would call time.sleep() via BaseLLMClient._wait() -
        # failing that assertion is a faster, clearer signal than letting a
        # regression here actually sleep for minutes during a test run.
        monkeypatch.setattr(time_module, "sleep", lambda _seconds: (_ for _ in ()).throw(
            AssertionError("should not retry/sleep on a daily quota exhaustion")))

        try:
            client._get_response_with_messages([{"role": "user", "content": "hi"}], model="gemini-3.8-flash")
            assert False, "expected GeminiDailyQuotaExhausted to be raised"
        except GeminiDailyQuotaExhausted:
            pass


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


class _FakeLLMClient:
    """Stands in for a real BaseLLMClient - orchestrator.py only ever calls
    generate_content_with_messages() on its client, so that's all this needs
    to implement to drive DecompositionOrchestrator.answer() with no network
    or DB access."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def generate_content_with_messages(self, messages, **llm_config) -> str:
        return self._responses.pop(0)


class TestOrchestratorOnRoundCallback:
    """
    answer() can take minutes across several real decomposition rounds with
    no other signal of progress - on_round is query_service/app.py's hook
    for posting live status to a page polling a background job. These
    confirm it fires at the right points and never changes answer()'s
    actual control flow or return value, on the cheapest possible path
    (the LLM declines to decompose at all, so no retrieval/Qdrant is hit).
    """

    def _immediate_answer_client(self):
        return _FakeLLMClient([
            '{"thinking": "no decomposition needed", "sub_questions": []}',
            '{"answer": "42", "rationale": "because"}',
        ])

    def test_on_round_fires_for_proposal_and_final_answer(self):
        from orchestrator import DecompositionOrchestrator
        orchestrator = DecompositionOrchestrator(llm_client=self._immediate_answer_client(), llm_config={})

        calls = []
        orchestrator.answer("What is it?", on_round=lambda n, phase: calls.append((n, phase)))

        assert calls == [(1, "proposing sub-questions"), (1, "composing final answer")]

    def test_omitting_on_round_does_not_affect_the_result(self):
        from orchestrator import DecompositionOrchestrator
        orchestrator = DecompositionOrchestrator(llm_client=self._immediate_answer_client(), llm_config={})

        result = orchestrator.answer("What is it?")

        assert result["answer"] == "42"
