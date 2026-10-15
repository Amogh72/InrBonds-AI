"""
Routes a sub-question to MySQL (sql_search_bond_terms / sql_get_issue_overview)
and/or Neo4j (graph_find_issues_by_participant) in addition to vector_search,
when the sub-question's text matches this project's small, closed
vocabulary - the handful of issuer and participant (trustee/lead-manager/
rating-agency) names already known from the loaded documents, plus a
generic "Series <roman numeral> / Category <roman numeral>" pattern.

Unlike an open-domain system, rule-based matching is appropriate here
specifically because every entity that could appear is already known in
full from the loaded data, not inferred at runtime - there are only ever
as many issuers/participants as documents have been loaded, and they're
one SQL call away to enumerate (see _get_known_entities below).

This also deliberately adds zero extra LLM round-trips: today's retrieval
(mcp_retrieval.py's retrieve_atom_info_candidates) is pure local work -
embed the sub-question, query Qdrant - and this stays that way, just with
two more local, non-LLM sources feeding the same candidate pool. An
LLM-driven tool-router (ask the model which tool fits, per sub-question)
was the more "standard" design, but costs one extra LLM call per
sub-question - 6-8 extra calls per round, on top of the propose/select
calls already there - which isn't a reasonable tradeoff against how
unreliable the free-tier Gemini endpoint has been.

Results are converted into the same AtomRetrievalInfo shape vector_search
produces (see mcp_retrieval.py's own docstring for why that shape was
chosen) so they can be merged into one candidate pool and compete for
selection via the SAME, unchanged vendored selection prompt
(atom_question_selection_protocol) - no change needed there, or to the
decomposition loop's control flow in orchestrator.py beyond chaining this
module's output alongside retrieve_atom_info_candidates'.

Structured candidates are additive, never a replacement: vector_search
still always runs for every sub-question regardless of what this module
finds, so wiring this in can only improve recall over the previous
vector-search-only behavior, never regress it. A sub-question that
matches nothing in the known vocabulary (the common case for open-ended/
conceptual questions) costs nothing extra here - this returns an empty
list immediately after one already-cached vocabulary lookup.
"""

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
MCP_SERVER = ROOT / "mcp_server"
if str(MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(MCP_SERVER))

from db.sql_db import sql_db  # noqa: E402
from db.graph_db import graph_db  # noqa: E402

from pikerag.retrieval_types import AtomRetrievalInfo  # noqa: E402

_SERIES_RE = re.compile(r"\bseries\s+([ivx]+)\b", re.IGNORECASE)
_CATEGORY_RE = re.compile(r"\bcategor(?:y|ies)\s+([ivx]+)\b", re.IGNORECASE)

# Keywords from sql_get_issue_overview's own "Use for:" docstring - kept in
# sync with that by eye, not by import, since one's a docstring and the
# other's a routing list.
_ISSUE_OVERVIEW_KEYWORDS = (
    "trustee", "shelf limit", "green shoe", "rating", "lead manager",
    "listing", "exchange", "issue size", "security cover", "issue open",
    "issue close", "allotment",
)

_entity_cache: dict[str, Any] | None = None

# Trailing corporate-suffix words stripped repeatedly from a known entity's
# full name to get its "core" name - real questions say "Beacon Trusteeship"
# or "PFC's debenture trustee", never the full "Beacon Trusteeship Limited"
# or "Power Finance Corporation Limited". This handles suffix-stripping
# ("Beacon Trusteeship Services Limited" -> "Beacon Trusteeship"); the
# filename-derived alias below handles true acronyms ("PFC") that aren't a
# prefix of the full name at all.
_SUFFIX_WORDS = {"limited", "ltd", "services", "pvt", "private"}


def _core_name(name: str) -> str:
    words = name.split()
    while len(words) > 1 and words[-1].lower().strip(".") in _SUFFIX_WORDS:
        words.pop()
    return " ".join(words)


def _match_forms(name: str) -> set[str]:
    """Every form of a known entity name worth checking for in free text:
    the full name, its corporate-suffix-stripped core, and (if distinctive
    enough) its first word alone - covers "Power Finance Corporation
    Limited" referred to as "Power Finance Corporation", and "CRISIL
    Ratings Limited" referred to as bare "CRISIL"."""
    forms = {name, _core_name(name)}
    first_word = name.split()[0] if name.split() else ""
    if len(first_word) > 3:
        forms.add(first_word)
    return forms


def _get_known_entities() -> dict[str, Any]:
    """
    Discovers this project's issuer/participant vocabulary once, via a
    single sql_get_issue_overview() call with no filters (returns every
    loaded issue, each already carrying its debenture trustee plus nested
    ratings/lead_managers) - cheaper than writing dedicated "list distinct
    X" SQL, and already exactly the data needed.

    Cached for the life of the process. If a document gets loaded mid-
    process, this goes stale until restart - the same staleness window
    every other in-process cache in this codebase already has (e.g. the
    lazily-loaded embedding model), not a new limitation this introduces.
    """
    global _entity_cache
    if _entity_cache is not None:
        return _entity_cache

    issuers: set[str] = set()
    participants: set[str] = set()
    # issuer name -> set of match forms, including a document-filename-
    # derived short alias (e.g. "pfc" from "pfc_ncd_2026.pdf") - this is
    # how real acronyms like "PFC"/"IIFL" get covered generically, for
    # whatever documents happen to be loaded, without hardcoding any
    # specific issuer's name here.
    issuer_forms: dict[str, set[str]] = {}

    result = sql_db.get_issue_overview()
    if result.ok:
        for row in result.data:
            issuer = row.get("issuer")
            if issuer:
                issuers.add(issuer)
                forms = issuer_forms.setdefault(issuer, _match_forms(issuer))
                document_name = row.get("document_name")
                if document_name:
                    stem = Path(document_name).stem
                    alias = stem.split("_")[0]
                    if len(alias) > 2:
                        forms.add(alias)
            if row.get("debenture_trustee"):
                participants.add(row["debenture_trustee"])
            for rating in row.get("ratings") or []:
                if rating.get("agency"):
                    participants.add(rating["agency"])
            for lead_manager in row.get("lead_managers") or []:
                participants.add(lead_manager)

    _entity_cache = {"issuers": issuers, "issuer_forms": issuer_forms, "participants": participants}
    return _entity_cache


def _find_mentioned(text: str, candidates: set[str], forms_by_candidate: dict[str, set[str]] | None = None) -> str | None:
    """
    Whole-word, case-insensitive match of any known entity against the
    sub-question text, trying every match form for each candidate (see
    _match_forms) - not just its literal full name. Longest matching form
    wins, so a more specific/complete reference is preferred over a
    shorter one that happens to also match.
    """
    best_candidate, best_len = None, 0
    for candidate in candidates:
        forms = forms_by_candidate.get(candidate, {candidate}) if forms_by_candidate else _match_forms(candidate)
        for form in forms:
            if re.search(r"\b" + re.escape(form) + r"\b", text, re.IGNORECASE) and len(form) > best_len:
                best_candidate, best_len = candidate, len(form)
    return best_candidate


def _format_bond_terms_row(row: dict[str, Any]) -> str:
    parts = [f"{row.get('issuer')} Series {row.get('series_code')}"]
    if row.get("category_code"):
        parts[0] += f" Category {row['category_code']}"
    facts = []
    if row.get("tenor"):
        facts.append(f"tenor {row['tenor']}")
    if row.get("coupon_type"):
        facts.append(f"{row['coupon_type']}" + (f" ({row['frequency']})" if row.get("frequency") else ""))
    if row.get("face_value_inr") is not None:
        facts.append(f"face value INR {row['face_value_inr']}")
    if row.get("coupon_percent") is not None:
        facts.append(f"coupon {row['coupon_percent']}%")
    if row.get("effective_yield_percent") is not None:
        facts.append(f"effective yield {row['effective_yield_percent']}%")
    if row.get("issue_price_inr") is not None:
        facts.append(f"issue price INR {row['issue_price_inr']}")
    if row.get("maturity_amount_inr") is not None:
        facts.append(f"maturity amount INR {row['maturity_amount_inr']}")
    if row.get("nature_of_indebtedness"):
        facts.append(row["nature_of_indebtedness"])
    return parts[0] + ": " + ", ".join(facts) + "." if facts else parts[0] + "."


def _format_issue_overview_row(row: dict[str, Any]) -> str:
    facts = []
    if row.get("issue_size"):
        facts.append(f"issue size {row['issue_size']}")
    if row.get("debenture_trustee"):
        facts.append(f"debenture trustee {row['debenture_trustee']}")
    if row.get("security_type"):
        facts.append(f"security type {row['security_type']}")
    if row.get("security_cover"):
        facts.append(f"security cover {row['security_cover']}")
    if row.get("shelf_limit"):
        facts.append(f"shelf limit {row['shelf_limit']}")
    if row.get("green_shoe_option"):
        facts.append(f"green shoe option {row['green_shoe_option']}")
    if row.get("listing"):
        facts.append(f"listed on {row['listing']}" + (f" ({row['exchange']})" if row.get("exchange") else ""))
    if row.get("issue_open_date") or row.get("issue_close_date"):
        facts.append(f"open {row.get('issue_open_date')} to {row.get('issue_close_date')}")
    ratings = row.get("ratings") or []
    if ratings:
        rating_strs = [f"{r.get('agency')}: {r.get('rating')}" for r in ratings if r.get("agency")]
        if rating_strs:
            facts.append("ratings - " + "; ".join(rating_strs))
    lead_managers = row.get("lead_managers") or []
    if lead_managers:
        facts.append("lead managers - " + ", ".join(lead_managers))
    prefix = f"{row.get('issuer')} ({row.get('issue_name')})"
    return prefix + ": " + ", ".join(facts) + "." if facts else prefix + "."


def _format_participant_row(row: dict[str, Any]) -> str:
    return (
        f"{row.get('issuer')} ({row.get('issue_name')}, {row.get('document_name')}): "
        f"{row.get('participant_name')} served as {row.get('participant_role')}."
    )


def _bond_terms_candidates(sub_question: str, issuer: str | None) -> list[AtomRetrievalInfo]:
    series_match = _SERIES_RE.search(sub_question)
    category_match = _CATEGORY_RE.search(sub_question)
    if not series_match and not category_match:
        return []

    result = sql_db.search_bond_terms(
        issuer=issuer,
        series_code=series_match.group(1).upper() if series_match else None,
        category_code=category_match.group(1).upper() if category_match else None,
    )
    if not result.ok:
        return []

    candidates = []
    for row, citation in zip(result.data, result.citations):
        text = _format_bond_terms_row(row)
        label = f"{row.get('document_name')} — {citation.section}"
        candidates.append(
            AtomRetrievalInfo(
                atom_query=sub_question,
                atom=f"[SQL] {sub_question} [{label}]: {text}",
                source_chunk_title=f"[SQL] {label}",
                source_chunk=text,
                source_chunk_id=f"sql:{citation.source_id}",
                retrieval_score=1.0,
                atom_embedding=[],
            )
        )
    return candidates


def _issue_overview_candidates(sub_question: str, issuer: str | None) -> list[AtomRetrievalInfo]:
    if not issuer or not any(kw in sub_question.lower() for kw in _ISSUE_OVERVIEW_KEYWORDS):
        return []

    result = sql_db.get_issue_overview(issuer=issuer)
    if not result.ok:
        return []

    candidates = []
    for row, citation in zip(result.data, result.citations):
        text = _format_issue_overview_row(row)
        label = f"{row.get('document_name')} — {citation.section}"
        candidates.append(
            AtomRetrievalInfo(
                atom_query=sub_question,
                atom=f"[SQL] {sub_question} [{label}]: {text}",
                source_chunk_title=f"[SQL] {label}",
                source_chunk=text,
                source_chunk_id=f"sql:{citation.source_id}",
                retrieval_score=1.0,
                atom_embedding=[],
            )
        )
    return candidates


def _participant_candidates(sub_question: str, participants: set[str]) -> list[AtomRetrievalInfo]:
    participant = _find_mentioned(sub_question, participants)
    if not participant:
        return []

    result = graph_db.find_issues_by_participant(participant)
    if not result.ok:
        return []

    candidates = []
    for row in result.data:
        text = _format_participant_row(row)
        label = f"{row.get('document_name')} — {row.get('participant_role')}"
        # Graph citations are only numbered per-call ("cypher_result:0"),
        # not a stable fact identifier, so a chunk_id built from the row's
        # own content (issue_id + role + participant name) is used instead -
        # otherwise two different graph lookups across different rounds
        # could both produce "cypher_result:0" and get wrongly treated as
        # the same already-seen candidate by orchestrator.py's dedup logic.
        chunk_id = f"graph:{row.get('issue_id')}:{row.get('participant_role')}:{row.get('participant_name')}"
        candidates.append(
            AtomRetrievalInfo(
                atom_query=sub_question,
                atom=f"[Graph] {sub_question} [{label}]: {text}",
                source_chunk_title=f"[Graph] {label}",
                source_chunk=text,
                source_chunk_id=chunk_id,
                retrieval_score=1.0,
                atom_embedding=[],
            )
        )
    return candidates


def retrieve_structured_candidates(sub_question: str) -> list[AtomRetrievalInfo]:
    """
    The entry point orchestrator.py calls alongside
    mcp_retrieval.retrieve_atom_info_candidates() for every proposed
    sub-question. Returns [] if nothing in the sub-question matches this
    project's known vocabulary - the common case for open-ended/
    conceptual sub-questions, which stay exactly as semantic-search-only
    as they were before this module existed.
    """
    entities = _get_known_entities()
    issuer = _find_mentioned(sub_question, entities["issuers"], entities["issuer_forms"])

    candidates: list[AtomRetrievalInfo] = []
    candidates.extend(_bond_terms_candidates(sub_question, issuer))
    candidates.extend(_issue_overview_candidates(sub_question, issuer))
    candidates.extend(_participant_candidates(sub_question, entities["participants"]))
    return candidates
