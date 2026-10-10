"""
Minimal command-line entry point: ask one question, get back a decomposed,
cited answer.

Usage:
    python cli.py "Compare PFC's and IIFL's debenture trustee"
    python cli.py "What is PFC's Series III coupon?" --max-sub-questions 3

Requires either ANTHROPIC_API_KEY or GEMINI_API_KEY (in
pikerag_integration/.env, or the environment - see llm_clients/gemini_client.py
for why Gemini is offered as a free-tier-friendly alternative to Anthropic's
paid-only API) and the MCP server's databases (Qdrant specifically, for
vector_search) to be reachable - same mcp_server/.env this project's other
components use.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=ROOT / ".env")

from orchestrator import DecompositionOrchestrator, DEFAULT_MAX_SUB_QUESTIONS  # noqa: E402

PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-5",
    "gemini": "gemini-3.8-flash",
}


def _build_client(provider: str):
    # Imported lazily per-provider so picking one doesn't require both
    # providers' SDKs (and their API keys) to be installed/set.
    if provider == "anthropic":
        from llm_clients.anthropic_client import AnthropicClient
        return AnthropicClient()
    elif provider == "gemini":
        from llm_clients.gemini_client import GeminiClient
        return GeminiClient()
    raise ValueError(f"Unknown --provider '{provider}', expected 'anthropic' or 'gemini'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the bond document knowledge base via PIKE-RAG-style decomposition.")
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--provider", choices=["anthropic", "gemini"], default=os.getenv("LLM_PROVIDER", "anthropic"),
                         help="Which LLM backs the decomposition loop. Gemini has a free API tier; Anthropic's does not.")
    parser.add_argument("--model", default=None, help="Defaults to a sensible model for --provider if omitted.")
    parser.add_argument("--max-sub-questions", type=int, default=DEFAULT_MAX_SUB_QUESTIONS)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--show-trace", action="store_true", help="Print the full decomposition trace, not just the final answer.")
    args = parser.parse_args()

    model = args.model or os.getenv("ANTHROPIC_MODEL" if args.provider == "anthropic" else "GEMINI_MODEL") \
        or PROVIDER_DEFAULTS[args.provider]

    client = _build_client(args.provider)
    orchestrator = DecompositionOrchestrator(
        llm_client=client,
        llm_config={"model": model, "max_tokens": args.max_tokens, "temperature": 0},
        max_sub_questions=args.max_sub_questions,
    )

    result = orchestrator.answer(args.question)

    print(f"\nQuestion: {args.question}\n")
    print(f"Answer: {result.get('answer', '(no answer)')}\n")
    if result.get("rationale"):
        print(f"Rationale: {result['rationale']}\n")

    citations = result.get("citations", [])
    if citations:
        print("Citations:")
        for citation in citations:
            print(f"  - {citation['source']}  (chunk_id={citation['chunk_id']})")
    else:
        print("Citations: none (answered without retrieved context)")

    if args.show_trace:
        print("\n--- Decomposition trace ---")
        print(json.dumps(result.get("decomposition_trace", {}), indent=2))

    client.close()


if __name__ == "__main__":
    main()
