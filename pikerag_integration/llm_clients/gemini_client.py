"""
Gemini (Google Gen AI SDK) implementation of PIKE-RAG's BaseLLMClient
interface (vendor/pikerag/llm_client/base.py) - a free-tier-friendly
alternative to AnthropicClient for local testing, since Gemini's Flash
models have a genuinely free API tier (Anthropic's does not).

Translation notes, mirroring anthropic_client.py's approach:
- BaseLLMClient's messages are OpenAI chat-message shape
  (role: system/user/assistant). Gemini's SDK wants the system message
  pulled out as a separate `system_instruction` config field, and uses
  role "model" instead of "assistant" for prior turns.
- llm_config's "max_tokens" (the name every vendored prompt protocol's
  caller uses, matching AnthropicClient/OpenAI convention) is translated
  to Gemini's "max_output_tokens".
"""

import os
import re
from typing import Any, List

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger

DEFAULT_MAX_OUTPUT_TOKENS = 2048

# Unlike Claude (which tends to emit bare JSON when a prompt asks for it),
# Gemini routinely wraps JSON responses in a ```json ... ``` fence even when
# not asked to. vendor/pikerag/utils/json_parser.py's parse_json() finds
# JSON by rfind("{")/rfind("}") so a fence around otherwise-valid JSON
# shouldn't break it on its own, but a fence paired with any other hiccup
# (an empty/truncated response, say) turns into a confusing stray-brace
# failure instead of a clean "no JSON found" one. Stripping the fence here,
# before content ever reaches the vendored parser, removes that confound.
_MARKDOWN_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


class GeminiDailyQuotaExhausted(RuntimeError):
    """
    Raised instead of silently returning empty content when Gemini's
    free-tier DAILY (not per-minute) quota is exhausted. BaseLLMClient's
    generate_content_with_messages() (vendor/pikerag/llm_client/base.py)
    treats a None response as "" content with just a warning, not an
    exception - which downstream turns into a confusing JSON-parse failure
    ("Expecting value: line 1 column 1") with no indication of the real
    cause. Raising here instead gives orchestrator.py/query_service's
    caller a clear, specific error to show instead.
    """


def _is_daily_quota_exhaustion(exc: ClientError) -> bool:
    """
    True if this 429's quota violation is a per-DAY limit (quotaId like
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier", seen live) rather
    than a short-lived per-minute one - the only distinction that matters
    for whether retrying is worth attempting at all, since a per-day quota
    cannot recover within this client's retry window (minutes, not hours).
    A substring check rather than precise structure navigation, since the
    exact error JSON shape isn't documented and this only needs to be
    right, not strict.
    """
    return "PerDay" in str(exc)


class GeminiClient(BaseLLMClient):
    """
    yml-style config (matching the pattern every other PIKE-RAG client
    documents in its own docstring):
        llm_client:
            module_path: pikerag_integration.llm_clients.gemini_client
            class_name: GeminiClient
            args: { api_key: <your_api_key> }
            llm_config: { model: gemini-3.8-flash, temperature: 0, max_tokens: 2048 }
    """

    NAME = "GeminiClient"

    def __init__(
        self,
        location: str = None,
        auto_dump: bool = True,
        logger: Logger = None,
        max_attempt: int = 5,
        exponential_backoff_factor: int = None,
        unit_wait_time: int = 60,
        **kwargs: Any,
    ) -> None:
        super().__init__(location, auto_dump, logger, max_attempt, exponential_backoff_factor, unit_wait_time, **kwargs)

        api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _split_system_and_messages(messages: List[dict]) -> tuple[str, List["types.Content"]]:
        """Same role-separation anthropic_client.py does, adapted to Gemini's
        Content/Part shape and its "model" (not "assistant") role name."""
        system_parts: List[str] = []
        contents: List[types.Content] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(message.get("content", ""))
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append(types.Content(role=gemini_role, parts=[types.Part(text=message["content"])]))
        return "\n".join(part for part in system_parts if part), contents

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> Any:
        system_instruction, contents = self._split_system_and_messages(messages)

        model = llm_config.get("model", "gemini-3.8-flash")
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=llm_config.get("temperature"),
            max_output_tokens=llm_config.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
        )

        response = None
        num_attempt = 0
        while num_attempt < self._max_attempt:
            try:
                response = self._client.models.generate_content(model=model, contents=contents, config=config)
                break
            except ClientError as exc:
                if getattr(exc, "code", None) == 429 and _is_daily_quota_exhaustion(exc):
                    # A per-day quota (seen live: quotaId
                    # "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    # Google's own error suggesting a retry in ~17 hours)
                    # cannot possibly recover within this loop's retry
                    # window (at most max_attempt * unit_wait_time, minutes
                    # not hours) - retrying it anyway just burns the
                    # question's full timeout silently, with nothing to
                    # show for it. Fail fast instead; a per-minute rate
                    # limit (below) is the only 429 actually worth retrying.
                    self.warning(f"  Daily quota exhausted, not retrying (would need hours, not minutes): {exc}")
                    raise GeminiDailyQuotaExhausted(
                        "Gemini free-tier daily quota exhausted for this model. It will not recover "
                        "within minutes - wait for the daily reset, pass --model with a different "
                        "Gemini model, or use --provider anthropic instead."
                    ) from exc
                elif getattr(exc, "code", None) == 429:  # short-lived rate limit - worth retrying
                    self.warning(f"  Failed due to RateLimitError: {exc}")
                    num_attempt += 1
                    self._wait(num_attempt)
                    self.warning("  Retrying...")
                else:
                    self.warning(f"  Failed due to Exception: {exc}")
                    self.warning("  Skip this request...")
                    break
            except ServerError as exc:
                self.warning(f"  Failed due to Exception: {exc}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  Retrying...")
            except Exception as exc:  # noqa: BLE001 - mirror BaseLLMClient's other clients' broad retry
                self.warning(f"  Failed due to Exception: {exc}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  Retrying...")

        return response

    def _get_content_from_response(self, response: Any, messages: List[dict] = None) -> str:
        if response is None:
            return ""

        try:
            content = response.text or ""
            fence_match = _MARKDOWN_FENCE_RE.match(content.strip())
            if fence_match:
                content = fence_match.group(1)
        except Exception as exc:  # noqa: BLE001
            self.warning(f"Try to get content from response but get exception:\n  {exc}")
            self.debug(f"  Response: {response}\n  Last message: {messages}")
            content = ""

        if not content:
            candidates = getattr(response, "candidates", None) or []
            finish_reason = candidates[0].finish_reason if candidates else None
            self.warning(f"Non-content response, finish_reason={finish_reason}")

        return content

    def close(self):
        super().close()
        self._client.close()
