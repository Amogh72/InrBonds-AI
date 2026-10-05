"""
Claude (Anthropic Messages API) implementation of PIKE-RAG's BaseLLMClient
interface (vendor/pikerag/llm_client/base.py) - the piece upstream never
shipped, since every client it provides targets Azure OpenAI/OpenAI/HF Llama
endpoints.

BaseLLMClient's two abstract methods work in OpenAI chat-message shape
(messages: List[{"role": ..., "content": ...}]), since every vendored prompt
protocol (vendor/pikerag/prompts/...) builds messages in that shape via
MessageTemplate.format(). Anthropic's Messages API is close but not
identical: it takes `system` as a separate top-level string, not a message
with role="system", and requires `max_tokens` explicitly. This client
translates between the two at the boundary so none of the vendored prompt
code needs to know the difference.
"""

import os
from typing import Any, List

import anthropic

from pikerag.llm_client.base import BaseLLMClient
from pikerag.utils.logger import Logger

DEFAULT_MAX_TOKENS = 2048


class AnthropicClient(BaseLLMClient):
    """
    yml-style config (if ever driven by a config file rather than
    instantiated directly, matching the pattern every other PIKE-RAG client
    documents in its own docstring):
        llm_client:
            module_path: pikerag_integration.llm_clients.anthropic_client
            class_name: AnthropicClient
            args: { api_key: <your_api_key> }
            llm_config: { model: claude-sonnet-5, temperature: 0, max_tokens: 2048 }
    """

    NAME = "AnthropicClient"

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

        api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _split_system_and_messages(messages: List[dict]) -> tuple[str, List[dict]]:
        """
        PIKE-RAG's MessageTemplate always emits a leading ("system", ...)
        entry (see vendor/pikerag/prompts/decomposition/atom_based.py's
        templates) - Anthropic wants that pulled out as a separate `system`
        argument, not left in the `messages` list.
        """
        system_parts: List[str] = []
        chat_messages: List[dict] = []
        for message in messages:
            if message.get("role") == "system":
                system_parts.append(message.get("content", ""))
            else:
                chat_messages.append({"role": message["role"], "content": message["content"]})
        return "\n".join(part for part in system_parts if part), chat_messages

    def _get_response_with_messages(self, messages: List[dict], **llm_config) -> Any:
        system_prompt, chat_messages = self._split_system_and_messages(messages)

        request_kwargs = dict(llm_config)
        request_kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
        request_kwargs.pop("cache_config", None)  # BaseLLMClient's own concept, not an Anthropic API param

        response = None
        num_attempt = 0
        while num_attempt < self._max_attempt:
            try:
                response = self._client.messages.create(
                    system=system_prompt or anthropic.NOT_GIVEN,
                    messages=chat_messages,
                    **request_kwargs,
                )
                break
            except anthropic.RateLimitError as exc:
                self.warning(f"  Failed due to RateLimitError: {exc}")
                num_attempt += 1
                self._wait(num_attempt)
                self.warning("  Retrying...")
            except anthropic.BadRequestError as exc:
                self.warning(f"  Failed due to Exception: {exc}")
                self.warning("  Skip this request...")
                break
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
            text_blocks = [block.text for block in response.content if getattr(block, "type", None) == "text"]
            content = "".join(text_blocks)
        except Exception as exc:  # noqa: BLE001
            self.warning(f"Try to get content from response but get exception:\n  {exc}")
            self.debug(f"  Response: {response}\n  Last message: {messages}")
            content = ""

        if not content:
            self.warning(f"Non-content response, stop_reason={getattr(response, 'stop_reason', None)}")

        return content

    def close(self):
        super().close()
        self._client.close()
