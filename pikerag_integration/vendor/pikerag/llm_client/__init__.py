# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Trimmed from the original pikerag/llm_client/__init__.py: upstream eagerly
# imports AzureMetaLlamaClient/AzureOpenAIClient/HFMetaLlamaClient/
# StandardOpenAIClient too, which pull in openai/transformers/torch - none of
# which this project uses (see ../../../llm_clients/anthropic_client.py for
# the client actually used here). Only the base class is needed from this
# vendored copy.

from pikerag.llm_client.base import BaseLLMClient

__all__ = ["BaseLLMClient"]
