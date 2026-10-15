# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Trimmed from the original pikerag/prompts/qa/__init__.py: upstream also
# imports multiple_choice.py, which this project doesn't use (our final
# answer step is free-text generation, not multiple choice) and which
# wasn't vendored.

from pikerag.prompts.qa.generation import (
    generation_qa_protocol, generation_qa_template, generation_qa_with_reference_protocol,
    generation_qa_with_reference_template, GenerationQaParser,
)

__all__ = [
    "generation_qa_protocol", "generation_qa_template",
    "generation_qa_with_reference_protocol", "generation_qa_with_reference_template",
    "GenerationQaParser",
]
