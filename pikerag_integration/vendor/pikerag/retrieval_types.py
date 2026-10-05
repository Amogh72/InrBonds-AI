# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# AtomRetrievalInfo extracted from pikerag/knowledge_retrievers/chunk_atom_retriever.py
# (verbatim field set) - that module's ChunkAtomRetriever class pulls in chromadb/
# langchain to run its own two-vector-store retrieval, which this project doesn't use
# (see ../../retrieval/mcp_retrieval.py: we retrieve through our existing MCP tools
# instead). Only the dataclass itself - the shape every decomposition prompt protocol
# in prompts/decomposition/atom_based.py is actually written against - is needed here.

from dataclasses import dataclass
from typing import List


@dataclass
class AtomRetrievalInfo:
    atom_query: str
    atom: str
    source_chunk_title: str
    source_chunk: str
    source_chunk_id: str
    retrieval_score: float
    atom_embedding: List[float]
