"""
Centralized configuration for the Bond Document Assistant MCP server.
All values are pulled from environment variables (see .env.example).
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class SQLConfig:
    host: str = os.getenv("SQL_HOST", "localhost")
    port: int = int(os.getenv("SQL_PORT", "3306"))
    database: str = os.getenv("SQL_DATABASE", "inrbonds")
    user: str = os.getenv("SQL_USER", "root")
    password: str = os.getenv("SQL_PASSWORD", "")
    # "mysql" or "postgresql" - swap driver without touching calling code
    dialect: str = os.getenv("SQL_DIALECT", "mysql")


@dataclass(frozen=True)
class QdrantConfig:
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection: str = os.getenv("QDRANT_COLLECTION", "bond_documents")
    # Must match the embedding model used at ingestion time
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")


@dataclass(frozen=True)
class ServerConfig:
    name: str = os.getenv("MCP_SERVER_NAME", "inrbonds-document-assistant")
    transport: str = os.getenv("MCP_TRANSPORT", "stdio")  # stdio | streamable-http
    http_host: str = os.getenv("MCP_HTTP_HOST", "0.0.0.0")
    http_port: int = int(os.getenv("MCP_HTTP_PORT", "8000"))
    max_rows_returned: int = int(os.getenv("MAX_ROWS_RETURNED", "200"))
    query_timeout_seconds: int = int(os.getenv("QUERY_TIMEOUT_SECONDS", "15"))


sql_config = SQLConfig()
qdrant_config = QdrantConfig()
neo4j_config = Neo4jConfig()
server_config = ServerConfig()
