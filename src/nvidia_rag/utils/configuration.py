# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple configuration for NVIDIA RAG."""

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator
from pydantic import Field as PydanticField
from pydantic.fields import FieldInfo


def Field(default=None, *, env: str = None, description: str = None, **kwargs):
    """Pydantic Field with optional environment variable support.

    Args:
        default: Default value
        env: Environment variable name (optional)
        description: Description of what the field is for (optional)
        **kwargs: Other Pydantic Field parameters

    Example:
        name: str = Field(default="milvus", env="APP_VECTORSTORE_NAME", description="Vector store name")
    """
    if env:
        if "json_schema_extra" not in kwargs:
            kwargs["json_schema_extra"] = {}
        kwargs["json_schema_extra"]["env"] = env

    if description:
        kwargs["description"] = description

    return PydanticField(default=default, **kwargs)


class _ConfigBase(BaseModel):
    """Base configuration class with automatic environment variable loading.

    Usage:
        class MyConfig(_ConfigBase):
            server_url: str = Field(default="", env="MY_SERVER_URL")

    Priority: dict/yaml values > env vars > defaults
    """

    def __init__(self, **data):
        # Load values from environment variables
        env_values = {}
        for field_name, field_info in self.model_fields.items():
            # Check if Field has 'env' in json_schema_extra
            if isinstance(field_info, FieldInfo) and field_info.json_schema_extra:
                env_var_name = field_info.json_schema_extra.get("env")
                if env_var_name and env_var_name in os.environ:
                    raw_value = os.environ[env_var_name]
                    # Strip surrounding quotes if present (handles Docker Compose quoted values)
                    if isinstance(raw_value, str) and len(raw_value) >= 2:
                        # More robust quote stripping: strip whitespace first, then quotes
                        raw_value = raw_value.strip()
                        if (raw_value.startswith('"') and raw_value.endswith('"')) or (
                            raw_value.startswith("'") and raw_value.endswith("'")
                        ):
                            raw_value = raw_value[1:-1]
                    env_values[field_name] = raw_value

        # Merge: data overrides env vars, env vars override defaults
        merged_data = {**env_values, **data}

        super().__init__(**merged_data)

    def get_api_key(self) -> str | None:
        """Get API key with fallback to global NVIDIA_API_KEY or NGC_API_KEY.

        Returns:
            API key string if found, None otherwise.
        """
        if hasattr(self, "api_key") and self.api_key:
            api_key_value = self.api_key.get_secret_value()
            if api_key_value:
                return api_key_value

        return os.environ.get("NVIDIA_API_KEY") or os.environ.get("NGC_API_KEY")


class SearchType(StrEnum):
    """Allowed search types for vector store queries."""

    DENSE = "dense"
    HYBRID = "hybrid"


class RankerType(StrEnum):
    """Allowed ranker types for vector store in case of Hybrid Search"""

    RRF = "rrf"
    WEIGHTED = "weighted"


class VectorStoreConfig(_ConfigBase):
    """Vector Store configuration.

    Environment variables:
        APP_VECTORSTORE_NAME, APP_VECTORSTORE_URL, APP_VECTORSTORE_INDEXTYPE,
        APP_VECTORSTORE_SEARCHTYPE, COLLECTION_NAME, etc.
    """

    name: str = Field(
        default="milvus",
        env="APP_VECTORSTORE_NAME",
        description="Name of the vector store backend (e.g., milvus, elasticsearch)",
    )
    url: str = Field(
        default="http://localhost:19530",
        env="APP_VECTORSTORE_URL",
        description="URL endpoint for the vector store service",
    )

    @field_validator("name", "url", mode="before")
    @classmethod
    def normalize_string(cls, v: Any) -> Any:
        """Normalize string fields by stripping whitespace and quotes."""
        if isinstance(v, str):
            return v.strip().strip('"').strip("'")
        return v

    nlist: int = Field(
        default=64,
        env="APP_VECTORSTORE_NLIST",
        description="Number of clusters for IVF index",
    )
    nprobe: int = Field(
        default=16,
        env="APP_VECTORSTORE_NPROBE",
        description="Number of clusters to search during query",
    )
    index_type: str = Field(
        default="GPU_CAGRA",
        env="APP_VECTORSTORE_INDEXTYPE",
        description="Type of vector index (e.g., GPU_CAGRA, IVF_FLAT)",
    )
    enable_gpu_index: bool = Field(
        default=True,
        env="APP_VECTORSTORE_ENABLEGPUINDEX",
        description="Enable GPU acceleration for index building",
    )
    enable_gpu_search: bool = Field(
        default=True,
        env="APP_VECTORSTORE_ENABLEGPUSEARCH",
        description="Enable GPU acceleration for search operations",
    )
    search_type: SearchType = Field(
        default=SearchType.DENSE,
        env="APP_VECTORSTORE_SEARCHTYPE",
        description="Type of search to perform (dense, hybrid)",
    )
    ranker_type: RankerType = Field(
        default=RankerType.RRF,
        env="APP_VECTORSTORE_RANKER_TYPE",
        description="Type of ranker to use ('rrf', 'weighted')",
    )
    dense_weight: float = Field(
        default=0.5,
        env="APP_VECTORSTORE_DENSE_WEIGHT",
        description="Weight for dense vector search in case of weighted Hybrid Search",
    )
    sparse_weight: float = Field(
        default=0.5,
        env="APP_VECTORSTORE_SPARSE_WEIGHT",
        description="Weight for sparse vector search in case of weighted Hybrid Search",
    )
    default_collection_name: str = Field(
        default="multimodal_data",
        env="COLLECTION_NAME",
        description="Default collection/index name for storing vectors",
    )
    ef: int = Field(
        default=100,
        env="APP_VECTORSTORE_EF",
        description="Size of the dynamic candidate list for HNSW search",
    )
    username: str = Field(
        default="",
        env="APP_VECTORSTORE_USERNAME",
        description="Username for vector store authentication",
    )
    password: SecretStr | None = Field(
        default=None,
        env="APP_VECTORSTORE_PASSWORD",
        description="Password for vector store authentication",
    )

    # API key authentication for vector store (used by Elasticsearch)
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_VECTORSTORE_APIKEY",
        description="API key for vector store authentication (base64 form 'id:secret')",
    )
    api_key_id: str = Field(
        default="",
        env="APP_VECTORSTORE_APIKEY_ID",
        description="API key ID for vector store authentication",
    )
    api_key_secret: SecretStr | None = Field(
        default=None,
        env="APP_VECTORSTORE_APIKEY_SECRET",
        description="API key secret for vector store authentication",
    )

    # SSL/TLS configuration for Elasticsearch
    ssl_enabled: bool = Field(
        default=False,
        env="APP_VECTORSTORE_SSL_ENABLED",
        description="Enable SSL/TLS for Elasticsearch connections",
    )
    ca_certs: str | None = Field(
        default=None,
        env="APP_VECTORSTORE_CA_CERTS",
        description=(
            "Path to a CA bundle PEM file for verifying the Elasticsearch server certificate. "
            "Required when ssl_enabled=True and the server uses a self-signed or private CA cert "
            "(e.g., the ECK-generated CA mounted from the *-es-http-certs-public secret). "
            "Leave unset to use the system CA bundle (suitable for publicly-trusted certs)."
        ),
    )
    verify_certs: bool = Field(
        default=True,
        env="APP_VECTORSTORE_VERIFY_CERTS",
        description=(
            "Verify the Elasticsearch server certificate when ssl_enabled=True. "
            "Set to False only for development/testing; not recommended in production."
        ),
    )


class NvIngestConfig(_ConfigBase):
    """NV-Ingest configuration."""

    message_client_hostname: str = Field(
        default="localhost",
        env="APP_NVINGEST_MESSAGECLIENTHOSTNAME",
        description="Hostname for NV-Ingest message client",
    )
    message_client_port: int = Field(
        default=7670,
        env="APP_NVINGEST_MESSAGECLIENTPORT",
        description="Port for NV-Ingest message client",
    )

    @field_validator("message_client_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    extract_text: bool = Field(
        default=True,
        env="APP_NVINGEST_EXTRACTTEXT",
        description="Enable text extraction from documents",
    )
    extract_infographics: bool = Field(
        default=False,
        env="APP_NVINGEST_EXTRACTINFOGRAPHICS",
        description="Enable infographic extraction from documents",
    )
    extract_tables: bool = Field(
        default=True,
        env="APP_NVINGEST_EXTRACTTABLES",
        description="Enable table extraction from documents",
    )
    extract_charts: bool = Field(
        default=True,
        env="APP_NVINGEST_EXTRACTCHARTS",
        description="Enable chart extraction from documents",
    )
    extract_images: bool = Field(
        default=False,
        env="APP_NVINGEST_EXTRACTIMAGES",
        description="Enable image extraction from documents",
    )
    extract_page_as_image: bool = Field(
        default=False,
        env="APP_NVINGEST_EXTRACTPAGEASIMAGE",
        description="Extract entire pages as images",
    )
    structured_elements_modality: str = Field(
        default="",
        env="STRUCTURED_ELEMENTS_MODALITY",
        description="Modality for processing structured elements (tables, charts)",
    )
    image_elements_modality: str = Field(
        default="",
        env="IMAGE_ELEMENTS_MODALITY",
        description="Modality for processing image elements",
    )
    pdf_extract_method: str | None = Field(
        default=None,
        env="APP_NVINGEST_PDFEXTRACTMETHOD",
        description="Method to use for PDF extraction. Options: None, pdfium, pdfium_hybrid, ocr, nemotron_parse. "
        "pdfium_hybrid and ocr target scanned/image-based PDFs (added in nv-ingest 26.3.0).",
    )

    @field_validator("pdf_extract_method", mode="before")
    @classmethod
    def normalize_pdf_extract_method(cls, v: Any) -> Any:
        """Normalize string 'None'/'none' to Python None."""
        if isinstance(v, str) and v.lower() in ("none", "null", ""):
            return None
        return v

    text_depth: str = Field(
        default="page",
        env="APP_NVINGEST_TEXTDEPTH",
        description="Granularity level for text extraction (page, document)",
    )
    tokenizer: str = Field(
        default="intfloat/e5-large-unsupervised",
        env="APP_NVINGEST_TOKENIZER",
        description="Tokenizer model for text chunking",
    )
    chunk_size: int = Field(
        default=2048,
        env="APP_NVINGEST_CHUNKSIZE",
        description="Maximum size of text chunks in tokens (used by direct-ingest/embed_store path)",
    )
    nv_ingest_split_chunk_size: int = Field(
        default=512,
        env="APP_NVINGEST_SPLIT_CHUNKSIZE",
        description="Chunk size in tokens for the NV-Ingest split task (binary files via nv-ingest pipeline)",
    )
    chunk_overlap: int = Field(
        default=150,
        env="APP_NVINGEST_CHUNKOVERLAP",
        description="Number of overlapping tokens between chunks",
    )
    job_timeout: int = Field(
        default=600,
        env="APP_NVINGEST_JOB_TIMEOUT",
        description=(
            "Maximum seconds to wait for a single nv-ingest batch job to complete. "
            "If the job does not finish within this window (e.g. due to a corrupt PDF "
            "causing a PDFium hang in the Ray pipeline), the batch is cancelled and "
            "treated as a failure so the crawl/ingest task can continue."
        ),
    )
    max_concurrent_jobs: int = Field(
        default=2,
        env="APP_NVINGEST_MAX_CONCURRENT_JOBS",
        description=(
            "Maximum number of nv-ingest jobs that may be in-flight simultaneously "
            "across all concurrent crawler batches and ingest calls. An asyncio.Semaphore "
            "enforces this limit so that the Ray worker queue never accumulates more jobs "
            "than workers can drain, preventing timeout cascades on large PDFs."
        ),
    )
    enable_direct_ingest: bool = Field(
        default=False,
        env="APP_NVINGEST_ENABLE_DIRECT_INGEST",
        description=(
            "When True, nemoretriever-parse output chunks bypass nv-ingest entirely: "
            "text is embedded directly via the embedding service and written to ES "
            "using the async bulk API.  Eliminates Ray queue pressure and timeout "
            "risk for text-only content (HTML pages, pre-chunked PDF text)."
        ),
    )
    caption_model_name: str = Field(
        default="nvidia/nemotron-nano-12b-v2-vl",
        env="APP_NVINGEST_CAPTIONMODELNAME",
        description="Model name for generating image captions",
    )
    caption_endpoint_url: str = Field(
        default="https://integrate.api.nvidia.com/v1/chat/completions",
        env="APP_NVINGEST_CAPTIONENDPOINTURL",
        description="API endpoint for caption generation service",
    )

    @field_validator("caption_endpoint_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "NvIngestConfig":
        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self

    enable_pdf_splitter: bool = Field(
        default=True,
        env="APP_NVINGEST_ENABLEPDFSPLITTER",
        description="Enable PDF page splitting during ingestion",
    )
    segment_audio: bool = Field(
        default=False,
        env="APP_NVINGEST_SEGMENTAUDIO",
        description="Enable audio segmentation during ingestion",
    )
    save_to_disk: bool = Field(
        default=False,
        env="APP_NVINGEST_SAVETODISK",
        description="Save extracted content to disk for debugging",
    )
    # Batch processing configuration
    enable_batch_mode: bool = Field(
        default=True,
        env="ENABLE_NV_INGEST_BATCH_MODE",
        description="Process files in batches for better throughput",
    )
    files_per_batch: int = Field(
        default=16,
        env="NV_INGEST_FILES_PER_BATCH",
        description="Number of files to process in each batch",
    )
    enable_parallel_batch_mode: bool = Field(
        default=True,
        env="ENABLE_NV_INGEST_PARALLEL_BATCH_MODE",
        description="Enable parallel processing of multiple batches",
    )
    concurrent_batches: int = Field(
        default=4,
        env="NV_INGEST_CONCURRENT_BATCHES",
        description="Number of batches to process concurrently",
    )
    enable_dynamic_batching: bool = Field(
        default=False,
        env="ENABLE_NV_INGEST_DYNAMIC_BATCHING",
        description="Enable dynamic calculation of batch parameters based on file characteristics",
    )
    enable_pdf_split_processing: bool = Field(
        default=False,
        env="APP_NVINGEST_ENABLE_PDF_SPLIT_PROCESSING",
        description="Enable PDF split processing during ingestion",
    )
    pages_per_chunk: int = Field(
        default=16,
        env="APP_NVINGEST_PAGES_PER_CHUNK",
        description="Number of pages per chunk for PDF split processing",
    )


class NemoParseConfig(_ConfigBase):
    """nemoretriever-parse VLM configuration for complex document element routing."""

    enabled: bool = Field(
        default=False,
        env="APP_NEMOPARSE_ENABLED",
        description="Enable nemoretriever-parse routing for documents with complex data elements",
    )
    endpoint_url: str = Field(
        default="",
        env="APP_NEMOPARSE_SERVERURL",
        description="nemoretriever-parse inference endpoint URL (e.g. http://nemoretriever-parse-ms:8000/v1/chat/completions)",
    )
    model_name: str = Field(
        default="nvdev/nvidia/nemoretriever-parse",
        env="APP_NEMOPARSE_MODELNAME",
        description="Model identifier sent in the nemoretriever-parse API request",
    )
    api_key: str = Field(
        default="",
        env="APP_NEMOPARSE_APIKEY",
        description="Bearer token for the nemoretriever-parse endpoint (empty = no auth header)",
    )
    force_all: bool = Field(
        default=True,
        env="APP_NEMOPARSE_FORCE_ALL",
        description=(
            "When True, route ALL PDFs through nemoretriever-parse unconditionally "
            "(skips the two-pass complex-element pre-scan). Requires enabled=True."
        ),
    )
    figure_describe_endpoint: str = Field(
        default="",
        env="APP_NEMOPARSE_FIGURE_DESCRIBE_ENDPOINT",
        description=(
            "VLM endpoint for generating descriptions of Picture regions detected by "
            "nemoretriever-parse (e.g. http://nim-vlm:8000/v1/chat/completions). "
            "Empty string disables figure description."
        ),
    )
    figure_describe_model: str = Field(
        default="nvidia/nemotron-nano-12b-v2-vl",
        env="APP_NEMOPARSE_FIGURE_DESCRIBE_MODEL",
        description="Model name forwarded to the figure description VLM endpoint.",
    )
    max_parallel_docs: int = Field(
        default=4,
        env="APP_NEMOPARSE_MAX_PARALLEL_DOCS",
        description=(
            "Number of PDF documents to process in parallel through nemoretriever-parse. "
            "Each document uses max_parallel_pages workers internally. "
            "Set to match nemotron-parse replica count x max-num-seqs / max_parallel_pages."
        ),
    )
    page_batch_size: int = Field(
        default=32,
        env="APP_NEMOPARSE_PAGE_BATCH_SIZE",
        description=(
            "Number of PDF pages rasterised into memory at once per document. "
            "Smaller values reduce peak RAM at the cost of more convert_from_path calls. "
            "At 300 DPI JPEG each page is ~25 MB; batch_size=32 → ~800 MB peak per doc."
        ),
    )


class ModelParametersConfig(_ConfigBase):
    """Model parameters configuration."""

    max_tokens: int = Field(
        default=32768,
        env="LLM_MAX_TOKENS",
        description="Maximum number of tokens to generate in response",
    )
    min_tokens: int = Field(
        default=0,
        env="LLM_MIN_TOKENS",
        description="Minimum number of tokens to generate in response",
    )
    max_thinking_tokens: int = Field(
        default=0,
        env="LLM_MAX_THINKING_TOKENS",
        description="Maximum thinking tokens to allocate for reasoning models (0 = disabled by default)",
    )
    min_thinking_tokens: int = Field(
        default=0,
        env="LLM_MIN_THINKING_TOKENS",
        description="Minimum thinking tokens to allocate for reasoning models (0 = disabled by default)",
    )
    ignore_eos: bool = Field(
        default=False,
        env="LLM_IGNORE_EOS",
        description="Ignore end-of-sequence token during generation",
    )
    temperature: float = Field(
        default=0.0,
        env="LLM_TEMPERATURE",
        description="Sampling temperature for controlling randomness (0.0 = deterministic)",
    )
    top_p: float = Field(
        default=1.0,
        env="LLM_TOP_P",
        description="Nucleus sampling threshold for token selection",
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Temperature must be non-negative")
        return v

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v


class LLMConfig(_ConfigBase):
    """LLM configuration."""

    server_url: str = Field(
        default="",
        env="APP_LLM_SERVERURL",
        description="URL endpoint for the LLM inference service",
    )
    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        env="APP_LLM_MODELNAME",
        description="Name of the language model to use for generation",
    )
    model_engine: str = Field(
        default="nvidia-ai-endpoints",
        env="APP_LLM_MODELENGINE",
        description="Engine/provider for LLM inference (e.g., nvidia-ai-endpoints, openai)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_LLM_APIKEY",
        description="API key for LLM service (overrides global NVIDIA_API_KEY)",
    )
    parameters: ModelParametersConfig = PydanticField(
        default_factory=ModelParametersConfig, description="Model generation parameters"
    )

    @field_validator("server_url", "model_name", "model_engine", mode="before")
    @classmethod
    def normalize_string(cls, v: Any) -> Any:
        """Normalize string fields by stripping whitespace and quotes."""
        if isinstance(v, str):
            return v.strip().strip('"').strip("'")
        return v

    @field_validator("server_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL has a scheme."""
        if v and not v.startswith(("http://", "https://")):
            return f"http://{v}"
        return v

    def get_model_parameters(self) -> dict:
        """Return model parameters as dict."""
        return {
            "min_tokens": self.parameters.min_tokens,
            "ignore_eos": self.parameters.ignore_eos,
            "max_tokens": self.parameters.max_tokens,
            "min_thinking_tokens": self.parameters.min_thinking_tokens,
            "max_thinking_tokens": self.parameters.max_thinking_tokens,
            "temperature": self.parameters.temperature,
            "top_p": self.parameters.top_p,
        }


class QueryRewriterConfig(_ConfigBase):
    """Query Rewriter configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        env="APP_QUERYREWRITER_MODELNAME",
        description="Model for rewriting user queries to improve retrieval",
    )
    server_url: str = Field(
        default="",
        env="APP_QUERYREWRITER_SERVERURL",
        description="URL endpoint for query rewriter service",
    )
    enable_query_rewriter: bool = Field(
        default=False,
        env="ENABLE_QUERYREWRITER",
        description="Enable automatic query rewriting before retrieval",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_QUERYREWRITER_APIKEY",
        description="API key for query rewriter (overrides global NVIDIA_API_KEY)",
    )
    multiturn_retrieval_simple: bool = Field(
        default=False,
        env="MULTITURN_RETRIEVER_SIMPLE",
        description="Enable concatenating conversation history with current query for retrieval (used when query rewriter is disabled)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class FilterExpressionGeneratorConfig(_ConfigBase):
    """Filter Expression Generator configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        env="APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
        description="Model for generating metadata filter expressions from queries",
    )
    server_url: str = Field(
        default="",
        env="APP_FILTEREXPRESSIONGENERATOR_SERVERURL",
        description="URL endpoint for filter expression generator service",
    )
    enable_filter_generator: bool = Field(
        default=False,
        env="ENABLE_FILTER_GENERATOR",
        description="Enable automatic filter expression generation from natural language",
    )
    temperature: float = Field(
        default=0.0,
        env="APP_FILTEREXPRESSIONGENERATOR_TEMPERATURE",
        description="Sampling temperature for filter generation",
    )
    top_p: float = Field(
        default=1.0,
        env="APP_FILTEREXPRESSIONGENERATOR_TOPP",
        description="Nucleus sampling threshold for filter generation",
    )
    max_tokens: int = Field(
        default=32768,
        env="APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS",
        description="Maximum tokens for filter expression generation",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_FILTEREXPRESSIONGENERATOR_APIKEY",
        description="API key for filter generator (overrides global NVIDIA_API_KEY)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class TextSplitterConfig(_ConfigBase):
    """Text Splitter configuration."""

    model_name: str = Field(
        default="Snowflake/snowflake-arctic-embed-l",
        env="APP_TEXTSPLITTER_MODELNAME",
        description="Tokenizer model for text splitting",
    )
    chunk_size: int = Field(
        default=510,
        env="APP_TEXTSPLITTER_CHUNKSIZE",
        description="Target size for text chunks in tokens",
    )
    chunk_overlap: int = Field(
        default=200,
        env="APP_TEXTSPLITTER_CHUNKOVERLAP",
        description="Number of overlapping tokens between consecutive chunks",
    )

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "TextSplitterConfig":
        if self.chunk_overlap > self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(_ConfigBase):
    """Embedding configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.2-nv-embedqa-1b-v2",
        env="APP_EMBEDDINGS_MODELNAME",
        description="Model for generating text embeddings",
    )
    model_engine: str = Field(
        default="nvidia-ai-endpoints",
        env="APP_EMBEDDINGS_MODELENGINE",
        description="Engine/provider for embedding generation",
    )
    dimensions: int = Field(
        default=2048,
        env="APP_EMBEDDINGS_DIMENSIONS",
        description="Dimensionality of the embedding vectors",
    )
    server_url: str = Field(
        default="",
        env="APP_EMBEDDINGS_SERVERURL",
        description="URL endpoint for embedding service",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_EMBEDDINGS_APIKEY",
        description="API key for embedding service (overrides global NVIDIA_API_KEY)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class RankingConfig(_ConfigBase):
    """Ranking configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        env="APP_RANKING_MODELNAME",
        description="Model for reranking retrieved documents",
    )
    model_engine: str = Field(
        default="nvidia-ai-endpoints",
        env="APP_RANKING_MODELENGINE",
        description="Engine/provider for reranking service",
    )
    server_url: str = Field(
        default="",
        env="APP_RANKING_SERVERURL",
        description="URL endpoint for reranking service",
    )
    enable_reranker: bool = Field(
        default=True,
        env="ENABLE_RERANKER",
        description="Enable reranking of retrieved documents before generation",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_RANKING_APIKEY",
        description="API key for ranking service (overrides global NVIDIA_API_KEY)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class RetrieverConfig(_ConfigBase):
    """Retriever configuration."""

    top_k: int = Field(
        default=10,
        env="APP_RETRIEVER_TOPK",
        description="Number of top documents to return after retrieval and reranking",
    )
    vdb_top_k: int = Field(
        default=100,
        env="VECTOR_DB_TOPK",
        description="Number of documents to retrieve from vector database before reranking",
    )
    score_threshold: float = Field(
        default=0.25,
        env="APP_RETRIEVER_SCORETHRESHOLD",
        description="Minimum similarity score threshold for retrieved documents",
    )
    nr_url: str = Field(
        default="http://retrieval-ms:8000",
        env="APP_RETRIEVER_NRURL",
        description="URL for NVIDIA Retrieval microservice",
    )
    nr_pipeline: str = Field(
        default="ranked_hybrid",
        env="APP_RETRIEVER_NRPIPELINE",
        description="Retrieval pipeline to use (e.g., ranked_hybrid, dense, sparse)",
    )

    @field_validator("nr_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v

    @field_validator("vdb_top_k")
    @classmethod
    def validate_vdb_top_k(cls, v: int) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"vdb_top_k must be an integer, got {type(v).__name__}")
        if v <= 0:
            raise ValueError(
                f"vdb_top_k must be greater than 0, got {v}. "
                "Please provide a positive integer for the number of documents to retrieve from the vector database."
            )
        return v

    @model_validator(mode="after")
    def validate_reranker_top_k(self) -> "RetrieverConfig":
        if self.vdb_top_k is not None and self.top_k > self.vdb_top_k:
            raise ValueError(
                f"reranker_top_k ({self.top_k}) must be less than or equal to vdb_top_k ({self.vdb_top_k}). "
                "Please check your settings and try again."
            )
        return self


class TracingConfig(_ConfigBase):
    """Tracing configuration."""

    enabled: bool = Field(
        default=False,
        env="APP_TRACING_ENABLED",
        description="Enable distributed tracing and metrics collection",
    )
    otlp_http_endpoint: str = Field(
        default="",
        env="APP_TRACING_OTLPHTTPENDPOINT",
        description="OpenTelemetry HTTP endpoint for traces",
    )
    otlp_grpc_endpoint: str = Field(
        default="",
        env="APP_TRACING_OTLPGRPCENDPOINT",
        description="OpenTelemetry gRPC endpoint for traces",
    )
    prometheus_multiproc_dir: str = Field(
        default="/tmp/prom_data",
        env="PROMETHEUS_MULTIPROC_DIR",
        description="Directory for Prometheus multiprocess metrics",
    )

    @field_validator("otlp_http_endpoint", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class VLMConfig(_ConfigBase):
    """VLM configuration."""

    server_url: str = Field(
        default="http://localhost:8000/v1",
        env="APP_VLM_SERVERURL",
        description="URL endpoint for Vision-Language Model service",
    )
    model_name: str = Field(
        default="nvidia/nemotron-nano-12b-v2-vl",
        env="APP_VLM_MODELNAME",
        description="Vision-Language Model for processing images and text",
    )
    temperature: float = Field(
        default=0.7,
        env="APP_VLM_TEMPERATURE",
        description="Sampling temperature for VLM generation",
    )
    top_p: float = Field(
        default=1.0,
        env="APP_VLM_TOP_P",
        description="Top-p sampling mass for VLM generation",
    )
    max_tokens: int = Field(
        default=4096,
        env="APP_VLM_MAX_TOKENS",
        description="Maximum number of tokens to generate in any given VLM call",
    )
    max_total_images: int = Field(
        default=5,
        env="APP_VLM_MAX_TOTAL_IMAGES",
        description="Maximum total images sent to VLM per request (query + context)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="APP_VLM_APIKEY",
        description="API key for VLM service (overrides global NVIDIA_API_KEY)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class MinioConfig(_ConfigBase):
    """Minio configuration."""

    endpoint: str = Field(
        default="localhost:9010",
        env="MINIO_ENDPOINT",
        description="MinIO object storage endpoint",
    )
    access_key: SecretStr = Field(
        default=SecretStr("minioadmin"),
        env="MINIO_ACCESSKEY",
        description="MinIO access key for authentication",
    )
    secret_key: SecretStr = Field(
        default=SecretStr("minioadmin"),
        env="MINIO_SECRETKEY",
        description="MinIO secret key for authentication",
    )


class SummarizerConfig(_ConfigBase):
    """Summarizer configuration."""

    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        env="SUMMARY_LLM",
        description="Model for generating document summaries",
    )
    server_url: str = Field(
        default="",
        env="SUMMARY_LLM_SERVERURL",
        description="URL endpoint for summarization service",
    )
    max_chunk_length: int = Field(
        default=9000,
        env="SUMMARY_LLM_MAX_CHUNK_LENGTH",
        description="Maximum chunk size in tokens for the summarizer model",
    )
    chunk_overlap: int = Field(
        default=400,
        env="SUMMARY_CHUNK_OVERLAP",
        description="Overlap between chunks for iterative summarization (in tokens)",
    )
    temperature: float = Field(
        default=0.0,
        env="SUMMARY_LLM_TEMPERATURE",
        description="Sampling temperature for summary generation",
    )
    top_p: float = Field(
        default=1.0,
        env="SUMMARY_LLM_TOP_P",
        description="Nucleus sampling threshold for summary generation",
    )
    max_parallelization: int = Field(
        default=20,
        env="SUMMARY_MAX_PARALLELIZATION",
        description="Maximum concurrent summaries across entire system (coordinated via Redis)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="SUMMARY_LLM_APIKEY",
        description="API key for summarization service (overrides global NVIDIA_API_KEY)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class MetadataConfig(_ConfigBase):
    """Metadata configuration."""

    max_array_length: int = Field(
        default=1000,
        env="APP_METADATA_MAXARRAYLENGTH",
        description="Maximum length for array-type metadata fields",
    )
    max_string_length: int = Field(
        default=65535,
        env="APP_METADATA_MAXSTRINGLENGTH",
        description="Maximum length for string-type metadata fields",
    )
    allow_partial_filtering: bool = Field(
        default=False,
        env="APP_METADATA_ALLOWPARTIALFILTERING",
        description="Allow partial matches in metadata filtering",
    )


class QueryDecompositionConfig(_ConfigBase):
    """Query Decomposition configuration."""

    enable_query_decomposition: bool = Field(
        default=False,
        env="ENABLE_QUERY_DECOMPOSITION",
        description="Enable breaking down complex queries into sub-queries",
    )
    recursion_depth: int = Field(
        default=3,
        env="MAX_RECURSION_DEPTH",
        description="Maximum depth for recursive query decomposition",
    )


class ReflectionConfig(_ConfigBase):
    """Reflection configuration for context relevance and response groundedness."""

    enable_reflection: bool = Field(
        default=False,
        env="ENABLE_REFLECTION",
        description="Enable self-reflection to improve answer quality",
    )
    max_loops: int = Field(
        default=3,
        env="MAX_REFLECTION_LOOP",
        description="Maximum number of reflection iterations",
    )
    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        env="REFLECTION_LLM",
        description="Model for reflection and quality assessment",
    )
    server_url: str = Field(
        default="",
        env="REFLECTION_LLM_SERVERURL",
        description="URL endpoint for reflection service",
    )
    context_relevance_threshold: int = Field(
        default=1,
        env="CONTEXT_RELEVANCE_THRESHOLD",
        description="Minimum relevance score for context to be considered useful",
    )
    response_groundedness_threshold: int = Field(
        default=1,
        env="RESPONSE_GROUNDEDNESS_THRESHOLD",
        description="Minimum groundedness score for response to be considered factual",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="REFLECTION_LLM_APIKEY",
        description="API key for reflection service (overrides global NVIDIA_API_KEY)",
    )

    @field_validator("server_url", mode="before")
    @classmethod
    def normalize_url(cls, v: Any) -> Any:
        """Normalize URL fields by stripping whitespace/quotes and adding scheme."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith(("http://", "https://")):
                return f"http://{v}"
        return v


class TavilyConfig(_ConfigBase):
    """Tavily web search fallback configuration."""

    enabled: bool = Field(
        default=False,
        env="APP_TAVILY_ENABLED",
        description="Enable Tavily web search fallback when RAG context is insufficient",
    )
    api_key: SecretStr | None = Field(
        default=None,
        env="TAVILY_API_KEY",
        description="Tavily API key (injected from k8s secret tavily-api-key)",
    )
    max_results: int = Field(
        default=5,
        env="TAVILY_MAX_RESULTS",
        description="Maximum results per Tavily search round",
    )
    include_domains: str = Field(
        default="",
        env="TAVILY_INCLUDE_DOMAINS",
        description="Comma-separated domain allowlist (empty = unrestricted)",
    )
    score_threshold: float = Field(
        default=0.6,
        env="TAVILY_SCORE_THRESHOLD",
        description="Minimum Tavily relevance score to include a result (0–1)",
    )


class NvidiaRAGConfig(_ConfigBase):
    """Main NVIDIA RAG configuration.

    Priority order (highest to lowest):
    1. Config file values (YAML/JSON)
    2. Environment variables
    3. Default values
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    vector_store: VectorStoreConfig = PydanticField(default_factory=VectorStoreConfig)
    llm: LLMConfig = PydanticField(default_factory=LLMConfig)
    query_rewriter: QueryRewriterConfig = PydanticField(
        default_factory=QueryRewriterConfig
    )
    filter_expression_generator: FilterExpressionGeneratorConfig = PydanticField(
        default_factory=FilterExpressionGeneratorConfig
    )
    text_splitter: TextSplitterConfig = PydanticField(
        default_factory=TextSplitterConfig
    )
    embeddings: EmbeddingConfig = PydanticField(default_factory=EmbeddingConfig)
    ranking: RankingConfig = PydanticField(default_factory=RankingConfig)
    retriever: RetrieverConfig = PydanticField(default_factory=RetrieverConfig)
    nv_ingest: NvIngestConfig = PydanticField(default_factory=NvIngestConfig)
    nemo_parse: NemoParseConfig = PydanticField(default_factory=NemoParseConfig)
    tracing: TracingConfig = PydanticField(default_factory=TracingConfig)
    vlm: VLMConfig = PydanticField(default_factory=VLMConfig)
    minio: MinioConfig = PydanticField(default_factory=MinioConfig)
    summarizer: SummarizerConfig = PydanticField(default_factory=SummarizerConfig)
    metadata: MetadataConfig = PydanticField(default_factory=MetadataConfig)
    query_decomposition: QueryDecompositionConfig = PydanticField(
        default_factory=QueryDecompositionConfig
    )
    reflection: ReflectionConfig = PydanticField(default_factory=ReflectionConfig)
    tavily: TavilyConfig = PydanticField(default_factory=TavilyConfig)

    # Top-level flags
    enable_guardrails: bool = Field(
        default=False,
        env="ENABLE_GUARDRAILS",
        description="Enable safety guardrails for input/output filtering",
    )
    enable_citations: bool = Field(
        default=True,
        env="ENABLE_CITATIONS",
        description="Include source citations in generated responses",
    )
    enable_vlm_inference: bool = Field(
        default=False,
        env="ENABLE_VLM_INFERENCE",
        description="Enable Vision-Language Model for multimodal queries",
    )
    vlm_to_llm_fallback: bool = Field(
        default=True,
        env="VLM_TO_LLM_FALLBACK",
        description=(
            "When true, if ENABLE_VLM_INFERENCE is on but no images are present in query, "
            "messages, or context, the pipeline will fall back to the standard LLM RAG flow. "
            "When false, VLM will be invoked even for text-only queries."
        ),
    )
    default_confidence_threshold: float = Field(
        default=0.0,
        env="RERANKER_CONFIDENCE_THRESHOLD",
        description="Default confidence threshold for reranker scores",
    )
    temp_dir: str = Field(
        default="./tmp-data",
        env="TEMP_DIR",
        description="Temporary directory for file processing and storage",
    )
    pdf_repo_dir: str = Field(
        default="",
        env="APP_PDF_REPO_DIR",
        description=(
            "Persistent directory for storing uploaded PDFs after ingestion. "
            "PDFs are stored under <pdf_repo_dir>/<collection_name>/<filename> "
            "and are NOT deleted after ingest. Empty string disables persistence."
        ),
    )
    docs_repo_dir: str = Field(
        default="",
        env="APP_DOCS_REPO_DIR",
        description=(
            "Persistent directory for storing non-PDF Office documents (DOCX, XLSX, "
            "PPTX, DOC, XLS, PPT) downloaded during web crawls. "
            "Files are stored under <docs_repo_dir>/<collection_name>/<filename>. "
            "Empty string disables persistence (files go to a temp path instead)."
        ),
    )
    audio_repo_dir: str = Field(
        default="",
        env="APP_AUDIO_REPO_DIR",
        description=(
            "Persistent directory for storing audio files (MP3, WAV, FLAC, OGG, AAC, "
            "M4A, OPUS) downloaded during web crawls. "
            "Files are stored under <audio_repo_dir>/<collection_name>/<filename>. "
            "Empty string disables persistence (files go to a temp path instead). "
            "Ingestion is triggered manually via POST /ingest-media."
        ),
    )
    video_repo_dir: str = Field(
        default="",
        env="APP_VIDEO_REPO_DIR",
        description=(
            "Persistent directory for storing video files (MP4, MKV, MOV, AVI, WEBM, "
            "TS, M4V) downloaded during web crawls. "
            "Files are stored under <video_repo_dir>/<collection_name>/<filename>. "
            "Empty string disables persistence (files go to a temp path instead). "
            "Ingestion is triggered manually via POST /ingest-media."
        ),
    )
    max_media_file_mb: int = Field(
        default=500,
        env="APP_MAX_MEDIA_FILE_MB",
        description=(
            "Maximum file size in MB for audio/video downloads during web crawl. "
            "Files larger than this limit (checked via Content-Length header before "
            "downloading) are skipped and logged as errors. Default 500 MB."
        ),
    )

    @field_validator("default_confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {v}. "
                "The confidence threshold represents the minimum relevance score required for documents to be included."
            )
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NvidiaRAGConfig":
        """Create config from dictionary.

        Priority: dict values > env vars > defaults

        Args:
            data: Configuration dictionary

        Returns:
            NvidiaRAGConfig instance
        """
        # Direct instantiation - constructor args have priority over env vars in pydantic-settings
        return cls(**data)

    @classmethod
    def from_yaml(cls, filepath: str) -> "NvidiaRAGConfig":
        """Create config from YAML file.

        Priority: YAML values > env vars > defaults

        Args:
            filepath: Path to YAML file

        Returns:
            NvidiaRAGConfig instance
        """
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filepath: str) -> "NvidiaRAGConfig":
        """Create config from JSON file.

        Priority: JSON values > env vars > defaults

        Args:
            filepath: Path to JSON file

        Returns:
            NvidiaRAGConfig instance
        """
        path = Path(filepath)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def __str__(self) -> str:
        """Return formatted config as YAML-like string for easy reading.

        Uses mode='json' to properly mask SecretStr fields (api_key, password, etc.)
        as '**********' instead of exposing actual values.
        """
        return yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            width=120,
            allow_unicode=True,
        )


# ---------------------------------------------------------------------------
# URL → product mapping used by SimpleWebCrawler and backfill scripts.
# Each entry: (url_prefix_substring, product_family, product_name | None)
# First match wins; entries ordered from most-specific to least-specific.
# ---------------------------------------------------------------------------
CRAWLER_PRODUCT_URL_MAP: list[tuple[str, str, str | None]] = [
    # ── CUDA ecosystem ────────────────────────────────────────────────────────
    ("docs.nvidia.com/cuda/profiler-users-guide",   "CUDA",          "CUDA Profiler"),
    ("docs.nvidia.com/cuda/cuda-gdb",               "CUDA",          "CUDA-GDB"),
    ("docs.nvidia.com/cuda/cuda-memcheck",          "CUDA",          "CUDA-MEMCHECK"),
    ("docs.nvidia.com/cuda/thrust",                 "CUDA",          "Thrust"),
    ("docs.nvidia.com/cuda",                        "CUDA",          "CUDA Toolkit"),
    ("developer.nvidia.com/cuda",                   "CUDA",          "CUDA Toolkit"),
    # ── Deep Learning ─────────────────────────────────────────────────────────
    ("docs.nvidia.com/deeplearning/tensorrt",        "TensorRT",      "TensorRT"),
    ("docs.nvidia.com/deeplearning/cudnn",           "cuDNN",         "cuDNN"),
    ("docs.nvidia.com/deeplearning/nemo",            "NeMo",          "NeMo Framework"),
    ("docs.nvidia.com/deeplearning/triton",          "Triton",        "Triton Inference Server"),
    ("docs.nvidia.com/deeplearning/performance",     "Deep Learning", "Deep Learning Performance"),
    ("docs.nvidia.com/deeplearning/frameworks",      "Deep Learning", "Deep Learning Frameworks"),
    ("docs.nvidia.com/deeplearning",                 "Deep Learning", None),
    ("developer.nvidia.com/deep-learning",           "Deep Learning", None),
    # ── Triton (GitHub) ───────────────────────────────────────────────────────
    ("github.com/triton-inference-server",           "Triton",        "Triton Inference Server"),
    # ── NIM / NIMs ────────────────────────────────────────────────────────────
    ("docs.nvidia.com/nim",                          "NIM",           "NVIDIA NIM"),
    ("build.nvidia.com",                             "NIM",           "NVIDIA NIM"),
    # ── Data Center / GPUs ────────────────────────────────────────────────────
    ("docs.nvidia.com/datacenter/tesla",             "Data Center",   "Tesla GPU"),
    ("docs.nvidia.com/datacenter/cloud-native",      "Data Center",   "Cloud Native"),
    ("docs.nvidia.com/datacenter",                   "Data Center",   "Data Center GPU"),
    ("docs.nvidia.com/dgx",                          "DGX",           "NVIDIA DGX"),
    ("docs.nvidia.com/gpudirect-storage",            "Data Center",   "GPUDirect Storage"),
    # ── NVIDIA Virtual GPU (vGPU / GRID) ──────────────────────────────────────
    ("docs.nvidia.com/vgpu",                         "NVIDIA Virtual GPU", "NVIDIA vGPU Software"),
    ("docs.nvidia.com/grid",                         "NVIDIA Virtual GPU", "NVIDIA GRID"),
    ("docscontent.nvidia.com/dita",                  "NVIDIA Virtual GPU", "NVIDIA GRID/vGPU"),
    # ── CUDA deployment tools ─────────────────────────────────────────────────
    ("docs.nvidia.com/deploy",                       "CUDA",          "CUDA Deployment Tools"),
    # ── GPU architectures ─────────────────────────────────────────────────────
    ("docs.nvidia.com/blackwell",                    "Blackwell",     "Blackwell Architecture"),
    ("docs.nvidia.com/hopper",                       "Hopper",        "Hopper Architecture"),
    ("docs.nvidia.com/ampere",                       "Ampere",        "Ampere Architecture"),
    ("developer.nvidia.com/blackwell",               "Blackwell",     "Blackwell Architecture"),
    # ── Networking ────────────────────────────────────────────────────────────
    ("docs.nvidia.com/networking/mlnx_ofed",         "Networking",    "MLNX OFED"),
    ("docs.nvidia.com/networking",                   "Networking",    "NVIDIA Networking"),
    ("docs.mellanox.com",                            "Networking",    "NVIDIA Networking"),
    # ── Autonomous vehicles ───────────────────────────────────────────────────
    ("docs.nvidia.com/drive",                        "DRIVE",         "NVIDIA DRIVE"),
    # ── Embedded / Jetson ─────────────────────────────────────────────────────
    ("docs.nvidia.com/jetson",                       "Jetson",        "Jetson"),
    ("developer.nvidia.com/embedded",                "Jetson",        "Jetson"),
    ("developer.nvidia.com/jetson",                  "Jetson",        "Jetson"),
    # ── Isaac ─────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/isaac",                        "Isaac",         "Isaac"),
    ("developer.nvidia.com/isaac",                   "Isaac",         "Isaac"),
    # ── Profiling / Dev tools ─────────────────────────────────────────────────
    ("docs.nvidia.com/nsight-systems",               "Nsight",        "Nsight Systems"),
    ("docs.nvidia.com/nsight-compute",               "Nsight",        "Nsight Compute"),
    ("docs.nvidia.com/nsight-visual-studio-code",    "Nsight",        "Nsight VSCode"),
    ("docs.nvidia.com/nsight",                       "Nsight",        "Nsight"),
    # ── RAPIDS ────────────────────────────────────────────────────────────────
    ("docs.rapids.ai",                               "RAPIDS",        "RAPIDS"),
    ("developer.nvidia.com/rapids",                  "RAPIDS",        "RAPIDS"),
    ("rapids.ai",                                    "RAPIDS",        "RAPIDS"),
    ("docs.nvidia.com/spark-rapids",                 "RAPIDS",        "Spark RAPIDS"),
    ("docs.nvidia.com/cupynumeric",                  "RAPIDS",        "cuPyNumeric"),
    ("docs.nvidia.com/legate",                       "RAPIDS",        "Legate NumPy"),
    # ── Modulus / Physics ML ──────────────────────────────────────────────────
    ("docs.nvidia.com/modulus",                      "Modulus",       "NVIDIA Modulus"),
    ("developer.nvidia.com/modulus",                 "Modulus",       "NVIDIA Modulus"),
    ("docs.nvidia.com/physicsnemo",                  "Modulus",       "PhysicsNeMo"),
    # ── HPC ───────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/hpc-sdk",                      "HPC",           "NVIDIA HPC SDK"),
    ("docs.nvidia.com/nvpl",                         "HPC",           "NVIDIA Performance Libraries"),
    ("docs.nvidia.com/nvshmem",                      "HPC",           "NVSHMEM"),
    ("support.brightcomputing.com",                  "HPC",           "Bright Computing HPC"),
    # ── Holoscan / Clara ──────────────────────────────────────────────────────
    ("docs.nvidia.com/holoscan",                     "Holoscan",      "NVIDIA Holoscan"),
    ("docs.nvidia.com/clara-holoscan",               "Holoscan",      "NVIDIA Holoscan"),
    ("docs.nvidia.com/clara",                        "Holoscan",      "NVIDIA Clara"),
    # ── Aerial (CUDA-Accelerated RAN) ─────────────────────────────────────────
    ("docs.nvidia.com/aerial",                       "Aerial",        "NVIDIA Aerial"),
    # ── TAO Toolkit ───────────────────────────────────────────────────────────
    ("docs.nvidia.com/tao",                          "TAO Toolkit",   "NVIDIA TAO Toolkit"),
    # ── NeMo Framework ────────────────────────────────────────────────────────
    ("docs.nvidia.com/nemo",                         "NeMo",          "NeMo Framework"),
    ("docs.nvidia.com/nemo-framework",               "NeMo",          "NeMo Framework"),
    # ── BioNeMo ───────────────────────────────────────────────────────────────
    ("docs.nvidia.com/bionemo-framework",            "BioNeMo",       "NVIDIA BioNeMo"),
    # ── Cosmos ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/cosmos",                       "Cosmos",        "NVIDIA Cosmos"),
    # ── Metropolis ────────────────────────────────────────────────────────────
    ("docs.nvidia.com/metropolis",                   "Metropolis",    "NVIDIA Metropolis"),
    # ── Morpheus ──────────────────────────────────────────────────────────────
    ("docs.nvidia.com/morpheus",                     "Morpheus",      "NVIDIA Morpheus"),
    # ── Maxine ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/maxine",                       "Maxine",        "NVIDIA Maxine"),
    # ── DOCA (Data Center / Networking) ───────────────────────────────────────
    ("docs.nvidia.com/doca",                         "Networking",    "NVIDIA DOCA"),
    # ── cuOpt ─────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/cuopt",                        "cuOpt",         "NVIDIA cuOpt"),
    # ── ACE (AI Character Engine) ──────────────────────────────────────────────
    ("docs.nvidia.com/ace",                          "AI Enterprise", "NVIDIA ACE"),
    # ── NGC Catalog ───────────────────────────────────────────────────────────
    ("docs.nvidia.com/ngc",                          "NGC",           "NGC Catalog"),
    ("assets.ngc.nvidia.com",                        "NGC",           "NGC Catalog"),
    # ── Base Command Platform / LaunchPad ─────────────────────────────────────
    ("docs.nvidia.com/base-command-platform",        "Base Command",  "NVIDIA Base Command Platform"),
    ("docs.nvidia.com/base-command-manager",         "Base Command",  "NVIDIA Base Command Manager"),
    ("docs.nvidia.com/launchpad",                    "Base Command",  "NVIDIA LaunchPad"),
    # ── Dynamo ────────────────────────────────────────────────────────────────
    ("docs.nvidia.com/dynamo",                       "Dynamo",        "NVIDIA Dynamo"),
    ("docs.dynamo.nvidia.com",                       "Dynamo",        "NVIDIA Dynamo"),
    ("github.com/ai-dynamo",                         "Dynamo",        "NVIDIA Dynamo"),
    # ── AI Enterprise / Fleet Command / Licensing ─────────────────────────────
    ("docs.nvidia.com/ai-enterprise",                "AI Enterprise", "NVIDIA AI Enterprise"),
    ("docs.nvidia.com/fleet-command",                "Fleet Command", "Fleet Command"),
    ("docs.nvidia.com/license-system",               "AI Enterprise", "NVIDIA License System"),
    # ── CUDA tooling (compute-sanitizer, cutlass, cupti) ──────────────────────
    ("docs.nvidia.com/compute-sanitizer",            "CUDA",          "CUDA Compute Sanitizer"),
    ("docs.nvidia.com/cutlass",                      "CUDA",          "CUTLASS"),
    ("docs.nvidia.com/cupti",                        "CUDA",          "CUDA CUPTI"),
    # ── Data Center infra (HGX, multi-node NVLink, attestation) ──────────────
    ("docs.nvidia.com/hgx-platforms",                "Data Center",   "NVIDIA HGX"),
    ("docs.nvidia.com/multi-node-nvlink-systems",    "Data Center",   "Multi-Node NVLink"),
    ("docs.nvidia.com/attestation",                  "Data Center",   "NVIDIA Attestation"),
    # ── Developer blog / general ──────────────────────────────────────────────
    ("developer.nvidia.com/blog",                    "Blog",          "NVIDIA Developer Blog"),
    ("developer.nvidia.com/technical-blog",          "Blog",          "NVIDIA Technical Blog"),
    ("developer.nvidia.com",                         "NVIDIA Developer", None),
    ("docs.nvidia.com",                              "NVIDIA Docs",   None),
    ("www.nvidia.com/en-us/technologies",            "NVIDIA",        None),
    ("github.com/nvidia",                            "NVIDIA",        None),
    ("github.com/NVIDIA",                            "NVIDIA",        None),
]
