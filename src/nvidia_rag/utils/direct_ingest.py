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
"""
Compatibility shim — re-exports from nvidia_rag.storage.embed_store.

The canonical implementation has moved to nvidia_rag.storage.embed_store.
This module is retained so that any existing imports of
``nvidia_rag.utils.direct_ingest`` continue to work without modification.
"""
from nvidia_rag.storage.embed_store import (  # noqa: F401
    CHARS_PER_TOKEN,
    EMBED_BATCH_SIZE,
    ES_BULK_BATCH_SIZE,
    _build_doc,
    _build_ssl_ctx,
    bulk_write_to_es,
    chunk_text,
    embed_chunks,
    ingest_chunk_files,
    ingest_text,
)

__all__ = [
    "CHARS_PER_TOKEN",
    "EMBED_BATCH_SIZE",
    "ES_BULK_BATCH_SIZE",
    "_build_doc",
    "_build_ssl_ctx",
    "bulk_write_to_es",
    "chunk_text",
    "embed_chunks",
    "ingest_chunk_files",
    "ingest_text",
]
