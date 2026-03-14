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
Compatibility shim — re-exports from nvidia_rag.tools.crawl.

The canonical implementation has moved to nvidia_rag.tools.crawl.
This module is retained so that any existing imports of
``nvidia_rag.utils.web_crawler`` continue to work without modification.
"""
from nvidia_rag.tools.crawl import (  # noqa: F401
    _CRAWL_CANCEL,
    _CRAWL_PROGRESS,
    _MANIFEST,
    _UNCHANGED,
    SimpleWebCrawler,
    load_binary_manifest,
    save_binary_manifest,
)

__all__ = [
    "_CRAWL_CANCEL",
    "_CRAWL_PROGRESS",
    "_MANIFEST",
    "_UNCHANGED",
    "SimpleWebCrawler",
    "load_binary_manifest",
    "save_binary_manifest",
]
