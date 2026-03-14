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
nvidia_rag.tools — end-to-end ingest tools.

Each tool owns its full pipeline from input (file, URL) to storage (ES).
Tools use nvidia_rag.storage.embed_store as the shared storage primitive.

Tools:
    parse_document  — PDF → Nemotron-Parse → semantic chunks → ES
    crawl           — URL → BFS crawl → HTML chunks + binaries → ES  (Phase 3)
"""
