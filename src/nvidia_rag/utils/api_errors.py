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

"""Shared API error primitives used by both the RAG server and ingestor server.

Keeping these in utils/ ensures neither server package depends on the other.
"""

import logging

logger = logging.getLogger(__name__)


class ErrorCodeMapping:
    """Centralized mapping for HTTP status codes based on error types"""

    SUCCESS = 200
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    REQUEST_TIMEOUT = 408
    UNPROCESSABLE_ENTITY = 422
    CLIENT_CLOSED_REQUEST = 499
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


class APIError(Exception):
    """Custom exception class for API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        if status_code is None:
            status_code = ErrorCodeMapping.BAD_REQUEST
        logger.error("APIError occurred: %s with HTTP status: %d", message, status_code)
        self.message = message
        self.status_code = status_code
        super().__init__(message)
