// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { useQuery, useMutation } from "@tanstack/react-query";

export interface CrawlModeStatus {
  active: boolean;
  nim_llm_replicas: number;
  nemotron_parse_replicas: number;
}

async function fetchCrawlModeStatus(): Promise<CrawlModeStatus> {
  const res = await fetch("/api/crawl-mode/status");
  if (!res.ok) throw new Error(`crawl-mode/status ${res.status}`);
  return res.json();
}

async function postExitCrawlMode(): Promise<void> {
  const res = await fetch("/api/crawl-mode/exit", { method: "POST" });
  if (!res.ok) throw new Error(`crawl-mode/exit ${res.status}`);
}

/**
 * Polls /api/crawl-mode/status every 15 s to detect whether nim-llm is offline.
 * Returns the raw status and a boolean `isCrawlMode`.
 */
export function useCrawlModeStatus() {
  return useQuery<CrawlModeStatus>({
    queryKey: ["crawl-mode-status"],
    queryFn: fetchCrawlModeStatus,
    refetchInterval: 15_000,
    refetchIntervalInBackground: true,
    // Don't treat a failure as an error — k8s may be unavailable in dev
    retry: false,
    // Default to "not in crawl mode" while loading / on error
    placeholderData: { active: false, nim_llm_replicas: 1, nemotron_parse_replicas: 1 },
  });
}

/**
 * Mutation that calls POST /api/crawl-mode/exit and then refetches the status.
 */
export function useExitCrawlMode() {
  return useMutation({
    mutationFn: postExitCrawlMode,
  });
}
