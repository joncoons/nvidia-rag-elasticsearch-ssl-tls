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

import { useEffect, useRef, type JSX } from "react";
import { Stack, Flex, Text, Button, ProgressBar, Spinner } from "@kui/react";
import { Globe, Cpu, FolderOpen } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useCrawlModeStatus, useExitCrawlMode } from "../../hooks/useCrawlModeStatus";
import { useNotificationStore } from "../../store/useNotificationStore";
import type { TaskNotification } from "../../types/notifications";

interface CrawlModeGuardProps {
  children: JSX.Element | JSX.Element[];
}

/**
 * Blocks the Chat UI while nim-llm is offline (crawl mode active).
 *
 * Shows live crawl progress from pending crawl task notifications.
 * Auto-exits crawl mode once all pending crawl tasks finish.
 */
export function CrawlModeGuard({ children }: CrawlModeGuardProps) {
  const { data: crawlModeStatus } = useCrawlModeStatus();
  const exitCrawlMode = useExitCrawlMode();
  const queryClient = useQueryClient();
  const { notifications } = useNotificationStore();
  const autoExitFiredRef = useRef(false);
  const navigate = useNavigate();

  const isCrawlMode = crawlModeStatus?.active ?? false;

  // Crawl task notifications
  const crawlTasks = notifications.filter(
    (n): n is TaskNotification => n.type === "task" && n.task.task_type === "crawl"
  );
  const pendingCrawlTasks = crawlTasks.filter(n => n.task.state === "PENDING");
  const allCrawlsDone = crawlTasks.length > 0 && pendingCrawlTasks.length === 0;

  // Auto-exit crawl mode when all crawl tasks finish
  useEffect(() => {
    if (isCrawlMode && allCrawlsDone && !autoExitFiredRef.current && !exitCrawlMode.isPending) {
      autoExitFiredRef.current = true;
      exitCrawlMode.mutate(undefined, {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["crawl-mode-status"] });
        },
      });
    }
    // Reset flag when a new crawl starts
    if (!allCrawlsDone) {
      autoExitFiredRef.current = false;
    }
  }, [isCrawlMode, allCrawlsDone, exitCrawlMode, queryClient]);

  if (!isCrawlMode) {
    return <>{children}</>;
  }

  const handleManualExit = () => {
    exitCrawlMode.mutate(undefined, {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ["crawl-mode-status"] });
      },
    });
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 'calc(100vh - 48px)',
      padding: '32px',
      background: 'var(--background-color-surface-base)',
    }}>
      <Stack gap="density-xl" style={{ maxWidth: '640px', width: '100%' }}>

        {/* Header */}
        <Flex align="center" gap="density-md">
          <Globe size={32} style={{ color: 'var(--color-brand-400)' }} />
          <Stack gap="density-xs">
            <Text kind="body/bold/2xl">Crawl Mode Active</Text>
            <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
              Chat is unavailable while nim-llm is offline. GPU resources are allocated
              to Nemotron-Parse for maximum crawl throughput.
            </Text>
          </Stack>
        </Flex>

        {/* Active crawl tasks */}
        {pendingCrawlTasks.length > 0 && (
          <Stack gap="density-md"
            style={{
              background: 'var(--background-color-surface-raised)',
              borderRadius: '8px',
              padding: '16px',
              border: '1px solid var(--border-color-subtle)',
            }}
          >
            <Text kind="body/semibold/sm">Active Crawl Tasks</Text>
            {pendingCrawlTasks.map(n => {
              const { pages_crawled = 0, pages_queued = 0, pages_skipped = 0 } = n.task.result || {};
              const startUrl = n.task.start_url || n.task.documents?.[0]?.replace('Web crawl: ', '') || '';
              return (
                <Stack key={n.id} gap="density-xs">
                  <Flex align="center" gap="density-sm">
                    <Spinner size="small" aria-label="Crawling" />
                    <Stack gap="0">
                      <Text kind="body/semibold/xs">{n.task.collection_name}</Text>
                      <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)', wordBreak: 'break-all' }}>
                        {startUrl}
                      </Text>
                    </Stack>
                  </Flex>
                  <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
                    {pages_crawled > 0
                      ? `${pages_crawled} pages crawled · ${pages_skipped} unchanged · ~${pages_queued} queued`
                      : 'Starting crawl…'}
                  </Text>
                  <ProgressBar kind="indeterminate" aria-label="Crawl in progress" />
                </Stack>
              );
            })}
          </Stack>
        )}

        {/* All done — waiting for exit */}
        {allCrawlsDone && (
          <Flex align="center" gap="density-sm"
            style={{
              background: 'var(--background-color-surface-raised)',
              borderRadius: '8px',
              padding: '16px',
              border: '1px solid var(--border-color-subtle)',
            }}
          >
            <Cpu size={20} style={{ color: 'var(--color-success-400)' }} />
            <Stack gap="density-xs">
              <Text kind="body/semibold/sm">Crawl complete — restoring inference mode</Text>
              <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
                Scaling nim-llm back up. Chat will become available in a few minutes.
              </Text>
            </Stack>
          </Flex>
        )}

        {/* nim-llm loading indicator */}
        {crawlModeStatus && crawlModeStatus.nim_llm_replicas === 0 && (
          <Flex align="center" gap="density-sm" style={{ color: 'var(--text-color-subtle)' }}>
            <Cpu size={16} />
            <Text kind="body/regular/xs">
              nim-llm: offline &nbsp;·&nbsp; nemotron-parse: {crawlModeStatus.nemotron_parse_replicas} replica{crawlModeStatus.nemotron_parse_replicas !== 1 ? 's' : ''}
            </Text>
          </Flex>
        )}

        {/* Navigation + exit */}
        <Flex justify="between" align="center">
          <Button
            kind="tertiary"
            size="medium"
            onClick={() => navigate('/collections/new')}
          >
            <Flex align="center" gap="density-sm">
              <FolderOpen size={16} />
              Manage Collections
            </Flex>
          </Button>
          <Button
            kind="secondary"
            size="medium"
            onClick={handleManualExit}
            disabled={exitCrawlMode.isPending}
          >
            {exitCrawlMode.isPending ? (
              <Flex align="center" gap="density-sm">
                <Spinner size="small" aria-label="Exiting" />
                Restoring inference mode…
              </Flex>
            ) : (
              'Exit Crawl Mode'
            )}
          </Button>
        </Flex>

      </Stack>
    </div>
  );
}
