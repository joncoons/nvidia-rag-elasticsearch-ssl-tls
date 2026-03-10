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
import { Globe, Cpu, FolderOpen, CheckCircle } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { useCrawlModeStatus, useExitCrawlMode } from "../../hooks/useCrawlModeStatus";
import { useNotificationStore } from "../../store/useNotificationStore";
import type { TaskNotification } from "../../types/notifications";

interface CrawlModeGuardProps {
  children: JSX.Element | JSX.Element[];
}

/**
 * Blocks the Chat UI while nim-llm is offline (crawl mode) or warming up (restoring).
 *
 * Phase 1 — active:    nim-llm=0, crawl in progress. Shows live progress.
 * Phase 2 — restoring: nim-llm scaling up, not yet ready. Shows warm-up status.
 * Phase 3 — ready:     nim-llm ready_replicas >= spec. Guard disappears, chat accessible.
 */
export function CrawlModeGuard({ children }: CrawlModeGuardProps) {
  const { data: status } = useCrawlModeStatus();
  const exitCrawlMode = useExitCrawlMode();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { notifications } = useNotificationStore();
  const autoExitFiredRef = useRef(false);

  const isCrawlMode = status?.active ?? false;
  const isRestoring = status?.restoring ?? false;
  const shouldBlock = isCrawlMode || isRestoring;

  // Ingest task notifications (both crawl and file upload)
  const ingestTasks = notifications.filter(
    (n): n is TaskNotification =>
      n.type === "task" &&
      (n.task.task_type === "crawl" || n.task.task_type === "upload")
  );
  const pendingIngestTasks = ingestTasks.filter(n => n.task.state === "PENDING");
  const pendingCrawlTasks = pendingIngestTasks.filter(n => n.task.task_type === "crawl");
  const pendingUploadTasks = pendingIngestTasks.filter(n => n.task.task_type === "upload");
  const allIngestDone = ingestTasks.length > 0 && pendingIngestTasks.length === 0;

  // Auto-exit when all ingest tasks finish
  useEffect(() => {
    if (isCrawlMode && allIngestDone && !autoExitFiredRef.current && !exitCrawlMode.isPending) {
      autoExitFiredRef.current = true;
      exitCrawlMode.mutate(undefined, {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["crawl-mode-status"] });
        },
      });
    }
    if (!allIngestDone) {
      autoExitFiredRef.current = false;
    }
  }, [isCrawlMode, allIngestDone, exitCrawlMode, queryClient]);

  if (!shouldBlock) {
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

        {/* ── Phase 1: Ingestion active ── */}
        {isCrawlMode && (
          <>
            <Flex align="center" gap="density-md">
              <Globe size={32} style={{ color: 'var(--color-brand-400)' }} />
              <Stack gap="density-xs">
                <Text kind="body/bold/2xl">Ingestion Mode Active</Text>
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                  Chat is unavailable while nim-llm is offline. GPU resources are
                  allocated to Nemotron-Parse for maximum ingestion throughput.
                </Text>
              </Stack>
            </Flex>

            {/* Active crawl task progress */}
            {pendingCrawlTasks.length > 0 && (
              <Stack gap="density-md" style={{
                background: 'var(--background-color-surface-raised)',
                borderRadius: '8px',
                padding: '16px',
                border: '1px solid var(--border-color-subtle)',
              }}>
                <Text kind="body/semibold/sm">Active Web Crawls</Text>
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

            {/* Active file upload task progress */}
            {pendingUploadTasks.length > 0 && (
              <Stack gap="density-md" style={{
                background: 'var(--background-color-surface-raised)',
                borderRadius: '8px',
                padding: '16px',
                border: '1px solid var(--border-color-subtle)',
              }}>
                <Text kind="body/semibold/sm">Active File Ingestions</Text>
                {pendingUploadTasks.map(n => {
                  const { documents = [], total_documents = 0 } = n.task.result || {};
                  const fileCount = n.task.documents?.length ?? 0;
                  return (
                    <Stack key={n.id} gap="density-xs">
                      <Flex align="center" gap="density-sm">
                        <Spinner size="small" aria-label="Ingesting" />
                        <Stack gap="0">
                          <Text kind="body/semibold/xs">{n.task.collection_name}</Text>
                          <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
                            {fileCount} file{fileCount !== 1 ? 's' : ''}
                          </Text>
                        </Stack>
                      </Flex>
                      <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
                        {total_documents > 0
                          ? `${documents.length} / ${total_documents} processed`
                          : 'Processing…'}
                      </Text>
                      <ProgressBar kind="indeterminate" aria-label="Ingest in progress" />
                    </Stack>
                  );
                })}
              </Stack>
            )}

            {/* All ingest tasks done — waiting for exit to fire */}
            {allIngestDone && (
              <Flex align="center" gap="density-sm" style={{
                background: 'var(--background-color-surface-raised)',
                borderRadius: '8px',
                padding: '16px',
                border: '1px solid var(--border-color-subtle)',
              }}>
                <Spinner size="small" aria-label="Restoring" />
                <Text kind="body/regular/sm">
                  Ingestion complete — initiating inference mode restore…
                </Text>
              </Flex>
            )}

            {/* GPU state indicator */}
            {status && status.nim_llm_replicas === 0 && (
              <Flex align="center" gap="density-sm" style={{ color: 'var(--text-color-subtle)' }}>
                <Cpu size={16} />
                <Text kind="body/regular/xs">
                  nim-llm: offline &nbsp;·&nbsp; nemotron-parse: {status.nemotron_parse_replicas} replica{status.nemotron_parse_replicas !== 1 ? 's' : ''}
                </Text>
              </Flex>
            )}

            <Flex justify="between" align="center">
              <Button kind="tertiary" size="medium" onClick={() => navigate('/collections/new')}>
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
                    Restoring…
                  </Flex>
                ) : 'Exit Crawl Mode'}
              </Button>
            </Flex>
          </>
        )}

        {/* ── Phase 2: Inference mode restoring ── */}
        {isRestoring && (
          <>
            <Flex align="center" gap="density-md">
              <Cpu size={32} style={{ color: 'var(--color-brand-400)' }} />
              <Stack gap="density-xs">
                <Text kind="body/bold/2xl">Restoring Inference Mode</Text>
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                  nim-llm is loading. Chat will become available automatically once the
                  model is ready.
                </Text>
              </Stack>
            </Flex>

            <Stack gap="density-md" style={{
              background: 'var(--background-color-surface-raised)',
              borderRadius: '8px',
              padding: '16px',
              border: '1px solid var(--border-color-subtle)',
            }}>
              {/* nemotron-parse scaling down */}
              <Flex justify="between" align="center">
                <Flex align="center" gap="density-sm">
                  {(status?.nemotron_parse_ready_replicas ?? 0) === 0 ? (
                    <CheckCircle size={16} style={{ color: 'var(--color-success-400)' }} />
                  ) : (
                    <Spinner size="small" aria-label="Scaling down" />
                  )}
                  <Text kind="body/regular/sm">Nemotron-Parse</Text>
                </Flex>
                <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
                  {(status?.nemotron_parse_ready_replicas ?? 0) === 0
                    ? 'scaled down'
                    : `${status?.nemotron_parse_ready_replicas} pod${(status?.nemotron_parse_ready_replicas ?? 0) !== 1 ? 's' : ''} terminating`}
                </Text>
              </Flex>

              {/* nim-llm warming up */}
              <Flex justify="between" align="center">
                <Flex align="center" gap="density-sm">
                  {(status?.nim_llm_ready_replicas ?? 0) >= (status?.nim_llm_replicas ?? 1) ? (
                    <CheckCircle size={16} style={{ color: 'var(--color-success-400)' }} />
                  ) : (
                    <Spinner size="small" aria-label="Loading model" />
                  )}
                  <Text kind="body/regular/sm">nim-llm</Text>
                </Flex>
                <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
                  {(status?.nim_llm_ready_replicas ?? 0) > 0 ? 'ready' : 'loading model…'}
                </Text>
              </Flex>

              <ProgressBar kind="indeterminate" aria-label="Restoring inference mode" />
            </Stack>

            <Flex justify="start">
              <Button kind="tertiary" size="medium" onClick={() => navigate('/collections/new')}>
                <Flex align="center" gap="density-sm">
                  <FolderOpen size={16} />
                  Manage Collections
                </Flex>
              </Button>
            </Flex>
          </>
        )}

      </Stack>
    </div>
  );
}
