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

import { useCallback, useEffect, useState } from "react";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useNotificationStore } from "../../store/useNotificationStore";
import { openNotificationPanel } from "../notifications/NotificationBell";
import { Button, Stack, Flex, Text, Spinner, Checkbox } from "@kui/react";
import { X, RefreshCw, Music, Video } from "lucide-react";

interface MediaRow {
  source_uri: string;
  filename: string;
  local_path: string;
  referring_page_url: string;
  file_size_bytes: string;
  content_type: string;
  downloaded_at: string;
  media_type: "audio" | "video";
  collection_name: string;
}

function formatBytes(bytes: string | number): string {
  const n = Number(bytes);
  if (!n) return "unknown size";
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export const MediaQueueSection = () => {
  const { activeCollection, toggleMediaQueue } = useCollectionDrawerStore();
  const { addTaskNotification } = useNotificationStore();
  const [rows, setRows] = useState<MediaRow[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const collectionName = activeCollection?.collection_name ?? "";

  const fetchQueue = useCallback(async () => {
    if (!collectionName) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(
        `/api/media-queue?collection_name=${encodeURIComponent(collectionName)}`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRows(data.pending ?? []);
    } catch (err) {
      setError(`Failed to load media queue: ${err}`);
    } finally {
      setLoading(false);
    }
  }, [collectionName]);

  useEffect(() => {
    fetchQueue();
  }, [fetchQueue]);

  const toggleRow = (path: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(path) ? next.delete(path) : next.add(path);
      return next;
    });

  const toggleAll = () =>
    setSelected((prev) =>
      prev.size === rows.length
        ? new Set()
        : new Set(rows.map((r) => r.local_path))
    );

  const handleIngest = async () => {
    if (!selected.size || !collectionName) return;
    setIngesting(true);
    setError(null);
    try {
      const res = await fetch("/api/ingest-media", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_name: collectionName,
          local_paths: Array.from(selected),
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.message ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data?.task_id) {
        addTaskNotification({
          id: data.task_id,
          collection_name: collectionName,
          documents: Array.from(selected).map((p) => p.split("/").pop() ?? p),
          task_type: "upload",
          state: "PENDING",
          created_at: new Date().toISOString(),
        });
      }
      setSelected(new Set());
      setTimeout(() => openNotificationPanel(), 100);
      // Refresh queue after a short delay so ingested rows disappear.
      setTimeout(() => fetchQueue(), 2000);
    } catch (err) {
      setError(`Ingest failed: ${err}`);
    } finally {
      setIngesting(false);
    }
  };

  const audioRows = rows.filter((r) => r.media_type === "audio");
  const videoRows = rows.filter((r) => r.media_type === "video");

  const MediaRow = ({ row }: { row: MediaRow }) => (
    <Flex
      align="center"
      gap="density-sm"
      style={{
        padding: '8px',
        borderRadius: '4px',
        background: 'var(--surface-color-raised)',
        cursor: 'pointer',
      }}
      onClick={() => toggleRow(row.local_path)}
    >
      <Checkbox
        checked={selected.has(row.local_path)}
        onCheckedChange={() => toggleRow(row.local_path)}
        aria-label={`Select ${row.filename}`}
      />
      <Stack gap="density-xs" style={{ flex: 1, minWidth: 0 }}>
        <Text
          kind="body/regular/sm"
          style={{
            color: 'white',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {row.filename}
        </Text>
        <Flex gap="density-sm">
          <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
            {formatBytes(row.file_size_bytes)}
          </Text>
          {row.referring_page_url && (
            <Text
              kind="body/regular/xs"
              style={{
                color: 'var(--color-brand)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                maxWidth: '200px',
              }}
            >
              <a
                href={row.referring_page_url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                style={{ color: 'inherit' }}
              >
                source page
              </a>
            </Text>
          )}
        </Flex>
      </Stack>
    </Flex>
  );

  return (
    <Stack
      gap="density-xl"
      style={{
        borderTop: '1px solid var(--border-color-subtle)',
        paddingTop: '24px',
        marginTop: '24px',
      }}
    >
      <Flex justify="between" align="center">
        <Text kind="body/bold/lg" style={{ color: 'white' }}>
          Media Queue
        </Text>
        <Flex gap="density-sm" align="center">
          <Button
            kind="tertiary"
            size="small"
            onClick={fetchQueue}
            disabled={loading}
            aria-label="Refresh media queue"
          >
            <RefreshCw size={14} />
          </Button>
          <Button
            kind="tertiary"
            size="small"
            onClick={() => toggleMediaQueue(false)}
            aria-label="Close media queue"
          >
            <X size={14} />
          </Button>
        </Flex>
      </Flex>

      <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
        Audio and video files downloaded during web crawls. Select files below and click
        "Ingest Selected" to transcribe and index them via the Riva ASR pipeline.
      </Text>

      {loading && (
        <Flex align="center" gap="density-sm">
          <Spinner size="small" aria-label="Loading media queue" />
          <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>Loading...</Text>
        </Flex>
      )}

      {error && (
        <Text kind="body/regular/sm" style={{ color: 'var(--color-danger)' }}>
          {error}
        </Text>
      )}

      {!loading && rows.length === 0 && !error && (
        <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
          No pending media files. Run a web crawl with "Extract linked files" enabled to
          populate the queue.
        </Text>
      )}

      {rows.length > 0 && (
        <>
          <Flex justify="between" align="center">
            <Flex gap="density-sm" align="center">
              <Checkbox
                checked={selected.size === rows.length && rows.length > 0}
                onCheckedChange={toggleAll}
                aria-label="Select all"
              />
              <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                {selected.size > 0 ? `${selected.size} of ${rows.length} selected` : `${rows.length} files`}
              </Text>
            </Flex>
          </Flex>

          {audioRows.length > 0 && (
            <Stack gap="density-sm">
              <Flex gap="density-xs" align="center">
                <Music size={14} style={{ color: 'var(--text-color-subtle)' }} />
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                  Audio ({audioRows.length})
                </Text>
              </Flex>
              {audioRows.map((row) => (
                <MediaRow key={row.local_path} row={row} />
              ))}
            </Stack>
          )}

          {videoRows.length > 0 && (
            <Stack gap="density-sm">
              <Flex gap="density-xs" align="center">
                <Video size={14} style={{ color: 'var(--text-color-subtle)' }} />
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                  Video ({videoRows.length})
                </Text>
              </Flex>
              {videoRows.map((row) => (
                <MediaRow key={row.local_path} row={row} />
              ))}
            </Stack>
          )}

          <Button
            onClick={handleIngest}
            disabled={selected.size === 0 || ingesting}
            kind="primary"
            color="brand"
            size="large"
            style={{ width: '100%' }}
          >
            {ingesting ? (
              <Flex align="center" gap="density-sm">
                <Spinner size="small" aria-label="Ingesting" />
                Starting ingest...
              </Flex>
            ) : (
              `Ingest Selected (${selected.size})`
            )}
          </Button>
        </>
      )}
    </Stack>
  );
};
