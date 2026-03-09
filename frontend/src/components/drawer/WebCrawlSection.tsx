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

import { useCallback, useState } from "react";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useCollectionActions } from "../../hooks/useCollectionActions";
import { Button, Stack, Flex, Text, Switch, Spinner, TextInput } from "@kui/react";

const CloseIcon = () => (
  <svg style={{ width: '16px', height: '16px', color: 'white' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

export const WebCrawlSection = () => {
  const { crawlConfig, setCrawlConfig } = useNewCollectionStore();
  const { toggleCrawler } = useCollectionDrawerStore();
  const { handleStartCrawl } = useCollectionActions();
  const [isCrawling, setIsCrawling] = useState(false);
  const [urlError, setUrlError] = useState("");

  const handleClose = useCallback(() => {
    toggleCrawler(false);
    useNewCollectionStore.getState().reset();
  }, [toggleCrawler]);

  const validateUrl = (url: string): boolean => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  };

  const handleSubmit = async () => {
    if (!crawlConfig.startUrl || !validateUrl(crawlConfig.startUrl)) {
      setUrlError("Please enter a valid URL");
      return;
    }
    setIsCrawling(true);
    try {
      await handleStartCrawl();
    } finally {
      setIsCrawling(false);
    }
  };

  const canSubmit = !!crawlConfig.startUrl && !urlError && !isCrawling;

  return (
    <Stack
      gap="density-xl"
      style={{
        borderTop: '1px solid var(--border-color-subtle)',
        paddingTop: '24px',
        marginTop: '24px',
      }}
    >
      <Flex justify="between" align="center" style={{ marginBottom: '16px' }}>
        <Text kind="body/bold/lg" style={{ color: 'white' }}>
          Crawl Website
        </Text>
        <Button
          onClick={handleClose}
          disabled={isCrawling}
          kind="tertiary"
          size="small"
          data-testid="crawler-close-button"
        >
          <CloseIcon />
        </Button>
      </Flex>

      <Stack gap="density-sm">
        <Text kind="body/regular/sm" style={{ color: 'white' }}>
          Start URL
        </Text>
        <TextInput
          value={crawlConfig.startUrl}
          onValueChange={(val: string) => {
            setCrawlConfig({ startUrl: val });
            if (val && !validateUrl(val)) {
              setUrlError("Please enter a valid URL (e.g. https://example.com)");
            } else {
              setUrlError("");
            }
          }}
          placeholder="https://docs.example.com/"
          disabled={isCrawling}
          style={{ width: '100%' }}
        />
        {urlError && (
          <Text kind="body/regular/xs" style={{ color: 'var(--color-danger)' }}>
            {urlError}
          </Text>
        )}
      </Stack>

      <Stack gap="density-sm">
        <Text kind="body/regular/sm" style={{ color: 'white' }}>
          Max pages
        </Text>
        <Flex align="center" gap="density-md">
          <TextInput
            type="number"
            value={crawlConfig.maxPages === null ? "" : String(crawlConfig.maxPages)}
            onValueChange={(val: string) => {
              if (crawlConfig.maxPages !== null) {
                setCrawlConfig({ maxPages: Math.max(1, Number(val) || 50) });
              }
            }}
            disabled={isCrawling || crawlConfig.maxPages === null}
            style={{ width: '120px' }}
            placeholder="50"
          />
          <Switch
            checked={crawlConfig.maxPages === null}
            onCheckedChange={(checked: boolean) =>
              setCrawlConfig({ maxPages: checked ? null : 50 })
            }
            size="medium"
            slotLabel="Unlimited"
            disabled={isCrawling}
          />
        </Flex>
      </Stack>

      <Stack gap="density-sm">
        <Text kind="body/regular/sm" style={{ color: 'white' }}>
          Max depth
        </Text>
        <Flex align="center" gap="density-md">
          <TextInput
            type="number"
            value={crawlConfig.maxDepth === null ? "" : String(crawlConfig.maxDepth)}
            onValueChange={(val: string) => {
              if (crawlConfig.maxDepth !== null) {
                setCrawlConfig({ maxDepth: Math.max(0, Number(val) || 0) });
              }
            }}
            disabled={isCrawling || crawlConfig.maxDepth === null}
            style={{ width: '120px' }}
            placeholder="unlimited"
          />
          <Switch
            checked={crawlConfig.maxDepth === null}
            onCheckedChange={(checked: boolean) =>
              setCrawlConfig({ maxDepth: checked ? null : 3 })
            }
            size="medium"
            slotLabel="Unlimited"
            disabled={isCrawling}
          />
        </Flex>
        <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
          BFS hops from start URL. Depth 0 = start page only; 1 = directly linked pages, etc.
        </Text>
      </Stack>

      <Stack gap="density-sm">
        <Text kind="body/regular/sm" style={{ color: 'white' }}>
          URL prefix filter
        </Text>
        <TextInput
          value={crawlConfig.allowedUrlPrefixes}
          onValueChange={(val: string) => setCrawlConfig({ allowedUrlPrefixes: val })}
          placeholder="Auto-derived from start URL path"
          disabled={isCrawling}
          style={{ width: '100%' }}
        />
        <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
          Restrict crawl to URLs starting with these prefixes (comma-separated). Leave blank to auto-derive from start URL path, or crawl the full domain if the start URL has no path.
        </Text>
      </Stack>

      <Flex style={{ paddingTop: '4px' }}>
        <Stack gap="density-xs">
          <Switch
            checked={crawlConfig.extractLinkedFiles}
            onCheckedChange={(checked: boolean) => setCrawlConfig({ extractLinkedFiles: checked })}
            size="medium"
            slotLabel="Extract linked files (PDF / DOCX / XLSX)"
            disabled={isCrawling}
          />
          <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
            Download and ingest binary documents found as href links on crawled pages.
          </Text>

          <Switch
            checked={crawlConfig.useCrawlNemotronParse || crawlConfig.forceCrawlNemotronParse}
            onCheckedChange={(checked: boolean) => setCrawlConfig({
              useCrawlNemotronParse: checked,
              forceCrawlNemotronParse: checked ? crawlConfig.forceCrawlNemotronParse : false,
            })}
            size="medium"
            slotLabel="Nemotron Parse (complex data elements)"
            disabled={isCrawling || crawlConfig.forceCrawlNemotronParse}
          />
          <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
            Route PDFs containing tables, charts, or infographics through nemoretriever-parse VLM.
          </Text>

          <Switch
            checked={crawlConfig.forceCrawlNemotronParse}
            onCheckedChange={(checked: boolean) => setCrawlConfig({
              forceCrawlNemotronParse: checked,
              useCrawlNemotronParse: checked ? true : crawlConfig.useCrawlNemotronParse,
            })}
            size="medium"
            slotLabel="Nemotron Parse (all)"
            disabled={isCrawling}
          />
          <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
            Skip classification and run all PDF pages through VLM. Implies Nemotron Parse enabled.
          </Text>
        </Stack>
      </Flex>

      <Button
        onClick={handleSubmit}
        disabled={!canSubmit}
        kind="primary"
        color="brand"
        size="large"
        style={{ width: '100%' }}
      >
        {isCrawling ? (
          <Flex align="center" gap="density-sm">
            <Spinner size="small" aria-label="Starting crawl" />
            Starting crawl...
          </Flex>
        ) : (
          "Start Crawl"
        )}
      </Button>
    </Stack>
  );
};
