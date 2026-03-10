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

import { useEffect, useCallback, useState } from "react";
import NvidiaUpload from "../components/files/NvidiaUpload";
import MetadataSchemaEditor from "../components/schema/MetadataSchemaEditor";
import NewCollectionButtons from "../components/collections/NewCollectionButtons";
import { CollectionConfigurationPanel } from "../components/collections/CollectionConfigurationPanel";
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import { Block, FormField, Grid, GridItem, PageHeader, Panel, Stack, TextInput, Select, Text, Tag, Flex, Switch } from "@kui/react";
import { X, ChevronDown, BookOpen, Globe } from "lucide-react";

/**
 * New Collection page component for creating collections.
 * 
 * Provides a multi-step interface for collection creation including
 * file upload, metadata schema definition, and collection naming.
 * 
 * @returns The new collection page component
 */
// Business domain options
const BUSINESS_DOMAINS = [
  'Engineering',
  'Finance', 
  'Legal',
  'Marketing',
  'Operations',
  'Product',
  'Sales',
  'Support',
  'Other'
];

// Status options
const STATUS_OPTIONS = ['Active', 'Archived', 'Deprecated'];

// Catalog Metadata Section Component
interface CatalogMetadataSectionProps {
  catalogMetadata: {
    description: string;
    tags: string[];
    owner: string;
    business_domain: string;
    status: 'Active' | 'Archived' | 'Deprecated';
  };
  setCatalogMetadata: (updates: Partial<CatalogMetadataSectionProps['catalogMetadata']>) => void;
  onAddTag: (tag: string) => void;
  onRemoveTag: (tag: string) => void;
}

function CatalogMetadataSection({ 
  catalogMetadata, 
  setCatalogMetadata, 
  onAddTag, 
  onRemoveTag 
}: CatalogMetadataSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [tagInput, setTagInput] = useState('');

  return (
    <Panel
      slotHeading={
        <Flex 
          align="center" 
          justify="between" 
          style={{ width: '100%', cursor: 'pointer' }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span>Data Catalog</span>
          <ChevronDown 
            size={16} 
            style={{ 
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease'
            }} 
          />
        </Flex>
      }
      slotIcon={<BookOpen size={20} />}
    >
      <Text kind="body/bold/md">
        Optional metadata for organizing, categorizing, and governing your collections.
      </Text>

      {isExpanded && (
        <Stack gap="density-md" style={{ marginTop: 'var(--spacing-density-lg)' }}>
          <FormField
            slotLabel="Description"
            slotHelp="Human-readable description of the collection."
          >
            <TextInput
              value={catalogMetadata.description}
              onValueChange={(value) => setCatalogMetadata({ description: value })}
              placeholder="e.g., Q4 2024 Financial Reports"
            />
          </FormField>

          <FormField
            slotLabel="Tags"
            slotHelp="Tags for categorization and discovery. Press Enter to add."
          >
            <Stack gap="density-sm">
              <TextInput
                placeholder="Add a tag and press Enter"
                value={tagInput}
                onValueChange={setTagInput}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    if (tagInput.trim()) {
                      onAddTag(tagInput.trim());
                      setTagInput('');
                    }
                  }
                }}
              />
              {catalogMetadata.tags.length > 0 && (
                <Flex gap="density-xs" style={{ flexWrap: 'wrap' }}>
                  {catalogMetadata.tags.map((tag) => (
                    <Tag
                      key={tag}
                      color="gray"
                      kind="outline"
                      density="compact"
                      onClick={() => onRemoveTag(tag)}
                      style={{ cursor: 'pointer' }}
                    >
                      {tag} <X size={12} />
                    </Tag>
                  ))}
                </Flex>
              )}
            </Stack>
          </FormField>

          <FormField
            slotLabel="Owner"
            slotHelp="Team or person responsible for this collection."
          >
            <TextInput
              value={catalogMetadata.owner}
              onValueChange={(value) => setCatalogMetadata({ owner: value })}
              placeholder="e.g., Finance Team"
            />
          </FormField>

          <FormField
            slotLabel="Business Domain"
            slotHelp="Business domain or department."
          >
            <Select
              items={BUSINESS_DOMAINS}
              value={catalogMetadata.business_domain}
              onValueChange={(value) => setCatalogMetadata({ business_domain: value })}
              placeholder="Select a domain"
            />
          </FormField>

          <FormField
            slotLabel="Status"
            slotHelp="Collection lifecycle status."
          >
            <Select
              items={STATUS_OPTIONS}
              value={catalogMetadata.status}
              onValueChange={(value) => setCatalogMetadata({ status: value as 'Active' | 'Archived' | 'Deprecated' })}
            />
          </FormField>
        </Stack>
      )}
    </Panel>
  );
}

// Web Crawl Panel Component
function WebCrawlPanel() {
  const { crawlConfig, setCrawlConfig } = useNewCollectionStore();
  const [isExpanded, setIsExpanded] = useState(false);
  const [urlError, setUrlError] = useState('');

  const validateUrl = (url: string): boolean => {
    try { new URL(url); return true; } catch { return false; }
  };

  return (
    <Panel
      slotHeading={
        <Flex
          align="center"
          justify="between"
          style={{ width: '100%', cursor: 'pointer' }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span>Web Crawl</span>
          <ChevronDown
            size={16}
            style={{
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease',
            }}
          />
        </Flex>
      }
      slotIcon={<Globe size={20} />}
    >
      <Text kind="body/bold/md">
        Optionally seed this collection by crawling a website after creation.
      </Text>

      {isExpanded && (
        <Stack gap="density-md" style={{ marginTop: 'var(--spacing-density-lg)' }}>
          <FormField
            slotLabel="Start URL"
            slotHelp="Root page to begin crawling from."
          >
            <TextInput
              value={crawlConfig.startUrl}
              onValueChange={(val: string) => {
                setCrawlConfig({ startUrl: val });
                setUrlError(val && !validateUrl(val) ? 'Please enter a valid URL (e.g. https://docs.example.com/)' : '');
              }}
              placeholder="https://docs.example.com/"
            />
            {urlError && (
              <Text kind="body/regular/xs" style={{ color: 'var(--color-danger)' }}>
                {urlError}
              </Text>
            )}
          </FormField>

          <Stack gap="density-xs">
            <Switch
              checked={crawlConfig.useSitemap}
              onCheckedChange={(checked: boolean) => setCrawlConfig({ useSitemap: checked })}
              size="medium"
              slotLabel="Sitemap seeding"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Discover all pages via robots.txt sitemaps before crawling. Recommended for sites with JavaScript-rendered navigation.
            </Text>
          </Stack>

          <FormField
            slotLabel="Max pages"
            slotHelp="Maximum number of HTML pages to crawl. Toggle Unlimited to crawl the entire site."
          >
            <Flex align="center" gap="density-md">
              <TextInput
                type="number"
                value={crawlConfig.maxPages === null ? "" : String(crawlConfig.maxPages)}
                onValueChange={(val: string) => {
                  if (crawlConfig.maxPages !== null) {
                    setCrawlConfig({ maxPages: Math.max(1, Number(val) || 50) });
                  }
                }}
                disabled={crawlConfig.maxPages === null}
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
              />
            </Flex>
          </FormField>

          <FormField
            slotLabel="Max depth"
            slotHelp="BFS hops from the start URL. Depth 0 = start page only; 1 = directly linked pages. Toggle Unlimited for no depth cap."
          >
            <Flex align="center" gap="density-md">
              <TextInput
                type="number"
                value={crawlConfig.maxDepth === null ? "" : String(crawlConfig.maxDepth)}
                onValueChange={(val: string) => {
                  if (crawlConfig.maxDepth !== null) {
                    setCrawlConfig({ maxDepth: Math.max(0, Number(val) || 0) });
                  }
                }}
                disabled={crawlConfig.maxDepth === null}
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
              />
            </Flex>
          </FormField>

          <FormField
            slotLabel="Batch ingest size"
            slotHelp="Files accumulated before dispatching an ingest batch (1–500). Smaller = more parallelism; larger = less overhead. Default is 20."
          >
            <TextInput
              type="number"
              value={String(crawlConfig.batchIngestSize)}
              onValueChange={(val: string) =>
                setCrawlConfig({ batchIngestSize: Math.max(1, Math.min(500, Number(val) || 20)) })
              }
              style={{ width: '120px' }}
            />
          </FormField>

          <FormField
            slotLabel="URL prefix filter"
            slotHelp="Restrict crawl to URLs starting with these prefixes (comma-separated). Leave blank to auto-derive from start URL path, or allow the full domain if the start URL has no path."
          >
            <TextInput
              value={crawlConfig.allowedUrlPrefixes}
              onValueChange={(val: string) => setCrawlConfig({ allowedUrlPrefixes: val })}
              placeholder="Auto-derived from start URL path"
              style={{ width: '100%' }}
            />
          </FormField>

          <Stack gap="density-xs">
            <Switch
              checked={crawlConfig.extractLinkedFiles}
              onCheckedChange={(checked: boolean) => setCrawlConfig({ extractLinkedFiles: checked })}
              size="medium"
              slotLabel="Extract linked files (PDF / DOCX / XLSX)"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Download and ingest binary documents found as href links on crawled pages.
            </Text>

            <Switch
              checked={crawlConfig.useCrawlNemotronParse || crawlConfig.forceCrawlNemotronParse}
              onCheckedChange={(checked: boolean) =>
                setCrawlConfig({
                  useCrawlNemotronParse: checked,
                  forceCrawlNemotronParse: checked ? crawlConfig.forceCrawlNemotronParse : false,
                })
              }
              size="medium"
              slotLabel="Nemotron Parse (complex data elements)"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Route PDFs with tables/charts through nemoretriever-parse VLM.
            </Text>

            <Switch
              checked={crawlConfig.forceCrawlNemotronParse}
              onCheckedChange={(checked: boolean) =>
                setCrawlConfig({
                  forceCrawlNemotronParse: checked,
                  useCrawlNemotronParse: checked ? true : crawlConfig.useCrawlNemotronParse,
                })
              }
              size="medium"
              slotLabel="Nemotron Parse (all pages)"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Skip classification and run all PDF pages through VLM.
            </Text>
          </Stack>
        </Stack>
      )}
    </Panel>
  );
}

export default function NewCollection() {
  const { 
    collectionName, 
    setCollectionName, 
    setCollectionNameTouched, 
    catalogMetadata,
    setCatalogMetadata,
    collectionConfig,
    setCollectionConfig,
    reset 
  } = useNewCollectionStore();

  useEffect(() => {
    // cleanup when leaving the page
    return () => {
      reset();
    };
  }, [reset]);

  const handleValidationChange = useCallback((hasInvalidFiles: boolean) => {
    const { setHasInvalidFiles } = useNewCollectionStore.getState();
    setHasInvalidFiles(hasInvalidFiles);
  }, []);

  const handleFilesChange = useCallback((files: File[]) => {
    const { setFiles } = useNewCollectionStore.getState();
    setFiles(files);
  }, []);

  const handleAddTag = useCallback((tag: string) => {
    if (tag.trim() && !catalogMetadata.tags.includes(tag.trim())) {
      setCatalogMetadata({ tags: [...catalogMetadata.tags, tag.trim()] });
    }
  }, [catalogMetadata.tags, setCatalogMetadata]);

  const handleRemoveTag = useCallback((tagToRemove: string) => {
    setCatalogMetadata({ 
      tags: catalogMetadata.tags.filter(t => t !== tagToRemove) 
    });
  }, [catalogMetadata.tags, setCatalogMetadata]);

  return (
    <Grid cols={12} gap="density-lg" padding="density-lg">
      <GridItem cols={12}>
        <Block padding="density-lg">
          <PageHeader
            slotHeading="Create New Collection"
            slotSubheading="Upload source files and define metadata schema for this collection."
          />
        </Block>
      </GridItem>
      <GridItem cols={6}>
        <Panel>
          <Stack gap="density-lg">
            <FormField
              slotLabel="Collection Name"
              slotHelp="We will automatically try to validate the collection name."
              required
            >
              <TextInput
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value.replace(/\s+/g, "_"))}
                onBlur={() => setCollectionNameTouched(true)}
              />
            </FormField>

            {/* Catalog Metadata Section */}
            <CatalogMetadataSection 
              catalogMetadata={catalogMetadata}
              setCatalogMetadata={setCatalogMetadata}
              onAddTag={handleAddTag}
              onRemoveTag={handleRemoveTag}
            />

            {/* Collection Configuration Section */}
            <CollectionConfigurationPanel
              generateSummary={collectionConfig.generateSummary}
              onGenerateSummaryChange={(value) => setCollectionConfig({ generateSummary: value })}
              useNemotronParse={collectionConfig.useNemotronParse}
              onUseNemotronParseChange={(value) => setCollectionConfig({ useNemotronParse: value, forceNemotronParse: value ? collectionConfig.forceNemotronParse : false })}
              forceNemotronParse={collectionConfig.forceNemotronParse}
              onForceNemotronParseChange={(value) => setCollectionConfig({ forceNemotronParse: value, useNemotronParse: value ? true : collectionConfig.useNemotronParse })}
            />

            <MetadataSchemaEditor />
          </Stack>
        </Panel>
      </GridItem>
      <GridItem cols={6}>
        <Stack gap="density-lg">
          <Panel>
            <Stack
              gap="density-xl"
              style={{
                borderTop: '1px solid var(--border-color-subtle)',
              }}
            >
              <NvidiaUpload
                onFilesChange={handleFilesChange}
                onValidationChange={handleValidationChange}
                acceptedTypes={['.bmp', '.docx', '.html', '.jpeg', '.json', '.md', '.pdf', '.png', '.pptx', '.sh', '.tiff', '.txt', '.mp3', '.wav', '.mp4', '.mov', '.avi', '.mkv']}
                maxFileSize={400}
              />
            </Stack>
          </Panel>
          <WebCrawlPanel />
        </Stack>
      </GridItem>
      <GridItem cols={12}>
        <NewCollectionButtons />
      </GridItem>
    </Grid>
  );
}