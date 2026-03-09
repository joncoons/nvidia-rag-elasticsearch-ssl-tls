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

import { create } from "zustand";
import type { UIMetadataField } from "../types/collections";

/**
 * Get the default value for a metadata field based on its type.
 */
const getDefaultValueForField = (field: UIMetadataField): unknown => {
  switch (field.type) {
    case "string":
    case "datetime":
      return "";
    case "integer":
    case "float":
    case "number":
      return null;
    case "boolean":
      return false;
    case "array":
      return [];
    default:
      return "";
  }
};

/**
 * Catalog metadata for collection organization and governance.
 */
interface CatalogMetadata {
  description: string;
  tags: string[];
  owner: string;
  created_by: string;
  business_domain: string;
  status: 'Active' | 'Archived' | 'Deprecated';
}

/**
 * Collection configuration settings for ingestion behavior.
 */
interface CollectionConfiguration {
  /** Whether to generate summaries when uploading documents */
  generateSummary: boolean;
  /** Whether to route complex data element pages through nemoretriever-parse VLM */
  useNemotronParse: boolean;
  /** Skip Pass 1 classification and unconditionally run all PDF pages through VLM */
  forceNemotronParse: boolean;
}

/**
 * Configuration for web crawl ingestion.
 */
interface CrawlConfiguration {
  startUrl: string;
  maxPages: number | null;       // null = unlimited
  maxDepth: number | null;       // null = unlimited BFS depth
  allowedUrlPrefixes: string;    // comma-separated; auto-derived from startUrl path
  extractLinkedFiles: boolean;
  useCrawlNemotronParse: boolean;
  forceCrawlNemotronParse: boolean;
}

/**
 * State interface for the new collection creation flow.
 */
interface NewCollectionState {
  collectionName: string;
  collectionNameTouched: boolean;
  selectedFiles: File[];
  fileMetadata: Record<string, Record<string, unknown>>;
  metadataSchema: UIMetadataField[];
  isLoading: boolean;
  uploadComplete: boolean;
  error: string | null;
  hasInvalidFiles: boolean;
  // Catalog metadata
  catalogMetadata: CatalogMetadata;
  // Collection configuration
  collectionConfig: CollectionConfiguration;
  // Crawl configuration
  crawlConfig: CrawlConfiguration;
  setCollectionName: (name: string) => void;
  setCollectionNameTouched: (touched: boolean) => void;
  setMetadataSchema: (schema: UIMetadataField[]) => void;
  setIsLoading: (v: boolean) => void;
  setUploadComplete: (v: boolean) => void;
  setError: (msg: string | null) => void;
  setHasInvalidFiles: (hasInvalidFiles: boolean) => void;
  // Catalog metadata setters
  setCatalogMetadata: (updates: Partial<CatalogMetadata>) => void;
  // Collection configuration setters
  setCollectionConfig: (updates: Partial<CollectionConfiguration>) => void;
  // Crawl configuration setters
  setCrawlConfig: (updates: Partial<CrawlConfiguration>) => void;
  addFiles: (files: File[]) => void;
  setFiles: (files: File[]) => void;
  removeFile: (index: number) => void;
  updateMetadataField: (filename: string, field: string, value: unknown) => void;
  reset: () => void;
}

/**
 * Zustand store for managing the new collection creation process.
 * 
 * Handles collection name, file selection, metadata schema, and upload state
 * throughout the multi-step collection creation workflow.
 * 
 * @returns Store with collection creation state and actions
 * 
 * @example
 * ```tsx
 * const { collectionName, selectedFiles, setCollectionName, addFiles } = useNewCollectionStore();
 * setCollectionName("my-collection");
 * addFiles([file1, file2]);
 * ```
 */
/**
 * Built-in fields automatically managed by the platform — present in every collection.
 * Marked isDefault=true so the delete button is hidden in FieldDisplayCard.
 */
const DEFAULT_METADATA_FIELDS: UIMetadataField[] = [
  {
    name: "filename",
    type: "string",
    required: false,
    description: "Name of the uploaded file",
    isDefault: true,
  },
  {
    name: "page_number",
    type: "integer",
    required: false,
    description: "Page number where content appears in the document (1-indexed, first page is 1)",
    isDefault: true,
  },
  {
    name: "start_time",
    type: "integer",
    required: false,
    description: "Start timestamp in milliseconds for audio or video segments",
    isDefault: true,
  },
  {
    name: "end_time",
    type: "integer",
    required: false,
    description: "End timestamp in milliseconds for audio or video segments",
    isDefault: true,
  },
];

/**
 * Standard lineage fields automatically injected by the ingestor for every document.
 * Pre-populated so users can see them and remove any that are not needed.
 */
const STANDARD_DOCUMENT_FIELDS: UIMetadataField[] = [
  {
    name: "document_type",
    type: "string",
    required: false,
    description: "Lowercase file extension of the source file: pdf, docx, md, etc.",
    max_length: 32,
  },
  {
    name: "source_uri",
    type: "string",
    required: false,
    description: "Original filename or URL the document was sourced from",
    max_length: 1024,
  },
  {
    name: "source_system",
    type: "string",
    required: false,
    description: "Origin label: sharepoint, s3, local, web_crawl, etc.",
    max_length: 128,
  },
  {
    name: "pipeline_type",
    type: "string",
    required: false,
    description: "Pipeline that processed the document: nv_ingest | nemoretriever_parse",
    max_length: 64,
  },
  {
    name: "section_path",
    type: "string",
    required: false,
    description: "H1>H2>H3 breadcrumb of the chunk position, e.g. Results > Revenue > Q4 (nemoretriever_parse only)",
    max_length: 512,
  },
  {
    name: "chunk_index",
    type: "integer",
    required: false,
    description: "0-based position of this chunk within the document",
  },
  {
    name: "total_chunks",
    type: "integer",
    required: false,
    description: "Total chunks produced from the source document",
  },
  {
    name: "page_count",
    type: "integer",
    required: false,
    description: "Number of pages in the source PDF (nemoretriever_parse only)",
  },
  {
    name: "detected_element_types",
    type: "array",
    array_type: "string",
    required: false,
    description: "Complex element types found in the document, e.g. table, chart (nemoretriever_parse only)",
  },
];

/**
 * Extra fields for collections that will hold web-crawled content.
 * Not pre-populated — added via the "Add web crawl fields" preset button.
 */
export const WEB_CRAWL_FIELDS: UIMetadataField[] = [
  {
    name: "domain",
    type: "string",
    required: false,
    description: "Netloc of the seed URL, e.g. docs.nvidia.com",
    max_length: 253,
  },
  {
    name: "crawl_depth",
    type: "integer",
    required: false,
    description: "BFS hop distance from the seed URL; 0 = seed page",
  },
  {
    name: "page_title",
    type: "string",
    required: false,
    description: "HTML <title> of the crawled page",
    max_length: 512,
  },
  {
    name: "section_h1",
    type: "string",
    required: false,
    description: "Text of the first <h1> element on the page",
    max_length: 512,
  },
  {
    name: "meta_description",
    type: "string",
    required: false,
    description: "Content of <meta name=description> or og:description",
    max_length: 512,
  },
  {
    name: "crawl_session_id",
    type: "string",
    required: false,
    description: "UUID shared by all pages from one crawl run",
    max_length: 36,
  },
  {
    name: "last_crawled_at",
    type: "datetime",
    required: false,
    description: "ISO-8601 UTC timestamp of when the page was last crawled",
  },
];

const defaultCatalogMetadata: CatalogMetadata = {
  description: '',
  tags: [],
  owner: '',
  created_by: 'current_user', // Auto-filled
  business_domain: '',
  status: 'Active',
};

const defaultCollectionConfig: CollectionConfiguration = {
  generateSummary: true,
  useNemotronParse: false,
  forceNemotronParse: false,
};

const defaultCrawlConfig: CrawlConfiguration = {
  startUrl: "",
  maxPages: 50,
  maxDepth: null,
  allowedUrlPrefixes: "",
  extractLinkedFiles: false,
  useCrawlNemotronParse: false,
  forceCrawlNemotronParse: false,
};

export const useNewCollectionStore = create<NewCollectionState>((set, get) => ({
  collectionName: "",
  collectionNameTouched: false,
  selectedFiles: [],
  fileMetadata: {},
  metadataSchema: [...DEFAULT_METADATA_FIELDS, ...STANDARD_DOCUMENT_FIELDS],
  isLoading: false,
  uploadComplete: false,
  error: null,
  hasInvalidFiles: false,
  catalogMetadata: { ...defaultCatalogMetadata },
  collectionConfig: { ...defaultCollectionConfig },
  crawlConfig: { ...defaultCrawlConfig },

  setCollectionName: (name) => set({ collectionName: name }),
  setCollectionNameTouched: (touched) => set({ collectionNameTouched: touched }),
  setIsLoading: (v) => set({ isLoading: v }),
  setUploadComplete: (v) => set({ uploadComplete: v }),
  setError: (msg) => set({ error: msg }),
  setHasInvalidFiles: (hasInvalidFiles) => set({ hasInvalidFiles }),
  setCatalogMetadata: (updates) => set((state) => ({
    catalogMetadata: { ...state.catalogMetadata, ...updates }
  })),
  setCollectionConfig: (updates) => set((state) => ({
    collectionConfig: { ...state.collectionConfig, ...updates }
  })),
  setCrawlConfig: (updates) => set((state) => ({
    crawlConfig: { ...state.crawlConfig, ...updates }
  })),

  setMetadataSchema: (schema) => {
    const { selectedFiles, fileMetadata } = get();
    const updatedMetadata: Record<string, Record<string, unknown>> = {};

    for (const file of selectedFiles) {
      const existing = fileMetadata[file.name] || {};
      updatedMetadata[file.name] = {};

      for (const field of schema) {
        updatedMetadata[file.name][field.name] = existing[field.name] ?? getDefaultValueForField(field);
      }
    }

    set({
      metadataSchema: schema,
      fileMetadata: updatedMetadata,
    });
  },

  addFiles: (files) => {
    const { selectedFiles, metadataSchema, fileMetadata } = get();
    const updated = [...selectedFiles, ...files];
    const updatedMetadata = { ...fileMetadata };

    for (const file of files) {
      if (!updatedMetadata[file.name]) {
        updatedMetadata[file.name] = {};
        for (const field of metadataSchema) {
          updatedMetadata[file.name][field.name] = getDefaultValueForField(field);
        }
      }
    }

    set({
      selectedFiles: updated,
      fileMetadata: updatedMetadata,
    });
  },

  setFiles: (files) => {
    console.log('🟢 Store: setFiles called with', files.length, 'files');
    const { metadataSchema } = get();
    const fileMetadata: Record<string, Record<string, unknown>> = {};

    for (const file of files) {
      fileMetadata[file.name] = {};
      for (const field of metadataSchema) {
        fileMetadata[file.name][field.name] = getDefaultValueForField(field);
      }
    }

    set({
      selectedFiles: files,
      fileMetadata,
    });
    console.log('🟢 Store: selectedFiles is now', get().selectedFiles.length, 'files');
  },

  removeFile: (index) => {
    const { selectedFiles, fileMetadata } = get();
    const file = selectedFiles[index];
    const updatedFiles = selectedFiles.filter((_, i) => i !== index);
    const updatedMetadata = { ...fileMetadata };
    delete updatedMetadata[file.name];

    set({
      selectedFiles: updatedFiles,
      fileMetadata: updatedMetadata,
    });
  },

  updateMetadataField: (filename, field, value) => {
    const { fileMetadata, metadataSchema } = get();
    
    // Find the field definition to ensure proper typing
    const fieldDef = metadataSchema.find(f => f.name === field);
    let processedValue = value;
    
    // Type conversion based on field definition
    if (fieldDef) {
      switch (fieldDef.type) {
        case "integer":
          if (typeof value === "string") {
            const num = parseInt(value);
            processedValue = isNaN(num) || value.trim() === "" ? null : num;
          }
          break;
        case "float":
        case "number":
          if (typeof value === "string") {
            const num = parseFloat(value);
            processedValue = isNaN(num) || value.trim() === "" ? null : num;
          }
          break;
        case "boolean":
          if (typeof value === "string") {
            processedValue = value === "true";
          }
          break;
        case "array":
          // Arrays should already be processed as arrays or JSON strings
          if (typeof value === "string" && value.startsWith("[")) {
            try {
              processedValue = JSON.parse(value);
            } catch {
              processedValue = [];
            }
          }
          break;
        case "string":
        case "datetime":
        default:
          // Keep as-is for strings and datetime
          break;
      }
    }
    
    set({
      fileMetadata: {
        ...fileMetadata,
        [filename]: {
          ...fileMetadata[filename],
          [field]: processedValue,
        },
      },
    });
  },

  reset: () =>
    set({
      collectionName: "",
      collectionNameTouched: false,
      selectedFiles: [],
      fileMetadata: {},
      metadataSchema: [...DEFAULT_METADATA_FIELDS, ...STANDARD_DOCUMENT_FIELDS],
      isLoading: false,
      uploadComplete: false,
      error: null,
      hasInvalidFiles: false,
      catalogMetadata: { ...defaultCatalogMetadata },
      collectionConfig: { ...defaultCollectionConfig },
      crawlConfig: { ...defaultCrawlConfig },
    }),
}));
