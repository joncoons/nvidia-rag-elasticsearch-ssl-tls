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

import { useCallback } from "react";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { MetadataField } from "../files/MetadataField";
import { Divider, Stack, Text } from "@kui/react";
import type { UIMetadataField } from "../../types/collections";

/**
 * Renders non-default metadata schema fields as global crawl defaults.
 *
 * Mirrors FileMetadataForm but operates on crawlMetadata (a flat key→value
 * dict) rather than per-file metadata.  Only shown when the active collection
 * has custom schema fields beyond the built-in defaults.
 */
export const CrawlMetadataForm = () => {
  const { metadataSchema, crawlMetadata, setCrawlMetadataField } = useNewCollectionStore();

  const handleFieldChange = useCallback((fieldName: string, value: unknown) => {
    setCrawlMetadataField(fieldName, value);
  }, [setCrawlMetadataField]);

  // Only show fields the user explicitly added — skip built-in defaults (isDefault)
  // and fields auto-populated by the crawler itself.
  const crawlerAutoFields = new Set([
    "filename", "source_uri", "crawl_depth", "page_title", "section_h1",
    "meta_description", "section_path", "heading", "source_system",
    "crawl_session_id", "last_crawled_at",
  ]);

  const customFields = metadataSchema.filter(
    (f: UIMetadataField) => !f.isDefault && !crawlerAutoFields.has(f.name)
  );

  if (customFields.length === 0) {
    return null;
  }

  return (
    <div style={{ marginTop: 'var(--spacing-density-md)' }}>
      <Divider />
      <Stack gap="density-md" style={{ paddingTop: 'var(--spacing-density-md)' }}>
        <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
          These values will be applied to every chunk ingested by this crawl.
        </Text>
        {customFields.map((field: UIMetadataField) => {
          const existingValue = crawlMetadata[field.name];
          const value = existingValue !== undefined ? existingValue : (() => {
            switch (field.type) {
              case "boolean": return false;
              case "array": return [];
              case "integer":
              case "float":
              case "number": return null;
              default: return "";
            }
          })();

          return (
            <MetadataField
              key={field.name}
              fileName="__crawl__"
              field={field}
              value={value}
              onChange={handleFieldChange}
            />
          );
        })}
      </Stack>
    </div>
  );
};
