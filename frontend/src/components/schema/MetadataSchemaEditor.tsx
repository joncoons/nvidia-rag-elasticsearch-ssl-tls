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

// src/components/MetadataSchemaEditor.tsx

import { useCallback } from "react";
import { useSchemaEditor } from "../../hooks/useSchemaEditor";
import { FieldsList } from "./FieldsList";
import { NewFieldForm } from "./NewFieldForm";
import { Panel, Text, Button } from "@kui/react";
import { useNewCollectionStore, WEB_CRAWL_FIELDS } from "../../store/useNewCollectionStore";

// Export all schema editor components for external use
export { FieldEditForm } from "./FieldEditForm";
export { FieldDisplayCard } from "./FieldDisplayCard";
export { FieldsList } from "./FieldsList";
export { NewFieldForm } from "./NewFieldForm";

const SchemaIcon = () => (
  <svg 
    className="w-5 h-5 text-[var(--nv-green)]" 
    fill="none" 
    stroke="currentColor" 
    viewBox="0 0 24 24"
    data-testid="schema-icon"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
  </svg>
);

const WebCrawlPresetIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
      d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const SchemaContent = () => {
  const { metadataSchema, setMetadataSchema } = useNewCollectionStore();

  const hasWebCrawlFields = WEB_CRAWL_FIELDS.every(
    (wf) => metadataSchema.some((f) => f.name === wf.name)
  );

  const handleAddWebCrawlFields = useCallback(() => {
    const existing = new Set(metadataSchema.map((f) => f.name));
    const toAdd = WEB_CRAWL_FIELDS.filter((f) => !existing.has(f.name));
    setMetadataSchema([...metadataSchema, ...toAdd]);
  }, [metadataSchema, setMetadataSchema]);

  return (
    <>
      <Text kind="body/bold/md">Define metadata fields for this collection.</Text>

      {!hasWebCrawlFields && (
        <div style={{ marginBottom: "8px" }}>
          <Button
            kind="secondary"
            size="small"
            onClick={handleAddWebCrawlFields}
          >
            <WebCrawlPresetIcon />
            &nbsp;Add web crawl fields
          </Button>
        </div>
      )}

      <FieldsList />
      {<NewFieldForm />}
    </>
  );
}

export default function MetadataSchemaEditor() {
  const { showSchemaEditor } = useSchemaEditor();
  
  return (
    <Panel
      slotHeading="Metadata Schema"
      slotIcon={<SchemaIcon />}
    >
      {showSchemaEditor && <SchemaContent />}
    </Panel>
  );
}
