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

import { useState } from "react";
import { Text, Flex, Stack, Panel, Switch } from "@kui/react";
import { Settings, ChevronDown } from "lucide-react";

interface CollectionConfigurationPanelProps {
  generateSummary: boolean;
  onGenerateSummaryChange: (value: boolean) => void;
  useNemotronParse: boolean;
  onUseNemotronParseChange: (value: boolean) => void;
  forceNemotronParse: boolean;
  onForceNemotronParseChange: (value: boolean) => void;
}

/**
 * Expandable panel for collection-specific configuration settings.
 *
 * Displays settings that control how documents are processed when uploaded
 * to this collection. Follows the same pattern as CatalogMetadataSection.
 */
export function CollectionConfigurationPanel({
  generateSummary,
  onGenerateSummaryChange,
  useNemotronParse,
  onUseNemotronParseChange,
  forceNemotronParse,
  onForceNemotronParseChange,
}: CollectionConfigurationPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <Panel
      slotHeading={
        <Flex
          align="center"
          justify="between"
          style={{ width: '100%', cursor: 'pointer' }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span>Collection Configuration</span>
          <ChevronDown
            size={16}
            style={{
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease'
            }}
          />
        </Flex>
      }
      slotIcon={<Settings size={20} />}
    >
      <Text kind="body/bold/md">
        Settings that control how documents are processed during ingestion.
      </Text>

      {isExpanded && (
        <Stack gap="density-md" style={{ marginTop: 'var(--spacing-density-lg)' }}>
          {/* Summarization Toggle */}
          <Stack gap="density-xs">
            <Switch
              checked={generateSummary}
              onCheckedChange={onGenerateSummaryChange}
              size="medium"
              slotLabel="Document Summarization"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Automatically generate summaries for uploaded documents. This feature could increase costs and processing time.
            </Text>
          </Stack>

          {/* Nemotron Parse Toggles */}
          <Stack gap="density-xs">
            <Switch
              checked={useNemotronParse || forceNemotronParse}
              onCheckedChange={(checked: boolean) => {
                onUseNemotronParseChange(checked);
                if (!checked) onForceNemotronParseChange(false);
              }}
              size="medium"
              slotLabel="Nemotron Parse (complex data elements)"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Route PDFs with tables, charts, or infographics through nemoretriever-parse VLM for high-quality markdown extraction.
            </Text>

            <Switch
              checked={forceNemotronParse}
              onCheckedChange={(checked: boolean) => {
                onForceNemotronParseChange(checked);
                if (checked) onUseNemotronParseChange(true);
              }}
              size="medium"
              slotLabel="Nemotron Parse (all pages)"
            />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Skip classification and run all PDF pages through VLM. Recommended for dense financial or technical documents.
            </Text>
          </Stack>
        </Stack>
      )}
    </Panel>
  );
}
