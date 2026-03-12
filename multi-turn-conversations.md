# Multi-Turn Conversation Support

**Baseline**: [NVIDIA-AI-Blueprints/rag](https://github.com/NVIDIA-AI-Blueprints/rag)

The baseline blueprint ships with the full multi-turn conversation architecture already implemented — the `messages` array, history splitting, truncation formula, and query rewriting are all present upstream. However, conversation history is **off by default** (`CONVERSATION_HISTORY=0`). The local deployment activates it via a single env var and adds Tavily web search as an additional context source.

---

## How Multi-Turn Works End-to-End

```
Frontend (Zustand store)
    Full messages[] array accumulated across turns
    ↓
POST /generate  { messages: [{role, content}, ...] }
    ↓
response_generator.py: prepare_llm_request()
    Splits: query = last user message
            chat_history = all prior turns (except last user)
    ↓
main.py: CONVERSATION_HISTORY truncation
    chat_history = chat_history[-(N_turns * 2):]
    ↓
Optional: query rewriting (enable_query_rewriting=True)
    Standalone query reformulated for VDB retrieval
    ↓
VDB retrieval (using rewritten or original query)
    ↓
_handle_prompt_processing()
    Base system prompt from prompt.yaml
    Prior turns formatted as text block appended as user message
    ↓
LLM (OpenAI-compatible streaming)
    system: <prompt.yaml system prompt>
    user:   <prompt template with {context}>
    user:   "Conversation history:\nUser: ...\nAssistant: ..."
    user:   "Query: {question}\n\nAnswer: "
```

---

## Section 1 — What the Baseline Already Provides

No code changes are needed for these — they work in the baseline as-is.

### 1.1 `Prompt` Model — Full Message Array

**File**: `src/nvidia_rag/rag_server/server.py`

The generate endpoint accepts a full conversation array, not a single query string:

```python
class Prompt(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="A list of messages comprising the conversation so far.",
        max_items=50000,
    )
```

Each `Message` has `role` (`user` | `assistant` | `system`) and `content` (text or multimodal list). There is no per-turn limit in the schema — truncation happens server-side via `CONVERSATION_HISTORY`.

### 1.2 `prepare_llm_request()` — History Splitting

**File**: `src/nvidia_rag/rag_server/response_generator.py`

Called before every generate request. Splits the messages array into the current query and prior history:

```python
def prepare_llm_request(messages: list[dict]) -> tuple[str, list[dict]]:
    # Strip empty-content assistant messages (in-flight streaming placeholders)
    chat_history = [
        msg for msg in messages
        if not (msg["role"] == "assistant" and _is_empty_content(msg["content"]))
    ]
    # Extract last user message as the current query
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i]["role"] == "user":
            query = chat_history[i]["content"]
            chat_history = chat_history[:i] + chat_history[i+1:]
            break
    return query, chat_history
```

Result: `query` = the current user question; `chat_history` = all prior turns (user + assistant) in order.

### 1.3 `CONVERSATION_HISTORY` Truncation

**File**: `src/nvidia_rag/rag_server/main.py` — applied in `_rag_chain()`, `_llm_chain()`, and `_vlm_direct_chain()`

```python
conversation_history_count = int(os.environ.get("CONVERSATION_HISTORY", 0))
if conversation_history_count == 0:
    chat_history = []
else:
    history_count = conversation_history_count * 2 * -1
    chat_history = chat_history[history_count:]
```

`conversation_history_count * 2` captures the last N complete turns (each turn = 1 user message + 1 assistant message = 2 entries). `CONVERSATION_HISTORY=5` → keep the last 10 messages.

**Default is 0 — history is completely disabled in the baseline unless this env var is set.**

### 1.4 History Injection into the LLM Prompt

**File**: `src/nvidia_rag/rag_server/main.py` — `_rag_chain()` and `_llm_chain()`

Prior turns are serialised as a text block and injected as an additional `user` message before the current query:

```python
if conversation_history:
    formatted_history = "\n".join([
        f"{role.title()}: {content}"
        for role, content in conversation_history
    ])
    message += [("user", f"Conversation history:\n{formatted_history}")]

message += [("user", "Query: {question}\n\nAnswer: ")]
```

Note: history is a formatted text dump, not alternating `user`/`assistant` chat turns. The LLM receives the full prompt structure:

```
system:  <prompt.yaml system prompt>
user:    <RAG template with {context}>
user:    "Conversation history:\nUser: ...\nAssistant: ..."
user:    "Query: {question}\n\nAnswer: "
```

**VLM path is different** — for VLM models history is passed as native OpenAI-format message dicts (`[{role, content}, ...]`), not a text dump. This is appropriate since VLMs expect structured message lists for image interleaving.

### 1.5 Query Rewriting for Retrieval Disambiguation

**File**: `src/nvidia_rag/rag_server/main.py`, `_rag_chain()`

When `enable_query_rewriting=True` and `CONVERSATION_HISTORY > 0`, a preliminary LLM call reformulates the current query to be self-contained before it is used for VDB retrieval:

```python
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
```

Example: "What about the second one?" → "What are the features of the second GPU model mentioned?" The rewritten query is used for VDB retrieval only — the original `{question}` is still passed to the final LLM prompt. If `CONVERSATION_HISTORY=0`, query rewriting is skipped with a warning since it is meaningless without history.

### 1.6 System Prompt via `prompt.yaml`

**File**: `src/nvidia_rag/rag_server/main.py` — `_handle_prompt_processing()`

The system prompt is loaded from `PROMPT_CONFIG_FILE=/prompt.yaml` at startup. `chat_template` is used for direct LLM chains; `rag_template` for RAG chains. Any message with `role="system"` in the chat history is appended to the base system prompt rather than treated as a separate turn — this allows callers to inject per-session system instructions.

### 1.7 Frontend History Accumulation

**File**: `frontend/src/store/useChatStore.ts`

A Zustand store maintains `messages: ChatMessage[]` in memory. Every new user submission appends to this array; the full array is sent on every request:

```typescript
// useMessageSubmit.ts
const currentMessages = [...messages, userMessage];   // full history + new turn
const request = createRequest(currentMessages);        // serialise all turns
await sendMessage({ request, assistantId });
```

The assistant's empty streaming placeholder added to the store at submit time is excluded from the sent array (captured before it is added). History persists for the browser session only — page reload clears all turns.

---

## Section 2 — Changes Required to Activate Multi-Turn

Only one change is required to go from the baseline's default (history disabled) to an active multi-turn deployment:

### 2.1 Set `CONVERSATION_HISTORY` in Helm Values

**File**: `deploy/helm/values-local.yaml`

```yaml
# rag-server envVars
CONVERSATION_HISTORY: "10"
```

This activates the last 10 complete turns (20 messages: 10 user + 10 assistant). The formula inside `_rag_chain()` / `_llm_chain()` / `_vlm_direct_chain()` uses `conversation_history_count * 2 * -1` to slice the history array.

**Choosing a value**: Each turn adds approximately 200–800 tokens to the LLM prompt depending on response length. With `nim-llm` configured at `NIM_MAX_MODEL_LEN=131072` (128k context) and typical RAG responses of ~300 tokens, 10 turns adds ~3–8k tokens — well within budget. Increase conservatively; very large history values combined with large retrieved contexts can push total prompt length toward the model's context limit.

No other backend code changes are needed. All other multi-turn machinery (splitting, truncation, injection, query rewriting) is already present in the baseline.

---

## Section 3 — Local Customisations

These changes extend the baseline multi-turn capability. They are present in the local overlay but not in the upstream blueprint.

### 3.1 Tavily Web Search as Additional Context

**Files**: `src/nvidia_rag/rag_server/server.py`, `src/nvidia_rag/utils/tavily_search.py`, `src/nvidia_rag/utils/configuration.py`

An `enable_tavily_search` flag added to the `Prompt` model routes the current query to the Tavily web search API and appends results to the VDB-retrieved context before the LLM call. Tavily results are additive — they do not replace VDB context and are not stored in the conversation history. Each turn makes a fresh Tavily call if the flag is set; there is no cross-turn caching of web results.

```python
# server.py — Prompt model addition
enable_tavily_search: bool = Field(
    default=False,
    description="When True, extend retrieval with live Tavily web search results.",
)
```

```typescript
// frontend/src/types/requests.ts
enable_tavily_search?: boolean;
```

Requires: `APP_TAVILY_ENABLED=true` + `TAVILY_API_KEY` secret in the deployment. Controlled per-message via the "Extend Search Online" toggle in `MessageInput.tsx`.

### 3.2 `page_number` Citation Fix

**File**: `src/nvidia_rag/rag_server/response_generator.py`, lines ~807 and ~819

Web-crawled HTML chunks store `page_number=None` in ES (HTML has no concept of PDF page numbers). The baseline code performs arithmetic on this value without guarding against `None`, causing a `TypeError` when citations are assembled for responses that include web content in the retrieved context.

```python
# Both branches in the citation assembly loop need this guard:
page_number = chunk.get("page_number") or 0
```

Without this fix, any multi-turn conversation that retrieves HTML chunks from the web-crawl collection raises an exception mid-stream and silently truncates the streamed response.

### 3.3 Thinking Token Support (Nemotron)

**File**: `src/nvidia_rag/rag_server/server.py`

Two fields added to `Prompt` for Nemotron extended-thinking mode:

```python
min_thinking_tokens: int | None = Field(default=None)
max_thinking_tokens: int | None = Field(default=None)
```

When set, `_handle_prompt_processing()` injects `"detailed thinking on"` into the system prompt and passes the token budget to the LLM. This is a generation-quality improvement for complex multi-turn reasoning tasks, not a structural change to conversation handling.

---

## Summary

| Aspect | Baseline | Local Deployment |
|---|---|---|
| `messages` array in API | Present — `list[Message]`, max 50000 | Identical + `enable_tavily_search` field |
| History splitting (`prepare_llm_request`) | Present | Identical |
| `CONVERSATION_HISTORY` default | `0` — history **disabled** | `10` — last 10 turns active |
| Truncation formula | `N * 2 * -1` slice | Identical |
| LLM prompt injection | Text dump in `user` message | Identical |
| VLM history injection | Native message list | Identical |
| Query rewriting | Present (requires `CONVERSATION_HISTORY > 0`) | Identical |
| System prompt source | `prompt.yaml` via `PROMPT_CONFIG_FILE` | Identical |
| Frontend history store | Full array sent per request | Identical |
| Tavily web search | Not present | Added — appended to VDB context per turn |
| `page_number` citation fix | Crashes on `None` from HTML chunks | Fixed with `or 0` guard |
| Thinking token fields | Not present | Added to `Prompt` model |

**Minimum change to activate multi-turn from baseline**: set `CONVERSATION_HISTORY=<N>` in the rag-server environment.

---

## File Reference

| File | Role |
|---|---|
| `src/nvidia_rag/rag_server/server.py` | `Prompt` model; `/generate` endpoint; `enable_tavily_search`, thinking token fields |
| `src/nvidia_rag/rag_server/response_generator.py` | `prepare_llm_request()` history split; `page_number or 0` citation fix |
| `src/nvidia_rag/rag_server/main.py` | `CONVERSATION_HISTORY` truncation; `_handle_prompt_processing()`; query rewriting; final LLM prompt assembly |
| `src/nvidia_rag/utils/tavily_search.py` | Async Tavily web search |
| `src/nvidia_rag/utils/configuration.py` | `TavilyConfig`; `APP_TAVILY_ENABLED` |
| `deploy/helm/values-local.yaml` | `CONVERSATION_HISTORY: "10"`; `APP_TAVILY_ENABLED: "true"` |
| `frontend/src/store/useChatStore.ts` | Zustand messages array |
| `frontend/src/hooks/useMessageSubmit.ts` | `createRequest()` — full history serialised per turn |
| `frontend/src/types/requests.ts` | `GenerateRequest` type with `enable_tavily_search` |
