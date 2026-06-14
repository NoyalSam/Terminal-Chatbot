# RAG GenAI Project

Production-style Retrieval Augmented Generation system built with LangChain, FAISS, and Google / OpenAI LLMs.

---

## Architecture

```
DocumentLoader
     │  (PDF / TXT)
     ▼
SemanticChunker          ← topic-aware splitting (no fixed chunk size)
     │
     ▼
EmbeddingFactory         ← Google Generative AI  OR  OpenAI
     │
     ▼
VectorStoreManager       ← FAISS index + similarity retriever
     │
     ├──▶  RAGChain      ← stuff-documents chain + system prompt (anti-hallucination)
     │         │
     │         └──▶  MemoryManager  (InMemoryChatMessageHistory, sliding window)
     │
     └──▶  AgentExecutor ← tool-calling agent
               ├── search_documents  (RAG retrieval)
               └── calculator        (safe arithmetic)
```

---

## Project structure

```
rag-genai-project/
├── app.py              # All layers + CLI entry point
├── config.py           # Centralised configuration (env-driven)
├── requirements.txt
├── .env.example        # Copy to .env and fill in your keys
└── data/
    └── sample_document.pdf   # Drop your document here
```

---

## Quick start

```bash
# 1. Clone / copy the project
cd rag-genai-project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY or OPENAI_API_KEY

# 5. Add a document
# Drop a PDF into data/  (default name: sample_document.pdf)
# OR use the load command at runtime

# 6. Run
python app.py
```

---

## CLI commands

| Command | Description |
|---|---|
| `ask <question>` | Answer via RAG chain with memory |
| `agent <question>` | Answer via tool-calling agent with memory |
| `load <path>` | Load / replace the active document |
| `clear` | Clear conversation memory for current session |
| `quit` | Exit |

Bare input (no prefix) is routed to the agent automatically.

---

## Using as a library

```python
from app import RAGApplication

app = RAGApplication()
app.load_document("data/my_report.pdf")

# RAG chain (answers strictly from context)
answer = app.ask("What were the key findings?")
print(answer)

# Agent (picks the right tool automatically)
answer = app.ask_agent("Summarise section 3 and calculate 1024 * 0.07")
print(answer)

# Multi-turn with explicit session IDs
answer = app.ask("Who wrote this?", session_id="user-42")
answer = app.ask("What else did they publish?", session_id="user-42")

# Clear memory when done
app.clear_memory("user-42")
```

---

## Switching providers

Edit `.env`:

```
# Use OpenAI for both layers
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Or mix: Google embeddings + OpenAI LLM
EMBEDDING_PROVIDER=google
LLM_PROVIDER=openai
```

---

## Key design decisions

**Semantic chunking** — `SemanticChunker` groups sentences by embedding similarity instead of splitting on fixed token counts. This keeps semantically related content together, improving retrieval precision.

**Anti-hallucination system prompt** — The RAG chain's system prompt explicitly forbids the model from using pre-training knowledge that isn't present in the retrieved context. If the context lacks the answer, the model says so.

**InMemoryChatMessageHistory with sliding window** — Each session maintains a rolling window of the last N message pairs, bounding context size while preserving conversational coherence.

**Tool-calling agent** — Rather than always running RAG, the agent decides whether a question needs document retrieval (`search_documents`), arithmetic (`calculator`), or neither. This avoids unnecessary API calls and lets the system handle mixed queries naturally.

**Provider abstraction** — `EmbeddingFactory` and `LLMFactory` hide provider-specific imports behind a single `.build()` call, making it trivial to swap Google ↔ OpenAI via environment variables.
