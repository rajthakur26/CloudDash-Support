# CloudDash Multi-Agent Support System

A production-ready multi-agent customer support system for **CloudDash** — a fictional cloud infrastructure monitoring SaaS platform. Built with Python, FastAPI, Google Gemini, and ChromaDB.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         REST API / Streamlit UI                      │
│                           (api/app.py / ui.py)                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Orchestrator                                │
│                      (agents/orchestrator.py)                        │
│  • Manages ConversationState (trace ID, message history, entities)   │
│  • Routes messages to the active agent                               │
│  • Drives the handover loop (max 3 handovers per turn)               │
│  • Applies input/output guardrails                                   │
└──────┬──────────────┬───────────────┬──────────────┬────────────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌────────────────┐
│   Triage   │ │  Technical   │ │ Billing  │ │  Escalation    │
│   Agent    │ │  Support     │ │  Agent   │ │    Agent       │
│            │ │  Agent       │ │          │ │                │
│ Classifies │ │ KB retrieval │ │ KB + mock│ │ Ticket gen,    │
│ intent,    │ │ step-by-step │ │ account  │ │ context pkg,   │
│ routes     │ │ resolution   │ │ lookup   │ │ human handover │
└────────────┘ └──────┬───────┘ └────┬─────┘ └────────────────┘
                      │              │
                      ▼              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG Retrieval Pipeline                           │
│                      (retrieval/retriever.py)                        │
│                                                                      │
│  KB Articles (JSON) → Chunking → Embeddings (all-MiniLM-L6-v2)      │
│        → ChromaDB (vector store) + BM25 index                       │
│        → Hybrid retrieval → Ranked chunks → KB Citations            │
└─────────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Handover Protocol                                │
│                      (handover/protocol.py)                          │
│  • Validates transitions • Packages context snapshot                 │
│  • Classifies priority • Appends JSONL audit log                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM | Google Gemini 1.5 Flash | Fast, cost-effective, generous free tier |
| Vector Store | ChromaDB (persistent) | Embedded, no infra needed, production-ready |
| Embeddings | `all-MiniLM-L6-v2` | Lightweight, good quality, runs on CPU |
| Retrieval | Hybrid: Vector + BM25 (0.7/0.3) | Better recall for keyword-heavy technical queries |
| Agent Framework | Custom orchestrator | Full control over handover logic and state management |
| Config | YAML-driven | New agents require zero orchestration code changes |
| Logging | Structured JSON | Compatible with any log aggregation pipeline |
| API | FastAPI | Async, automatic OpenAPI docs, type-safe |
| Deployment | Render free tier | Simple, free, HTTPS out of the box |

---

## Project Structure

```
clouddash-support/
├── agents/
│   ├── models.py           # Pydantic models: ConversationState, AgentResponse, HandoverPayload
│   ├── guardrails.py       # Input (injection, off-topic) + output (PII) guardrails
│   ├── llm_client.py       # Gemini API wrapper with retry logic
│   ├── base_agent.py       # Abstract base: KB retrieval + LLM chat
│   ├── triage_agent.py     # Intent classification + routing
│   ├── technical_agent.py  # Technical troubleshooting + KB resolution
│   ├── billing_agent.py    # Billing policies + plan changes + refunds
│   ├── escalation_agent.py # Human handover + ticket generation
│   └── orchestrator.py     # Central router + conversation lifecycle
├── retrieval/
│   └── retriever.py        # Chunking, ChromaDB ingestion, hybrid retrieval
├── handover/
│   └── protocol.py         # Handover validation, context packaging, audit logging
├── knowledge_base/
│   └── articles.json       # 20 KB articles across 5 categories
├── api/
│   └── app.py              # FastAPI REST endpoints
├── config/
│   ├── settings.py         # Pydantic settings (env vars)
│   ├── logging_config.py   # Structured JSON logging
│   └── agents.yaml         # Agent prompts, routing rules, guardrail config
├── tests/
│   └── test_system.py      # Unit + integration tests
├── ui.py                   # Streamlit web UI (bonus)
├── main.py                 # Uvicorn entry point
├── render.yaml             # Render deployment config
├── requirements.txt
└── .env.example
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free)

### 1. Clone / Extract the project

```bash
cd clouddash-support
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90MB). This is a one-time download.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Run the API server

```bash
python main.py
```

The API will be available at: **http://localhost:8000**

Interactive API docs: **http://localhost:8000/docs**

---

## Running the Web UI (Bonus)

In a second terminal (with the API running):

```bash
streamlit run ui.py
```

The UI will open at: **http://localhost:8501**

The sidebar includes **one-click test scenario buttons** for all 4 assessment scenarios.

---

## Running Tests

```bash
# Unit tests (no API required)
pytest tests/ -v

# All tests including integration (requires running API on localhost:8000)
pytest tests/ -v -m integration
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/conversations` | Start a new conversation |
| `POST` | `/conversations/{id}/messages` | Send a message |
| `GET` | `/conversations/{id}` | Get conversation history |
| `GET` | `/conversations` | List all conversations |
| `GET` | `/conversations/{id}/handovers` | Handover audit log |
| `GET` | `/handovers` | All handover events |

### Quick curl examples

```bash
# Start a conversation
curl -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{}'

# Send a message (replace CONV_ID)
curl -X POST http://localhost:8000/conversations/CONV_ID/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "My alerts stopped firing after updating AWS credentials. I am on the Pro plan."}'
```

---

## Test Scenarios

All 4 required scenarios can be triggered from the Streamlit UI sidebar or via the API.

### Scenario 1 — Single-Agent Resolution
```
"My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. I'm on the Pro plan."
```
**Flow:** Triage → Technical Support (KB-006 AWS integration + KB-004 alerts) → cited resolution

### Scenario 2 — Cross-Agent Handover
```
"I want to upgrade from Pro to Enterprise, but first can you check if the SSO integration issue I reported last week has been resolved?"
```
**Flow:** Triage → Technical Support (SSO: KB-007) → Billing Agent (upgrade: KB-010) with full context preserved

### Scenario 3 — Escalation to Human
```
"I've been charged twice for April. I need an immediate refund and I want to speak to a manager."
```
**Flow:** Triage → Billing Agent → Escalation Agent → ticket generated with HIGH priority

### Scenario 4 — KB Retrieval Failure
```
"Does CloudDash support integration with Datadog for cross-platform alerting?"
```
**Flow:** Technical Support searches KB → no relevant article → transparent acknowledgment → escalation offered

---

## Deployment to Render (Free Tier)

1. Push the project to a **private GitHub repository**.
2. Go to [render.com](https://render.com) → New → Web Service.
3. Connect your GitHub repo.
4. Render auto-detects `render.yaml`.
5. In the Render dashboard, add the environment variable:
   - `GEMINI_API_KEY` → your Gemini API key
6. Click **Deploy**.

Your live URL will be: `https://clouddash-support-api.onrender.com`

> **Free tier note:** The first request after inactivity may take ~30s (cold start). Subsequent requests are fast.

---

## Guardrails

### Input Guardrails
- **Prompt injection detection**: 10 patterns covering common jailbreak attempts
- **Off-topic filtering**: Blocks non-support requests (poems, images, etc.)
- **Length limit**: Max 2000 characters per message

### Output Guardrails
- **PII redaction**: Credit card numbers, SSNs, long account numbers
- **KB grounding**: Agents instructed to cite KB articles and acknowledge gaps honestly (no hallucination)

---

## Adding a New Agent

1. Create `agents/my_new_agent.py` inheriting from `BaseAgent`
2. Add agent config to `config/agents.yaml`
3. Register in `agents/orchestrator.py` `_build_registry()`
4. Add transition rules in `handover/protocol.py` `VALID_TRANSITIONS`

No other files need modification.

---

## Known Limitations

- **In-memory conversation storage**: Conversations are stored in RAM. Restarting the server clears all conversations. For production, replace with Redis or a database.
- **No authentication**: The API has no auth layer. In production, add API key or JWT middleware.
- **Gemini rate limits**: The free tier has request-per-minute limits. Under heavy load, retry logic handles this but latency increases.
- **ChromaDB on Render free tier**: The `/tmp` directory is ephemeral on Render — the KB is re-ingested on each cold start (~10-15s). For production, use a persistent disk or a hosted vector DB (Pinecone, Qdrant).
- **Single-turn handovers only**: The orchestrator processes one user message at a time and resolves handovers within that turn. Multi-turn handover chains (> 3 hops) are capped.
- **Mock account lookup**: The billing agent uses a static mock account database. Production would integrate with a real CRM/billing system.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI 0.115 |
| LLM | Google Gemini 1.5 Flash |
| Vector Store | ChromaDB 0.5 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword Search | rank-bm25 |
| Data Validation | Pydantic v2 |
| Web UI | Streamlit |
| Logging | Python structlog-style JSON |
| Testing | pytest |
| Deployment | Render |
