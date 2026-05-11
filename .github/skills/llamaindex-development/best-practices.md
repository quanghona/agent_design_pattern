# LlamaIndex Best Practices

A comprehensive guide of best practices for building production-ready LlamaIndex applications, synthesized from industry leaders, technical experts, and real-world implementations.

## Table of Contents

- [Core Architecture](#core-architecture)
- [Data Ingestion](#data-ingestion)
- [Indexing Strategies](#indexing-strategies)
- [Retrieval Optimization](#retrieval-optimization)
- [Query Engines](#query-engines)
- [Agents & Workflows](#agents--workflows)
- [Production Deployment](#production-deployment)
- [Observability](#observability)
- [Testing](#testing)

---

## Core Architecture

### 1. LlamaIndex Core Abstractions

LlamaIndex is a data framework for LLM applications. Keep these core abstractions in mind:

- **Data Connectors (LlamaHub):** Used to ingest data from various sources
- **Documents & Nodes:** The fundamental units of data. `Document` is the raw text/metadata, while `Node` is a chunk of a document with relationship information
- **Indices:** Data structures that allow for efficient retrieval (e.g., `VectorStoreIndex`, `SummaryIndex`, `KnowledgeGraphIndex`)
- **Retriever:** Takes a query and returns a list of `Node` objects
- **Query Engine:** Takes a query and returns a response (combining retriever and response synthesizer)
- **Post-processors:** Modules used to refine retrieved nodes (e.g., `SimilarityPostprocessor`, `Re-ranking`)

### 2. Decouple Ingestion from Indexing

Use an event bus (e.g., Redis Streams) to allow burst arrivals without backpressure:

```
Document Source → Redis Stream → Embedding Worker → Redis Stream → Indexing Worker → Vector Store
```

This enables:
- Idempotent processing (detect duplicates by document hash)
- Horizontal scaling of workers
- No backpressure on document sources

### 3. Separate Retrieval and Generation

Split into distinct microservices for independent scaling:

```
┌─────────────────────────────────────────────┐
│           Query Router                       │
│  - Intent classification                     │
│  - Fallback: BM25 + template answer          │
└──────┬──────────────────┬───────────────────┘
       │                  │
       ▼                  ▼
┌─────────────┐   ┌──────────────────┐
│ Retriever   │   │ Generator        │
│ - Vector DB │   │ - LLM (OpenAI)   │
│ - Embedding │   │ - Token limiter  │
│   cache     │   │ - Circuit breaker│
└─────────────┘   └──────────────────┘
```

---

## Data Ingestion

### 1. Data Connectors

Use LlamaHub connectors for various data sources:

```python
from llama_index.core import SimpleDirectoryReader, Document

# Local files
documents = SimpleDirectoryReader("./data").load_data()

# Web scraping
from llama_index.readers.web import SimpleWebPageReader
documents = SimpleWebPageReader(html_to_markdown=True).load_data(
    ["https://example.com/page1", "https://example.com/page2"]
)

# Database
from llama_index.readers.database import DatabaseReader
reader = DatabaseReader(uri="postgresql://...")
documents = reader.load_data(query="SELECT * FROM articles")
```

### 2. Document Processing

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

# Sentence-based splitting (good for most use cases)
sentence_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = sentence_parser.get_nodes_from_documents(documents)

# Token-based splitting (good for code or technical docs)
token_parser = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
nodes = token_parser.get_nodes_from_documents(documents)
```

### 3. Idempotent Ingestion

```python
import hashlib

def get_document_hash(doc: Document) -> str:
    return hashlib.sha256(doc.text.encode()).hexdigest()

# Store hashes to detect duplicates
processed_hashes = set()

for doc in documents:
    doc_hash = get_document_hash(doc)
    if doc_hash not in processed_hashes:
        # Process document
        processed_hashes.add(doc_hash)
```

---

## Indexing Strategies

### 1. Index Type Selection Guide

| Index Type | Use Case | Example |
|------------|----------|---------|
| `VectorStoreIndex` | Semantic search, general RAG | Default choice for most applications |
| `SummaryIndex` | Summarization, overview queries | "Give me an overview of this document" |
| `KnowledgeGraphIndex` | Relationship queries, entity tracking | "What is the relationship between X and Y?" |
| `TreeIndex` | Hierarchical navigation, top-down queries | "Summarize section X" |
| `KeywordTableIndex` | Keyword-based retrieval | "Find all mentions of term X" |

### 2. Vector Store Selection

```python
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Weaviate (good for production)
vector_store = WeaviateVectorStore(
    weaviate_client=get_client(),
    index_name="DocChunk",
    distance="cosine"  # Well-tested with OpenAI embeddings
)

# Pinecone (good for managed service)
from llama_index.vector_stores.pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    measurement_type="cosine"
)

# Chroma (good for development/small deployments)
from llama_index.vector_stores.chroma import ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=collection)
```

### 3. Embedding Model Selection

```python
# OpenAI embeddings (good quality, production-ready)
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    embed_batch_size=10
)

# Local embeddings (free, no rate limits)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pin your embedding model version in production
# Changes require re-indexing
```

---

## Retrieval Optimization

### 1. Decouple Retrieval vs. Synthesis Chunks

The optimal chunk representation for retrieval may differ from the optimal chunk for synthesis:

**Option A: Embed document summaries linked to chunks**
```python
from llama_index.core.indices.document_summary import DocumentSummaryIndex

# Embed summaries at document level, then retrieve chunks
index = DocumentSummaryIndex.from_documents(documents)
retriever = index.as_retriever()
```

**Option B: Embed sentences linked to windows**
```python
from llama_index.core.node_parser import SentenceWindowNodeParser

# Embed individual sentences, retrieve surrounding context
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)
nodes = node_parser.get_nodes_from_documents(documents)
```

### 2. Structured Retrieval for Large Document Sets

**Metadata Filters + Auto Retrieval**
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import MetadataFilter

# Tag documents with metadata
documents[0].metadata["document_type"] = "policy"
documents[0].metadata["year"] = "2024"

# Filter during retrieval
retriever = index.as_retriever(
    filters=[MetadataFilter(key="document_type", value="policy")]
)
```

**Recursive Retrieval (Summaries → Raw Chunks)**
```python
from llama_index.core.retrievers import RecursiveRetriever

# Fetch at document level first, then chunk level
retriever = RecursiveRetriever(
    root_id="doc_summary",
    retriever_dict={"doc_summary": summary_retriever, "chunk": chunk_retriever}
)
```

### 3. Post-processors

```python
from llama_index.core.postprocessor import (
    SimilarityPostprocessor,
    SentenceTransformerRerank,
    MetadataReplacementPostProcessor
)

# Filter by similarity threshold
similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)

# Re-rank retrieved nodes
reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5
)

# Combine post-processors
node_postprocessors = [
    SimilarityPostprocessor(similarity_cutoff=0.75),
    SentenceTransformerRerank(top_n=5)
]
```

### 4. Optimize Context Embeddings

Fine-tune embeddings over your specific data corpus:

```python
# Use label-free fine-tuning for your domain
# See: https://developers.llamaindex.ai/python/examples/finetuning/embeddings/finetune_embedding
```

---

## Query Engines

### 1. Query Engine Types

| Query Engine | Use Case | Example |
|--------------|----------|---------|
| `RetrieverQueryEngine` | Standard RAG | Default choice |
| `RouterQueryEngine` | Task-specific routing | QA vs. summarization |
| `SubQuestionQueryEngine` | Multi-part questions | "Compare X and Y" |
| `CondenseQuestionChatEngine` | Conversational QA | Multi-turn chat |
| `CondensePlusContextChatEngine` | Chat with context | Chat with RAG |

### 2. Router Query Engine

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

# Define sub-query engines
qa_engine = index.as_query_engine()
summary_engine = index.as_query_engine(response_mode="tree_summarize")

# Create router
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_dict={
        "question answering": qa_engine,
        "summarization": summary_engine
    }
)
```

### 3. Sub-Question Query Engine

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

# Handles multi-part questions automatically
query_engine = SubQuestionQueryEngine.from_defaults(
    query_components=[index.as_query_engine()]
)

# Example: "Compare the Q1 and Q2 revenue"
# → Creates sub-questions for Q1 and Q2 separately
# → Combines answers
```

### 4. Chat Engines

```python
from llama_index.core.chat_engine import (
    CondenseQuestionChatEngine,
    CondensePlusContextChatEngine
)

# Simple conversational QA
chat_engine = CondenseQuestionChatEngine.from_defaults(
    index=index,
    llm=llm
)

# Chat with RAG context
chat_engine = CondensePlusContextChatEngine.from_defaults(
    index=index,
    llm=llm
)

# Use
response = chat_engine.chat("What is the refund policy?")
response = chat_engine.chat("Can you explain more?")
```

---

## Agents & Workflows

### 1. LlamaIndex Agents

```python
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import QueryEngineTool

# Define tools
query_tool = QueryEngineTool.from_defaults(
    query_engine=index.as_query_engine(),
    description="Useful for answering questions about the documents"
)

# Create agent
agent = AgentRunner.from_tools(
    [query_tool],
    llm=llm,
    verbose=True
)

# Use
response = agent.run("What does the document say about AI?")
```

### 2. AgentWorkflow (Advanced)

```python
from llama_index.core.workflow import Workflow, Context, Event, StartEvent, StopEvent
from llama_index.core.tools import FunctionTool

class RAGWorkflow(Workflow):
    @Step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> Event:
        query = ev.get("query")
        nodes = await self.retriever.aretrieve(query)
        await ctx.set("nodes", nodes)
        return RetrieveEvent(nodes=nodes)

    @Step
    async def synthesize(self, ctx: Context, ev: RetrieveEvent) -> StopEvent:
        nodes = ev.nodes
        response = await self.synthesizer.asynthesize(query="", nodes=nodes)
        return StopEvent(result=response)
```

### 3. Tool Definition

```python
from llama_index.core.tools import FunctionTool

def search_database(query: str) -> str:
    """Search the database for relevant information."""
    # Implementation
    return results

tool = FunctionTool.from_defaults(fn=search_database)
```

---

## Production Deployment

### 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway (Kong/Envoy)                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │     Query Router Service      │
              │  - Intent classification        │
              │  - Fallback: BM25 + template  │
              └───────┬───────────────┬───────┘
                      │               │
          ┌───────────┴───┐   ┌───────┴──────────┐
          │ Retriever     │   │ Generator        │
          │ Service       │   │ Service          │
          │ - Weaviate    │   │ - OpenAI/Ollama  │
          │ - Embedding   │   │ - Token limiter  │
          │   cache       │   │ - Circuit breaker│
          └───────────────┘   └──────────────────┘
```

### 2. Latency Budget

| Component | p50 Latency | p95 Latency |
|-----------|-------------|-------------|
| Retriever | 42ms | 86ms |
| Generator | 1.2s | 3.4s |
| Total | ~1.3s | ~3.5s |

### 3. Scaling Playbook

| Traffic Tier | Query Router | Retriever | Generator | Cost/Month |
|--------------|-------------|-----------|-----------|------------|
| Low (500 QPS) | 2 pods | 3 pods | 3 pods | $1,800 |
| Medium (5K QPS) | 10 pods | 15 pods | 20 pods | $8,500 |
| High (50K QPS) | 50 pods | 80 pods | 100 pods | $42,000 |

### 4. Circuit Breaker Pattern

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=3, reset_timeout=30):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0

    def call(self, func, *args):
        if self.failures >= self.threshold:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.failures = 0  # half-open
            else:
                return {"fallback": True, "data": "High load. Please retry."}
        try:
            result = func(*args)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            return {"fallback": True, "data": "Error occurred. Please retry."}
```

### 5. Token Budget Management

```python
def token_limiter(context: str, max_tokens: int = 4000) -> str:
    """Limit context to max_tokens, keeping first and last portions."""
    tokens = llm.tokenize(context)
    if len(tokens) > max_tokens:
        # Keep first and last portions (middle loss strategy)
        half = max_tokens // 2
        return llm.untokenize(tokens[:half]) + "\n...\n" + llm.untokenize(tokens[-half:])
    return context
```

### 6. Fallback Strategies

```python
# When LLM is slow, serve cached or template responses
def query_with_fallback(query: str, timeout: float = 10.0):
    try:
        response = query_engine.query(query, timeout=timeout)
        return response
    except TimeoutError:
        # Serve cached response or template
        return get_cached_or_template_response(query)
```

---

## Observability

### 1. OpenTelemetry Integration

Instrument every public method with OpenTelemetry spans:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("llamaindex_query")
def instrumented_query(query: str):
    response = query_engine.query(query)
    return response
```

### 2. Key Metrics to Track

- Retrieval latency (p50, p95, p99)
- Number of chunks retrieved
- LLM token usage (input, output)
- Error rates and types
- Cache hit rates
- Circuit breaker trips

### 3. Logging

```python
import logging

logger = logging.getLogger("llamaindex")
logger.setLevel(logging.INFO)

# Log retrieval details
logger.info(f"Retrieved {len(nodes)} chunks for query: {query}")
logger.info(f"Similarity scores: {[n.score for n in nodes]}")
```

---

## Testing

### 1. Mocking the LLM

```python
from unittest.mock import MagicMock
from llama_index.core.llms import MockLLM

# Use MockLLM for deterministic testing
mock_llm = MockLLM()
index = VectorStoreIndex.from_documents(documents, llm=mock_llm)
```

### 2. Testing Retrieval

```python
def test_retriever():
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve("test query")

    assert len(nodes) == 5
    assert all(node.score > 0 for node in nodes)
```

### 3. Testing Query Engine

```python
def test_query_engine():
    query_engine = index.as_query_engine()
    response = query_engine.query("test query")

    assert response is not None
    assert len(str(response)) > 0
```

### 4. Testing Post-processors

```python
def test_reranker():
    reranker = SentenceTransformerRerank(top_n=3)
    nodes = reranker.postprocess_nodes(
        nodes=test_nodes,
        query_str="test query"
    )

    assert len(nodes) == 3
```

### 5. Integration Testing

```python
def test_full_rag_pipeline():
    # Setup
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    # Test
    response = query_engine.query("What is the main topic?")

    # Assert
    assert "main topic" in str(response).lower()
```

---

## Common Pitfalls

### 1. Chunk Size Too Large

Large chunks lead to "lost in the middle" problems. Use sentence-based splitting with appropriate chunk sizes (512-1024 tokens).

### 2. No Similarity Cutoff

Without a similarity cutoff, irrelevant chunks may be included. Always set a `similarity_cutoff` (e.g., 0.75).

### 3. Not Using Post-processors

Post-processors (similarity filter, reranker) significantly improve retrieval quality. Always use them in production.

### 4. Ignoring Metadata

Metadata enables structured retrieval. Tag documents with relevant metadata (type, year, source, etc.).

### 5. Not Pinning Model Versions

Changes to embedding or LLM versions require re-indexing. Pin versions in production.

### 6. No Fallback Strategy

Always implement fallbacks for when the LLM is slow or unavailable (cached responses, templates).

---

## References

- [LlamaIndex Production RAG Guide](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)
- [LlamaIndex Production Architecture - MarkAI](https://markaicode.com/architecture/llm-architecture-with-llamaindex/)
- [LlamaIndex Complete Guide - Galileo AI](https://galileo.ai/blog/llamaindex-complete-guide-rag-data-workflows-llms)
- [LlamaIndex Official Documentation](https://docs.llamaindex.ai/)
- [Building Production-Ready RAG Systems - Medium](https://medium.com/@meeran03/building-production-ready-rag-systems-best-practices-and-latest-tools-581cae9518e7)
- [RAG Techniques Handbook - GitHub](https://github.com/sosanzma/rag-techniques-handbook)
