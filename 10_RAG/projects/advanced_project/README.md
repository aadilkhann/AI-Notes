# Advanced Project: Production RAG System

## Overview

Build a **production-grade** Retrieval-Augmented Generation system with:
- Multi-modal document ingestion (PDF, web, images)
- Hybrid search (dense + sparse + reranking)
- Streaming responses with citations
- Observability & monitoring
- Prompt caching & optimization
- Evaluation framework

**Goal**: Deploy a RAG system that outperforms GPT-4 on domain-specific questions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ingestion Pipeline                           │
│                                                                  │
│  Documents → Parser → Chunker → Embeddings → Vector DB          │
│  (PDF/Web)   (Unstructured)  (Recursive)  (OpenAI)  (Weaviate/  │
│                                                        Qdrant)    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Query Pipeline                              │
│                                                                  │
│  User Query → Query Transform → Hybrid Search → Reranking →     │
│               (HyDE, Multi-Q)   (BM25 + Vector) (Cohere)         │
│  → Context Compression → Prompt → LLM → Stream + Citations      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Observability Layer                          │
│                                                                  │
│  Tracing (LangSmith) + Metrics (Prometheus) + Eval (Ragas)      │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
advanced_project/
├── README.md
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── ingestion/
│   ├── parsers.py              # PDF, web, image parsing
│   ├── chunking.py             # Recursive, semantic chunking
│   ├── embeddings.py           # OpenAI, Cohere embeddings
│   └── pipeline.py             # Orchestration
├── retrieval/
│   ├── vector_store.py         # Weaviate, Qdrant clients
│   ├── hybrid_search.py        # Dense + sparse retrieval
│   ├── reranker.py             # Cohere reranking
│   └── query_transform.py     # HyDE, multi-query
├── generation/
│   ├── llm_client.py           # OpenAI, Anthropic clients
│   ├── prompt_templates.py     # Optimized prompts
│   ├── context_compression.py  # LongLLMLingua
│   └── streaming.py            # SSE streaming
├── evaluation/
│   ├── ragas_eval.py           # Faithfulness, relevance
│   ├── benchmarks.py           # Custom test sets
│   └── ab_testing.py           # A/B comparison
├── api/
│   ├── main.py                 # FastAPI application
│   ├── routes.py               # API endpoints
│   ├── websocket.py            # Streaming responses
│   └── middleware.py           # Auth, logging, rate limiting
├── monitoring/
│   ├── tracing.py              # LangSmith integration
│   ├── metrics.py              # Prometheus metrics
│   └── alerts.py               # Alert rules
├── frontend/
│   ├── streamlit_app.py        # Demo UI
│   └── components/             # Custom components
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_generation.py
    └── test_e2e.py
```

## Phase 1: Document Ingestion (Week 1-2)

### Task 1.1: Multi-Format Parsing

```python
# ingestion/parsers.py
from unstructured.partition.auto import partition
from unstructured.documents.elements import Title, NarrativeText, Table
from PIL import Image
import pytesseract

class DocumentParser:
    \"\"\"Parse documents from multiple formats.\"\"\"
    
    def parse_pdf(self, file_path: str) -> list[dict]:
        \"\"\"Extract text, tables, images from PDF.\"\"\"
        elements = partition(filename=file_path)
        
        parsed = []
        for elem in elements:
            if isinstance(elem, (Title, NarrativeText)):
                parsed.append({
                    'type': 'text',
                    'content': elem.text,
                    'metadata': elem.metadata.to_dict()
                })
            elif isinstance(elem, Table):
                parsed.append({
                    'type': 'table',
                    'content': str(elem),
                    'metadata': elem.metadata.to_dict()
                })
        
        return parsed
    
    def parse_webpage(self, url: str) -> dict:
        \"\"\"Scrape and parse web content.\"\"\"
        import trafilatura
        
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded,
                                   include_links=True,
                                   include_images=False)
        
        return {
            'content': text,
            'url': url,
            'retrieved_at': datetime.now().isoformat()
        }
    
    def parse_image(self, image_path: str) -> str:
        \"\"\"OCR for images.\"\"\"
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
```

### Task 1.2: Advanced Chunking

```python
# ingestion/chunking.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter

class AdvancedChunker:
    \"\"\"Smart chunking strategies.\"\"\"
    
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def recursive_chunk(self, text: str) -> list[str]:
        \"\"\"Recursive chunking with semantic boundaries.\"\"\"
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[\"\\n\\n\", \"\\n\", \".\", \" \", \"\"],
        )
        return splitter.split_text(text)
    
    def semantic_chunk(self, text: str) -> list[str]:
        \"\"\"Chunk by semantic similarity.\"\"\"
        splitter = TextSplitter(self.chunk_size)
        return splitter.chunks(text)
    
    def sliding_window_chunk(self, text: str) -> list[dict]:
        \"\"\"Overlapping windows with parent context.\"\"\"
        # Create large parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,
            chunk_overlap=0
        )
        parents = parent_splitter.split_text(text)
        
        # Create small child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = []
        for parent_idx, parent in enumerate(parents):
            children = child_splitter.split_text(parent)
            for child in children:
                chunks.append({
                    'chunk': child,
                    'parent': parent,
                    'parent_id': parent_idx
                })
        
        return chunks
```

### Task 1.3: Embedding & Indexing

```python
# ingestion/embeddings.py
from openai import OpenAI
import weaviate

class EmbeddingPipeline:
    \"\"\"Generate embeddings and index in vector DB.\"\"\"
    
    def __init__(self):
        self.openai_client = OpenAI()
        self.weaviate_client = weaviate.Client(\"http://localhost:8080\")
    
    def embed_batch(self, texts: list[str], model=\"text-embedding-3-large\") -> list:
        \"\"\"Batch embed texts.\"\"\"
        response = self.openai_client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def index_documents(self, chunks: list[dict]):
        \"\"\"Index chunks in Weaviate.\"\"\"
        # Create schema
        schema = {
            \"class\": \"Document\",
            \"vectorizer\": \"none\",  # We provide vectors
            \"properties\": [
                {\"name\": \"content\", \"dataType\": [\"text\"]},
                {\"name\": \"source\", \"dataType\": [\"text\"]},
                {\"name\": \"chunk_id\", \"dataType\": [\"int\"]},
            ]
        }
        
        # Batch index
        with self.weaviate_client.batch as batch:
            batch.batch_size = 100
            
            for chunk in chunks:
                embedding = self.embed_batch([chunk['content']])[0]
                
                properties = {
                    \"content\": chunk['content'],
                    \"source\": chunk['metadata'].get('source'),
                    \"chunk_id\": chunk['chunk_id']
                }
                
                batch.add_data_object(
                    data_object=properties,
                    class_name=\"Document\",
                    vector=embedding
                )
```

**Deliverables**:
- [ ] Support PDF, web, image parsing
- [ ] Implement 3 chunking strategies
- [ ] Index 10,000+ chunks in vector DB
- [ ] Metadata preservation

## Phase 2: Hybrid Retrieval (Week 3)

### Task 2.1: Dense + Sparse Search

```python
# retrieval/hybrid_search.py
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    \"\"\"Combine dense and sparse retrieval.\"\"\"
    
    def __init__(self, weaviate_client, documents):
        self.weaviate_client = weaviate_client
        self.documents = documents
        
        # Build BM25 index
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def dense_search(self, query: str, k=10) -> list[dict]:
        \"\"\"Vector similarity search.\"\"\"
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Search Weaviate
        result = self.weaviate_client.query\\
            .get(\"Document\", [\"content\", \"source\", \"chunk_id\"])\\
            .with_near_vector({\"vector\": query_embedding})\\
            .with_limit(k)\\
            .do()
        
        return result['data']['Get']['Document']
    
    def sparse_search(self, query: str, k=10) -> list[dict]:
        \"\"\"BM25 keyword search.\"\"\"
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        
        return [
            {'content': self.documents[i], 'score': scores[i]}
            for i in top_k_idx
        ]
    
    def hybrid_search(self, query: str, k=20, alpha=0.5) -> list[dict]:
        \"\"\"Combine dense and sparse with RRF (Reciprocal Rank Fusion).\"\"\"
        # Get both results
        dense_results = self.dense_search(query, k)
        sparse_results = self.sparse_search(query, k)
        
        # RRF scoring
        scores = {}
        for rank, doc in enumerate(dense_results):
            doc_id = doc['chunk_id']
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (rank + 60)
        
        for rank, doc in enumerate(sparse_results):
            doc_id = doc.get('chunk_id', hash(doc['content']))
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (rank + 60)
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
```

### Task 2.2: Reranking

```python
# retrieval/reranker.py
import cohere

class Reranker:
    \"\"\"Rerank results using cross-encoder.\"\"\"
    
    def __init__(self):
        self.cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
    
    def rerank(self, query: str, documents: list[str], top_k=5) -> list[dict]:
        \"\"\"Rerank documents using Cohere.\"\"\"
        response = self.cohere_client.rerank(
            model=\"rerank-english-v3.0\",
            query=query,
            documents=documents,
            top_n=top_k
        )
        
        return [
            {
                'content': documents[result.index],
                'relevance_score': result.relevance_score,
                'index': result.index
            }
            for result in response.results
        ]
```

### Task 2.3: Query Transformations

```python
# retrieval/query_transform.py
from openai import OpenAI

class QueryTransformer:
    \"\"\"Transform queries for better retrieval.\"\"\"
    
    def __init__(self):
        self.client = OpenAI()
    
    def hyde(self, query: str) -> str:
        \"\"\"Hypothetical Document Embeddings (HyDE).
        
        Generate a hypothetical answer, then search for it.
        \"\"\"
        prompt = f\"\"\"Generate a detailed answer to this question.
        
Question: {query}

Answer:\"\"\"
        
        response = self.client.chat.completions.create(
            model=\"gpt-4o-mini\",
            messages=[{\"role\": \"user\", \"content\": prompt}],
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    def multi_query(self, query: str, num_variations=3) -> list[str]:
        \"\"\"Generate multiple query variations.\"\"\"
        prompt = f\"\"\"Generate {num_variations} different ways to ask this question:

{query}

Output only the questions, one per line.\"\"\"
        
        response = self.client.chat.completions.create(
            model=\"gpt-4o-mini\",
            messages=[{\"role\": \"user\", \"content\": prompt}],
            max_tokens=150
        )
        
        variations = response.choices[0].message.content.strip().split('\\n')
        return [q.strip('1234567890. ') for q in variations if q.strip()]
```

**Deliverables**:
- [ ] Dense + sparse hybrid search
- [ ] Reranking with cross-encoder
- [ ] HyDE and multi-query expansion
- [ ] Retrieval evaluation (recall@k, MRR)

## Phase 3: Generation & Streaming (Week 4)

### Task 3.1: Optimized Prompts

```python
# generation/prompt_templates.py
class RAGPromptTemplates:
    \"\"\"Optimized prompts for RAG.\"\"\"
    
    @staticmethod
    def qa_with_citations():
        return \"\"\"You are a helpful AI assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer using ONLY information from the context
2. Include in-text citations like [1], [2], etc.
3. If the answer is not in the context, say \"I cannot answer based on the provided context.\"
4. Be concise and accurate

Answer:\"\"\"
    
    @staticmethod
    def system_message():
        return \"\"\"You are an expert RAG assistant. You provide accurate, cited answers based on retrieved context. Never hallucinate.\"\"\"
```

### Task 3.2: Streaming with Citations

```python
# generation/streaming.py
from openai import OpenAI
from typing import Generator

class StreamingRAG:
    \"\"\"Stream responses with real-time citations.\"\"\"
    
    def __init__(self):
        self.client = OpenAI()
    
    def stream_response(self, query: str, context: list[dict]) -> Generator:
        \"\"\"Stream response with citations.\"\"\"
        # Format context with source IDs
        context_text = \"\\n\\n\".join([
            f\"[{i+1}] {doc['content']}\\nSource: {doc['source']}\"
            for i, doc in enumerate(context)
        ])
        
        prompt = RAGPromptTemplates.qa_with_citations().format(
            context=context_text,
            question=query
        )
        
        # Stream
        stream = self.client.chat.completions.create(
            model=\"gpt-4o\",
            messages=[
                {\"role\": \"system\", \"content\": RAGPromptTemplates.system_message()},
                {\"role\": \"user\", \"content\": prompt}
            ],
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### Task 3.3: Context Compression

```python
# generation/context_compression.py
from llmlingua import PromptCompressor

class ContextCompressor:
    \"\"\"Compress context to fit within token limits.\"\"\"
    
    def __init__(self):
        self.compressor = PromptCompressor()
    
    def compress(self, context: str, query: str, target_tokens=2000) -> str:
        \"\"\"LongLLMLingua compression.\"\"\"
        compressed = self.compressor.compress_prompt(
            context.split(),
            instruction=\"\",
            question=query,
            target_token=target_tokens,
            condition_compare=True,
            reorder_context=\"sort\"
        )
        
        return compressed['compressed_prompt']
```

**Deliverables**:
- [ ] Streaming responses (SSE or WebSocket)
- [ ] In-line citations with source tracking
- [ ] Context compression for long docs
- [ ] Prompt optimization experiments

## Phase 4: Evaluation Framework (Week 5)

### Task 4.1: RAGAS Evaluation

```python
# evaluation/ragas_eval.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

class RAGEvaluator:
    \"\"\"Evaluate RAG system quality.\"\"\"
    
    def evaluate_pipeline(self, test_set: list[dict]) -> dict:
        \"\"\"
        test_set format:
        [
            {
                'question': str,
                'ground_truth': str,
                'answer': str,
                'contexts': list[str]
            },
            ...
        ]
        \"\"\"
        result = evaluate(
            test_set,
            metrics=[
                faithfulness,         # Does answer align with context?
                answer_relevancy,     # Does answer address question?
                context_recall,       # Is all necessary info retrieved?
            ]
        )
        
        return result
```

### Task 4.2: Custom Benchmarks

```python
# evaluation/benchmarks.py
class BenchmarkSuite:
    \"\"\"Domain-specific evaluation.\"\"\"
    
    def __init__(self, test_questions: list[dict]):
        self.test_questions = test_questions
    
    def run_benchmark(self, rag_system) -> dict:
        \"\"\"Run full evaluation.\"\"\"
        results = {
            'accuracy': [],
            'latency': [],
            'retrieval_quality': [],
            'hallucination_rate': []
        }
        
        for item in self.test_questions:
            start = time.time()
            
            # Get RAG answer
            answer = rag_system.query(item['question'])
            
            latency = time.time() - start
            
            # Evaluate
            is_correct = self.check_correctness(answer, item['ground_truth'])
            has_hallucination = self.detect_hallucination(answer, item['context'])
            
            results['accuracy'].append(is_correct)
            results['latency'].append(latency)
            results['hallucination_rate'].append(has_hallucination)
        
        return {
            'accuracy': np.mean(results['accuracy']),
            'p95_latency': np.percentile(results['latency'], 95),
            'hallucination_rate': np.mean(results['hallucination_rate'])
        }
```

**Deliverables**:
- [ ] RAGAS metrics (faithfulness, relevance)
- [ ] Custom domain-specific tests
- [ ] A/B testing framework
- [ ] Performance benchmarks

## Phase 5: Production Deployment (Week 6)

### Task 5.1: FastAPI Service

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title=\"Production RAG API\")

class QueryRequest(BaseModel):
    question: str
    stream: bool = False
    top_k: int = 5

@app.post(\"/query\")
async def query(request: QueryRequest):
    try:
        # Retrieve
        contexts = hybrid_retriever.search(request.question, k=request.top_k)
        
        # Generate
        if request.stream:
            return StreamingResponse(
                stream_generator(request.question, contexts),
                media_type=\"text/event-stream\"
            )
        else:
            answer = llm_client.generate(request.question, contexts)
            return {\"answer\": answer, \"sources\": contexts}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(\"/health\")
async def health():
    # Check all dependencies
    return {
        \"status\": \"healthy\",
        \"vector_db\": check_weaviate(),
        \"llm\": check_openai()
    }
```

### Task 5.2: Monitoring

```python
# monitoring/tracing.py
from langsmith import Client
from prometheus_client import Histogram, Counter

# LangSmith tracing
ls_client = Client()

# Prometheus metrics
QUERY_LATENCY = Histogram('rag_query_latency_seconds', 'Query latency')
RETRIEVAL_LATENCY = Histogram('rag_retrieval_latency_seconds', 'Retrieval latency')
GENERATION_LATENCY = Histogram('rag_generation_latency_seconds', 'Generation latency')
QUERY_COUNT = Counter('rag_queries_total', 'Total queries')

@ls_client.trace
def rag_query(question: str):
    with QUERY_LATENCY.time():
        QUERY_COUNT.inc()
        
        with RETRIEVAL_LATENCY.time():
            contexts = retrieve(question)
        
        with GENERATION_LATENCY.time():
            answer = generate(question, contexts)
        
        return answer
```

**Deliverables**:
- [ ] REST API (FastAPI)
- [ ] Streaming endpoints
- [ ] Prometheus metrics
- [ ] LangSmith tracing
- [ ] Grafana dashboard

## Success Criteria

**Minimum**:
- ✅ Ingest 1000+ documents
- ✅ P95 latency <2s
- ✅ Faithfulness score >0.8

**Production**:
- 🌟 Ingest 100k+ documents
- 🌟 P95 latency <1s
- 🌟 Hybrid search outperforms pure vector
- 🌟 Streaming responses
- 🌟 Full observability

**Industry-Grade**:
- 🏆 Multi-million document scaling
- 🏆 P95 latency <500ms
- 🏆 Multi-modal (text + images + tables)
- 🏆 Auto-evaluation + drift detection
- 🏆 A/B testing framework

## Resources

- [RAG Papers Collection](https://github.com/hymie122/RAG-Survey)
- [RAGAS Documentation](https://docs.ragas.io/)
- [Llamaindex](https://docs.llamaindex.ai/)
- [Haystack](https://haystack.deepset.ai/)

This is your chance to build state-of-the-art RAG!
