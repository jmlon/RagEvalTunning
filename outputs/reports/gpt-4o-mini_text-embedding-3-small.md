# RAG Benchmark Report

**Models (LLM / embedding):** `openai/gpt-4o-mini` / `openai/text-embedding-3-small`

Scores: perfect=5, good=4, partial=2, poor=1, wrong=0, no answer=0.

| System | LLM | Embedding | Questions | Mean /5 | perfect | good | partial | poor | wrong | no answer | Query latency (s) | Query tokens | Query cost ($) | Index time (s) | Index tokens | Index cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple-rag | openai/gpt-4o-mini | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 11.52 | 7545 | 0.001306 | 8.62 | 14842 | 0.000297 |
| simple-rag-reranker | openai/gpt-4o-mini | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 20.65 | 6321 | 0.001119 | 0.00 | 0 | 0.000000 |
| hybrid-rag | openai/gpt-4o-mini | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 15.65 | 7539 | 0.001303 | 8.32 | 14842 | 0.000297 |
| hybrid-rag-reranker | openai/gpt-4o-mini | openai/text-embedding-3-small | 5 | 4.20 | 3 | 1 | 1 | 0 | 0 | 0 | 20.35 | 6323 | 0.001117 | 0.00 | 0 | 0.000000 |
| graph-rag | openai/gpt-4o-mini | openai/text-embedding-3-small | 5 | 3.00 | 1 | 2 | 1 | 0 | 1 | 0 | 16.14 | 13915 | 0.002260 | 180.76 | 0 | 0.000000 |
| llm-wiki | openai/gpt-4o-mini | openai/text-embedding-3-small | 5 | 4.20 | 3 | 1 | 1 | 0 | 0 | 0 | 16.65 | 18650 | 0.003018 | 126.78 | 23511 | 0.007278 |

## Configuration

### simple-rag

- LLM: `openai/gpt-4o-mini`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `150`
- chunk_size: `400`
- temperature: `0.0`
- top_k: `15`

### simple-rag-reranker

- LLM: `openai/gpt-4o-mini`
- Embedding: `openai/text-embedding-3-small`
- rerank_model: `rerank-v3.5`
- retrieve_k: `40`
- source_system: `simple-rag`
- temperature: `0.0`
- top_k: `12`

### hybrid-rag

- LLM: `openai/gpt-4o-mini`
- Embedding: `openai/text-embedding-3-small`
- bm25_weight: `0.333`
- chunk_overlap: `150`
- chunk_size: `400`
- dense_weight: `0.667`
- temperature: `0.0`
- top_k: `15`

### hybrid-rag-reranker

- LLM: `openai/gpt-4o-mini`
- Embedding: `openai/text-embedding-3-small`
- bm25_weight: `0.333`
- dense_weight: `0.667`
- rerank_model: `rerank-v3.5`
- retrieve_k: `40`
- source_system: `hybrid-rag`
- temperature: `0.0`
- top_k: `12`

### graph-rag

- LLM: `openai/gpt-4o-mini`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `300`
- chunk_size: `3000`
- max_passages: `6`
- max_triples: `60`
- neo4j_password_env: `NEO4J_LOCAL_PASSWORD`
- neo4j_uri: `bolt://localhost:7687`
- neo4j_username: `neo4j`
- seed_k: `8`
- temperature: `0.0`

### llm-wiki

- LLM: `openai/gpt-4o-mini`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `300`
- chunk_size: `3000`
- link_hops: `1`
- route_max_pages: `8`
- temperature: `0.0`
