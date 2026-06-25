# RAG Benchmark Report

**Models (LLM / embedding):** `openai/gpt-4.1-nano` / `openai/text-embedding-3-small`

Scores: perfect=5, good=4, partial=2, poor=1, wrong=0, no answer=0.

| System | LLM | Embedding | Questions | Mean /5 | perfect | good | partial | poor | wrong | no answer | Query latency (s) | Query tokens | Query cost ($) | Index time (s) | Index tokens | Index cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple-rag | openai/gpt-4.1-nano | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 12.45 | 7538 | 0.000868 | 8.72 | 14842 | 0.000297 |
| simple-rag-reranker | openai/gpt-4.1-nano | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 13.71 | 6325 | 0.000748 | 0.00 | 0 | 0.000000 |
| hybrid-rag | openai/gpt-4.1-nano | openai/text-embedding-3-small | 5 | 4.40 | 4 | 0 | 1 | 0 | 0 | 0 | 10.29 | 7575 | 0.000883 | 7.97 | 14842 | 0.000297 |
| hybrid-rag-reranker | openai/gpt-4.1-nano | openai/text-embedding-3-small | 5 | 4.20 | 3 | 1 | 1 | 0 | 0 | 0 | 16.08 | 6324 | 0.000746 | 0.00 | 0 | 0.000000 |
| graph-rag | openai/gpt-4.1-nano | openai/text-embedding-3-small | 5 | 3.60 | 2 | 2 | 0 | 0 | 1 | 0 | 12.99 | 13843 | 0.001479 | 0.00 | 0 | 0.000000 |
| llm-wiki | openai/gpt-4.1-nano | openai/text-embedding-3-small | 5 | 2.80 | 2 | 1 | 0 | 0 | 2 | 0 | 12.11 | 16139 | 0.001719 | 84.51 | 22145 | 0.004319 |

## Configuration

### simple-rag

- LLM: `openai/gpt-4.1-nano`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `150`
- chunk_size: `400`
- temperature: `0.0`
- top_k: `15`

### simple-rag-reranker

- LLM: `openai/gpt-4.1-nano`
- Embedding: `openai/text-embedding-3-small`
- rerank_model: `rerank-v3.5`
- retrieve_k: `40`
- source_system: `simple-rag`
- temperature: `0.0`
- top_k: `12`

### hybrid-rag

- LLM: `openai/gpt-4.1-nano`
- Embedding: `openai/text-embedding-3-small`
- bm25_weight: `0.333`
- chunk_overlap: `150`
- chunk_size: `400`
- dense_weight: `0.667`
- temperature: `0.0`
- top_k: `15`

### hybrid-rag-reranker

- LLM: `openai/gpt-4.1-nano`
- Embedding: `openai/text-embedding-3-small`
- bm25_weight: `0.333`
- dense_weight: `0.667`
- rerank_model: `rerank-v3.5`
- retrieve_k: `40`
- source_system: `hybrid-rag`
- temperature: `0.0`
- top_k: `12`

### graph-rag

- LLM: `openai/gpt-4.1-nano`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `300`
- chunk_size: `3000`
- max_passages: `6`
- max_triples: `60`
- neo4j_password_env: `NEO4J_LOCAL_PASSWORD`
- neo4j_uri: `bolt://localhost:7687`
- neo4j_uri_env: `NEO4J_LOCAL_URI`
- neo4j_username: `neo4j`
- seed_k: `8`
- temperature: `0.0`

### llm-wiki

- LLM: `openai/gpt-4.1-nano`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `300`
- chunk_size: `3000`
- link_hops: `1`
- route_max_pages: `8`
- temperature: `0.0`
