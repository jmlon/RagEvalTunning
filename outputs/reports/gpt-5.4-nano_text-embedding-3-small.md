# RAG Benchmark Report

**Models (LLM / embedding):** `openai/gpt-5.4-nano` / `openai/text-embedding-3-small`

Scores: perfect=5, good=4, partial=2, poor=1, wrong=0, no answer=0.

| System | LLM | Embedding | Questions | Mean /5 | perfect | good | partial | poor | wrong | no answer | Query latency (s) | Query tokens | Query cost ($) | Index time (s) | Index tokens | Index cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple-rag | openai/gpt-5.4-nano | openai/text-embedding-3-small | 5 | 5.00 | 5 | 0 | 0 | 0 | 0 | 0 | 12.73 | 7726 | 0.002148 | 7.49 | 14842 | 0.000297 |
| simple-rag-reranker | openai/gpt-5.4-nano | openai/text-embedding-3-small | 5 | 4.40 | 4 | 0 | 1 | 0 | 0 | 0 | 16.37 | 6606 | 0.002025 | 0.00 | 0 | 0.000000 |
| hybrid-rag | openai/gpt-5.4-nano | openai/text-embedding-3-small | 5 | 4.20 | 3 | 1 | 1 | 0 | 0 | 0 | 14.40 | 7785 | 0.002221 | 6.73 | 14842 | 0.000297 |
| hybrid-rag-reranker | openai/gpt-5.4-nano | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 15.23 | 6603 | 0.002017 | 0.00 | 0 | 0.000000 |
| graph-rag | openai/gpt-5.4-nano | openai/text-embedding-3-small | 5 | 4.80 | 4 | 1 | 0 | 0 | 0 | 0 | 14.59 | 20991 | 0.004809 | 235.87 | 0 | 0.000000 |
| llm-wiki | openai/gpt-5.4-nano | openai/text-embedding-3-small | 5 | 3.40 | 3 | 0 | 1 | 0 | 1 | 0 | 19.84 | 45182 | 0.009840 | 162.83 | 39843 | 0.033950 |

## Configuration

### simple-rag

- LLM: `openai/gpt-5.4-nano`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `150`
- chunk_size: `400`
- temperature: `0.0`
- top_k: `15`

### simple-rag-reranker

- LLM: `openai/gpt-5.4-nano`
- Embedding: `openai/text-embedding-3-small`
- rerank_model: `rerank-v3.5`
- retrieve_k: `40`
- source_system: `simple-rag`
- temperature: `0.0`
- top_k: `12`

### hybrid-rag

- LLM: `openai/gpt-5.4-nano`
- Embedding: `openai/text-embedding-3-small`
- bm25_weight: `0.333`
- chunk_overlap: `150`
- chunk_size: `400`
- dense_weight: `0.667`
- temperature: `0.0`
- top_k: `15`

### hybrid-rag-reranker

- LLM: `openai/gpt-5.4-nano`
- Embedding: `openai/text-embedding-3-small`
- bm25_weight: `0.333`
- dense_weight: `0.667`
- rerank_model: `rerank-v3.5`
- retrieve_k: `40`
- source_system: `hybrid-rag`
- temperature: `0.0`
- top_k: `12`

### graph-rag

- LLM: `openai/gpt-5.4-nano`
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

- LLM: `openai/gpt-5.4-nano`
- Embedding: `openai/text-embedding-3-small`
- chunk_overlap: `300`
- chunk_size: `3000`
- link_hops: `1`
- route_max_pages: `8`
- temperature: `0.0`
