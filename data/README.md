# Data Templates

This directory contains templates for evaluation datasets and indexes.

Files

- qa.example.jsonl: Example QA pairs used by RAG evaluation
- index/: Documentation for building the hybrid (BM25 + vector) index

qa.jsonl schema

- id: unique string id
- question: user query text
- answer: reference answer text (for evaluation)
- citations: array of objects { doc_id, chunk_id } expected to support citation_rate metric

Example

{"id":"q1","question":"What is Rust ownership?","answer":"Ownership is a set of rules...","citations":[{"doc_id":"rust-book","chunk_id":"ch4-1"}]}
{"id":"q2","question":"What is RAG?","answer":"Retrieval augmented generation...","citations":[{"doc_id":"blog-rag","chunk_id":"sec-2"}]}

Index

See index/README.md for structure and required fields for BM25 and vector indices.
