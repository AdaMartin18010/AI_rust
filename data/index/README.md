# Index Structure

BM25

- docs.jsonl: each line { doc_id, text, metadata }
- tokenizer: description of tokenization and language
- fields: which fields are indexed (e.g., text)

Vector

- embeddings.parquet: columns { doc_id, chunk_id, vector: `list<float>`, dim }
- model.json: { name, version, dim, normalize }
- mapping.json: ties chunk_id to raw text spans and page/segment ids used for citations

Hybrid parameters

- k (retrieval depth), kprime (partial re-ranking size)
- reranker (cross-encoder model name)

Provenance

- Provide dataset/version, build command and checksum for each file.
