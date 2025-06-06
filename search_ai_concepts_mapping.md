
# 🔍 Search Types and AI Concepts Mapping

This document outlines the different search techniques used in your hybrid search system, and maps each one to the underlying AI, NLP, and IR concepts.

---

## 📊 Search Types and Concepts Table

| Search Type       | Description                                                                 | AI Concepts Involved                            | Notes                                                                 |
|-------------------|-----------------------------------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------|
| **Full-Text Search** | Traditional keyword-based match on fields like `notes`                     | 🔹 BM25<br>🔹 Text Inverted Index                 | No semantic understanding; exact word matches only                   |
| **Vector Search**   | Search using semantic embeddings (`combined_text`) via KNN                 | 🔹 Dense Vectors (384-d)<br>🔹 HNSW<br>🔹 Cosine Similarity | Returns semantically similar results even without exact word match   |
| **Semantic Search** | Vector-based search with post-filtering (`min_score`)                      | 🔹 Sentence Embeddings<br>🔹 Cosine<br>🔹 ANN (HNSW)<br>🔹 Thresholding | Adds quality gating to avoid irrelevant vector matches               |
| **Hybrid Search**   | Combines vector similarity with lexical match & optional filters           | 🔹 BM25<br>🔹 Vectors<br>🔹 Cosine<br>🔹 Boolean logic<br>🔹 Score blending | Stronger than either alone; combines full-text and semantic match    |
| **Future RAG (Planned)** | Retrieval + LLM reasoning using top-k documents                        | 🔹 RAG (Retrieval-Augmented Generation)<br>🔹 GPT / LLMs<br>🔹 Prompting | For answering complex questions by combining search with generation  |
| **Optional Reranking** | Reordering retrieved docs based on deep semantic similarity              | 🔹 Cross-encoders<br>🔹 Transformers<br>🔹 Score re-ranking | Improves ordering; common in production (e.g., Cohere/LLM rerankers) |
| **Filter Extraction (Semantic Parsing)** | Extracting filters like `status = Active` from user query          | 🔹 NER / Regex<br>🔹 Semantic Query Understanding | Automates field-based filtering from natural language input          |

---

## ✅ Concept Summary

| Concept             | Used? | Where?                                        |
|---------------------|-------|-----------------------------------------------|
| **BM25**            | ✅    | Full-text and hybrid searches                 |
| **Dense Vectors**   | ✅    | Semantic, vector, hybrid                      |
| **Cosine Similarity**| ✅   | Vector-based KNN via Elasticsearch            |
| **HNSW ANN**        | ✅    | Vector search acceleration                    |
| **Filter Parsing**  | ✅    | Hybrid (auto filters from text)               |
| **RAG**             | 🟡    | Optional for advanced reasoning + generation  |
| **Reranking**       | 🔜    | Optional add-on for production-quality search |
