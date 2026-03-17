# 🔍 Advanced Retrievers for RAG — Quick Cheat Sheet

> **Goal:** Go beyond simple vector search — retrieve smarter, faster, and more accurately.

---

## 🧠 Core Concepts at a Glance

| Concept | What It Does |
|---|---|
| **Semantic Search** | Uses embeddings to match *meaning*, not just words |
| **Keyword Matching** | Precise term-based search (exact specs, legal terms) |
| **Hierarchical Context** | Keeps relationships between document chunks intact |
| **Multi-Query** | Generates query variations → combines results |
| **Fusion** | Merges results from multiple retrieval methods |
| **MMR** | Picks relevant *and* diverse docs — avoids redundancy |

---

## 📦 LlamaIndex — Index Types

```
VectorStoreIndex      → Semantic retrieval (embedding-based)
DocumentSummaryIndex  → Summary-based retrieval (for large docs)
KeywordTableIndex     → Keyword extraction + exact term matching
```

> 💡 `DocumentSummaryIndex` returns **full documents**, not summaries — summaries are just used to *find* the right doc.

---

## 🦙 LlamaIndex Retrievers

### 1. Vector Index Retriever
```
Query → Embed → Cosine Similarity → Top-K Docs
```
- ✅ Best for: General RAG, semantic understanding
- ❌ Miss: Exact keyword matches

---

### 2. BM25 Retriever *(Keyword-based)*

**TF-IDF Foundation:**
```
TF  = How often term appears in doc
IDF = How rare term is across all docs
Score = TF × IDF
```

**BM25 Improvements:**
```
k1 ≈ 1.2  → Controls term frequency saturation
b  ≈ 0.75 → Normalizes for document length
```
- ✅ Best for: Legal, technical, exact terminology docs

---

### 3. Document Summary Index Retrievers

| Variant | Method | Trade-off |
|---|---|---|
| `LLMRetriever` | LLM compares query vs summaries | Smarter, slower, costly |
| `EmbeddingRetriever` | Embedding similarity on summaries | Faster, cheaper |

**Flow:** Summary filter → Return full document

---

### 4. Auto Merging Retriever
```
Index as hierarchy:  [Parent Chunk]
                    /     |      \
              [Child]  [Child]  [Child]

Retrieval: If enough children match → return Parent instead
```
- ✅ Best for: Long docs, legal papers, technical specs

---

### 5. Recursive Retriever
```
Node A → references → Node B → references → Node C
         (citations, metadata links)
```
- ✅ Best for: Academic papers, interconnected knowledge bases

---

### 6. Query Fusion Retriever

**Step 1:** Generate query variations via LLM  
**Step 2:** Run multiple retrievers  
**Step 3:** Fuse results using one of:

| Mode | Formula | Best For |
|---|---|---|
| **RRF** *(default)* | `score = Σ 1/(rank + 60)` | Production systems, most robust |
| **Relative Score** | `score = original / max_score` | When embedding confidence matters |
| **Distribution-Based** | Z-score / percentile ranking | Complex queries, varied distributions |

---

## 🦜 LangChain Retrievers

> Interface: **string query in → list of documents out**

---

### 1. Vector Store-Backed Retriever

| Search Type | Description |
|---|---|
| Similarity | Top-K by similarity score (default: 4) |
| MMR | Balances relevance + diversity |
| Score Threshold | Only returns docs above a set confidence |

---

### 2. Multi-Query Retriever
```
1 query → LLM → [variation 1, variation 2, variation 3]
                        ↓
               Retrieve for each variation
                        ↓
               Union of unique results ✅
```
- ✅ Fixes: Sensitivity to subtle wording changes

---

### 3. Self-Querying Retriever
```
"Movies about women rated above 8.5"
           ↓
    Semantic query: "movies about women"
    Metadata filter: rating > 8.5
           ↓
    Structured vector search ✅
```
- ✅ Best for: Apps combining semantic + attribute filtering
- ⚠️ Requires: Rich, structured metadata on documents

---

### 4. Parent Document Retriever
```
Stores:  [Small chunks] → for accurate embeddings
         [Parent chunks] → for full context

Retrieval:
  Match small chunk → look up parent ID → return large chunk ✅
```
- ✅ Solves: "Small for embeddings vs large for context" tradeoff

---

## ⚡ Decision Framework

| Need | LlamaIndex | LangChain |
|---|---|---|
| Simple semantic search | Vector Index Retriever | Vector Store-Backed (Similarity) |
| Exact keyword match | BM25 Retriever | — |
| Diverse results | — | Vector Store-Backed (MMR) |
| Multi-query fusion | Query Fusion (RRF) | Multi-Query Retriever |
| Hierarchical context | Auto Merging Retriever | Parent Document Retriever |
| Semantic + filters | — | Self-Querying Retriever |
| Citation following | Recursive Retriever | — |
| Large doc retrieval | Document Summary Index | — |

---

## 🔑 Key Formulas

```python
# BM25 Score
score = TF(term, doc) × IDF(term) 
      # with saturation (k1) and length norm (b)

# RRF Score
RRF_score(doc) = Σ [ 1 / (rank_i(doc) + 60) ]

# Relative Score Fusion
normalized = original_score / max_score
```

---

*Source: cognitiveclass.ai — Advanced Retrievers for RAG*
