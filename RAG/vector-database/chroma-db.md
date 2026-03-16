# Introduction to Vector Databases & Chroma DB — Cheat Sheet

---

## Distance & Similarity Metrics

| Metric | Sensitive to Magnitude | Normalized | Best For |
|---|---|---|---|
| **L2 Distance** | ✅ Yes | ❌ No | Spatial data, clustering, computer vision |
| **Cosine Distance** | ❌ No | ✅ Yes | Text, embeddings, NLP |
| **Dot Product** | ✅ Yes | ❌ No | Neural networks, recommender systems |

### Key Formulas

```
L2(a,b)        = sqrt(sum of squared differences)
Dot Product    = sum of element-wise products
Cosine Sim     = dot(a,b) / (||a|| * ||b||)
Cosine Dist    = 1 - cosine_similarity(a,b)
```

> 💡 If vectors are normalized, cosine similarity = dot product. Normalize when you only need cosine similarity.

---

## Vector DB vs Traditional DB

| Function | Traditional DB | Vector DB |
|---|---|---|
| Data Format | Tables, rows, columns | Multi-dimensional vectors |
| Search | SQL queries | Similarity / nearest neighbor search |
| Indexing | B-trees | Graph-based HNSW |
| Scalability | Sharding / resource augmentation | Horizontal scaling via distributed architecture |
| Use Case | Business apps, transactions | AI apps, NLP, multimedia, semantic search |

**Vector Libraries vs Vector Databases**
- **Libraries** — in-memory, read & update only
- **Databases** — persistent, full CRUD, enterprise-ready

---

## HNSW — Vector Index in Chroma DB

Chroma DB's only indexing method. Builds a **multi-layered graph**:
- Upper layers → sparse overview for fast navigation
- Bottom layer → all vectors for detailed search
- Search descends from top layer toward the query vector, pruning irrelevant regions early

**Why HNSW?** Fast · Accurate · Scalable · Works with any similarity metric

### HNSW Parameters

| Parameter | Effect | Tradeoff |
|---|---|---|
| `space` | Distance metric (`l2`, `ip`, `cosine`) | — |
| `ef_search` | Search breadth at query time | ↑ accuracy vs ↑ query time |
| `ef_construction` | Index quality at build time | ↑ accuracy vs ↑ build time |
| `max_neighbors` | Graph density | ↑ search quality vs ↑ memory |

---

## Chroma DB Setup

```python
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.Client()

collection = client.create_collection(
    name="my_collection",
    configuration={
        "hnsw": {"space": "cosine"},
        "embedding_function": ef
    }
)
```

---

## Data Operations

```python
# Add documents
collection.add(
    documents=["Doc text 1", "Doc text 2"],
    metadatas=[{"source": "a", "version": 0.1}, {"source": "b", "version": 0.2}],
    ids=["id1", "id2"]
)

# Get all
collection.get()

# Get with filter
collection.get(where={"source": "a"})
```

---

## Filtering

### Metadata Operators

```python
"$eq"   # equal (default if no operator given)
"$ne"   # not equal
"$gt" / "$gte"  # greater than / or equal
"$lt" / "$lte"  # less than / or equal
"$in"   # in list
"$nin"  # not in list
```

**Combined filters:**
```python
collection.get(
    where={
        "$and": [
            {"source": {"$in": ["langchain.com", "llamaindex.ai"]}},
            {"version": {"$lt": 0.3}}
        ]
    }
)
```

### Document (Content) Filtering

```python
# Contains
where_document={"$contains": "pandas"}

# Does not contain
where_document={"$not_contains": "library"}

# Combined
where_document={"$or": [{"$contains": "LangChain"}, {"$contains": "Python"}]}
```

> ⚠️ Document filtering is **case-sensitive**

---

## Similarity Search

```python
# Basic query
collection.query(query_texts=["search term"], n_results=3)

# With metadata filter
collection.query(query_texts=["polar bear"], n_results=1, where={"topic": "animals"})

# With document filter
collection.query(query_texts=["polar bear"], n_results=1, where_document={"$not_contains": "library"})

# Combined
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={"topic": "animals"},
    where_document={"$not_contains": "library"}
)
```

---

## Full Workflow

```python
# 1. Setup
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()

# 2. Create collection
collection = client.create_collection(
    name="collection_name",
    configuration={"hnsw": {"space": "cosine"}, "embedding_function": ef}
)

# 3. Add documents
collection.add(documents=texts, metadatas=metadata, ids=ids)

# 4. Search
results = collection.query(query_texts=["query"], n_results=5)

# 5. Process results
for i, (doc_id, score, text) in enumerate(
    zip(results['ids'][0], results['distances'][0], results['documents'][0])
):
    print(f"Rank {i+1}: {doc_id} | Score: {score:.4f} | {text}")
```

---

> *Author: Wojciech "Victor" Fulmyk*
