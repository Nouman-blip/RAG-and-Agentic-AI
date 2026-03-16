# Chroma DB — Filtering & Similarity Search

---

## Part 1: Filtering in Chroma DB

Chroma DB supports two types of filtering:

| Filter Type | How | SQL Equivalent |
|---|---|---|
| **Metadata Filtering** | Filter by document attributes (`source`, `date`, etc.) | `WHERE` clause |
| **Document Filtering** | Filter by content keywords (`$contains`, `$not_contains`) | `LIKE` / `CONTAINS` |

---

### Metadata Filtering — Operators

| Operator | Meaning |
|---|---|
| `$eq` | Equal to (default if no operator given) |
| `$ne` | Not equal to |
| `$gt` / `$gte` | Greater than / or equal |
| `$lt` / `$lte` | Less than / or equal |
| `$in` / `$nin` | In list / Not in list |
| `$and` / `$or` | Combine multiple filters |

**Basic match:**
```python
collection.get(where={"source": "langchain.com"})
```

**Combined filter:**
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

---

### Document (Content) Filtering

```python
# Find docs containing a keyword
collection.get(where_document={"$contains": "pandas"})

# Exclude docs with a keyword
collection.get(where_document={"$not_contains": "library"})
```

> ⚠️ Document filtering is **case-sensitive** — `"Pandas"` ≠ `"pandas"`

---

### Combine Both Filters

```python
collection.get(
    where={"version": {"$gt": 0.1}},
    where_document={
        "$or": [
            {"$contains": "LangChain"},
            {"$contains": "Python"}
        ]
    }
)
```

---

## Part 2: Similarity Search & HNSW

### What is a Vector Index?

A brute-force similarity search compares a query against every vector — slow at scale. A **vector index** organizes embeddings so only a small subset needs to be compared, enabling fast search across millions of vectors.

---

### HNSW — Hierarchical Navigable Small World

Chroma DB's only indexing method. A multi-layered graph where:
- **Upper layers** — sparse overview for fast navigation
- **Bottom layer** — all vectors for detailed search
- Each vector connects to a few nearby neighbors → most vectors reachable in just a few hops

**Search:** starts at the top layer, descends toward the query vector, pruning irrelevant regions early.

---

### Configuring HNSW

```python
collection = client.create_collection(
    name="my_collection",
    configuration={
        "hnsw": {
            "space": "cosine",      # l2 | ip | cosine
            "ef_search": 100,       # higher = more accurate, slower queries
            "ef_construction": 100, # higher = better index, slower build
            "max_neighbors": 16     # higher = denser graph, more memory
        },
        "embedding_function": ef
    }
)
```

| Parameter | Controls | Tradeoff |
|---|---|---|
| `space` | Distance metric | `cosine` recommended for text |
| `ef_search` | Search breadth at query time | ↑ accuracy vs ↑ query time |
| `ef_construction` | Index quality at build time | ↑ accuracy vs ↑ build time & memory |
| `max_neighbors` | Graph density | ↑ search quality vs ↑ memory |

---

### Querying with Similarity Search

```python
# Basic query
collection.query(query_texts=["cats"], n_results=3)

# With metadata filter
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where={"topic": "animals"}
)

# With document filter
collection.query(
    query_texts=["polar bear"],
    n_results=1,
    where_document={"$not_contains": "library"}
)
```

> 💡 When a query returns unexpected results, adding **metadata or document filters** is often the simplest fix — no need to change the embedding model.

---

## Key Takeaways

- Use `where` for metadata filtering and `where_document` for content filtering — combine both freely
- HNSW is Chroma DB's only index — tune `ef_search` for query speed, `ef_construction` and `max_neighbors` for index quality
- Semantic search can misfire on ambiguous terms — filters help narrow context and improve relevance

---

> *Authors: Wojciech "Victor" Fulmyk*
