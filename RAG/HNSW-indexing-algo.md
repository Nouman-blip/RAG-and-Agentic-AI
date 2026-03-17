# 🗺️ HNSW — Quick Cheat Sheet
### Hierarchical Navigable Small World

> **Analogy:** Finding a restaurant using Google Maps — zoom out (highways) → zoom in (streets) → arrive at destination. HNSW does the same, but for similar data points in massive databases.

---

## 🧱 Three Building Blocks

### 1. Small World Networks
```
You → Friends → Friends of Friends → ... → Anyone on Earth
              (avg. 6 degrees of separation)
```
- **High Clustering:** Nodes form tight-knit groups
- **Low Path Length:** Any node reachable in few hops

### 2. Navigable Networks — Greedy Routing
```
Start → Look at neighbors → Jump to closest → Repeat → Can't get closer? Stop.
```
- Time complexity: **O(log^k n)** — far faster than linear scan

### 3. Hierarchical Layers (The Key Insight)

```
Layer 2  [ • ——————————————— • ]        ← Express highways (few nodes)
Layer 1  [ • —— • —— • —— • —— • ]     ← Main roads
Layer 0  [ •-•-•-•-•-•-•-•-•-•-• ]    ← Local streets (ALL nodes)
```

> Each layer up = ~**half** the nodes. Assignment is **random + exponential decay**.

---

## 🔍 Search — Step by Step

```
1. Enter at random point in TOP layer
        ↓
2. Greedy search in current layer
   (move to neighbor closest to query)
        ↓
3. Hit local minimum? → Drop DOWN one layer
        ↓
4. Repeat greedy search with finer connections
        ↓
5. Reach Layer 0 → final greedy search
        ↓
6. Return Approximate Nearest Neighbor ✅
```

**Example gain:** 8 distance computations vs. 12 brute-force (scales massively with real data)

---

## 🏗️ Index Build — Step by Step

```
1. Empty graph → first point = entry point

2. New point arrives → assign random layer height
   (most points = Layer 0 only; few reach higher layers)

3. Greedy search from top layer → find best position

4. At each layer: connect to M closest neighbors (bidirectional)

5. Repeat for all points → multi-layer graph is complete
```

---

## ⚙️ Key Parameters

| Parameter | Role | Higher → | Lower → |
|---|---|---|---|
| **M** | Max connections per node | Better recall, more memory | Faster build, less memory |
| **efConstruction** | Candidates explored during build | Better graph quality, slower build | Faster build, lower quality |
| **efSearch** | Candidates explored during query | Better accuracy, slower search | Faster search, may miss matches |
| **ml** | Level multiplier | More tall buildings (nodes in higher layers) | Flatter hierarchy |

**Recommended defaults:** `M=16`, `efConstruction=200`  
**Main tuning knob at query time:** `efSearch` ← speed vs. accuracy

---

## ⚖️ Trade-offs

| Factor | Reality |
|---|---|
| **Accuracy** | Approximate — recall typically **90–99%** |
| **Speed** | Search: **O(log n)** vs brute-force **O(n)** |
| **Memory** | Higher M = more RAM needed |
| **Updates** | Best for **static** datasets; frequent inserts/deletes degrade graph |
| **Metrics** | Works best with **L2 (Euclidean)** and **Cosine similarity** |

---

## ✅ When to Use / ❌ When to Avoid

| ✅ Use HNSW | ❌ Avoid HNSW |
|---|---|
| Large-scale similarity search | Exact results required |
| High-dimensional data (text, images, audio) | Very small datasets |
| Good accuracy + fast query time | Memory is critically constrained |
| Mostly static datasets | Frequent inserts/deletes |

---

## 🔬 The Science

**Origin:** Malkov & Yashunin, 2016 — *"Efficient and robust approximate nearest neighbor search using HNSW graphs"*

**Key fusion of two ideas:**
```
Skip Lists (Pugh, 1989)          →  Probabilistic multi-level structure
Navigable Small World Networks   →  Greedy routing toward target
         ↓                  ↓
              HNSW ✅
```

**Complexity:**
```
Brute-force search:  O(n)        ← checks everything
HNSW search:         O(log n)    ← navigates hierarchy
```

---

## 🧠 One-Line Summary

> **HNSW = Multi-layer graph where top layers make big jumps (highways) and bottom layers find exact neighbors (local streets) — achieving near-perfect recall at logarithmic speed.**

---

*Source: cognitiveclass.ai — Hierarchical Navigable Small World (HNSW)*
