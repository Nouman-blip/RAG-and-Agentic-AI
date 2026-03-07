# Vector Databases vs Traditional Databases

> **Estimated Reading Time: 10 minutes**

---

## What You Will Learn

After completing this reading, you will be able to:

- Explain the concept of a vector database
- Summarize how vector databases differ from traditional relational databases
- Describe how each database type stores data
- Explain the difference between a vector library and a vector database
- Evaluate the advantages and disadvantages of each based on specific needs

---

## Vector Databases

A vector database is a specialized database designed to store and query vectorized data rapidly. Unlike conventional databases that organize data in tables, a vector database represents data as **vectors in a multi-dimensional space**. These vectors encapsulate essential attributes of the items they represent, making vector databases ideal for:

- Similarity searches
- Nearest neighbor queries
- Assessing distances or similarities between vectors

### How Vector Databases Store Data

Each data item is represented as a numerical vector, where each dimension corresponds to a specific attribute or feature of the object. For example:

- **Image database** — each image is represented as a vector of pixel values
- **Text database** — each piece of text is represented as a vector of word frequencies

The pipeline looks like this:

```
Images / Text / Audio
        ↓
  Transformer Model
        ↓
  Vector Embeddings
        ↓
  Vector Database
```

---

## Vector Libraries vs Vector Databases

| Feature | Vector Libraries | Vector Databases |
|---|---|---|
| Storage | In-memory only | Persistent storage |
| CRUD Support | Read & update only | Full CRUD (Create, Read, Update, Delete) |
| Use Case | Lightweight similarity search | Enterprise-level production deployments |
| Configuration | Pre-configured algorithms | Flexible, scalable architecture |

---

## Relational Databases

A relational database organizes data into **tables** using rows and columns. It uses **SQL (Structured Query Language)** for querying and manipulation, and excels at managing structured data with well-defined relationships between entities.

### How Relational Databases Store Data

- Each **table** represents a distinct entity or relationship
- Each **row** corresponds to a record
- Each **column** represents a property or attribute
- Tables are connected using **primary keys** and **foreign keys**

Common SQL operations: `SELECT`, `INSERT`, `UPDATE`, `DELETE`

---

## Side-by-Side Comparison

| Function | Traditional (Relational) Databases | Vector Databases |
|---|---|---|
| **Data Representation** | Tables, rows, and columns — ideal for structured, relational data | Multi-dimensional vectors — efficient for unstructured data like images, text, audio |
| **Search & Retrieval** | SQL queries for structured data | Similarity searches for vectorized data (image retrieval, recommendations, anomaly detection) |
| **Indexing** | B-trees and similar methods | Metric trees and hashing suited for high-dimensional spaces |
| **Scalability** | Challenging — often requires resource augmentation or data sharding | Built for scale — distributed architectures support horizontal scaling |
| **Typical Applications** | Business apps, transactional systems, structured data processing | Scientific research, NLP, multimedia analysis, recommendation systems |

---

## Key Takeaways

- A **vector database** represents data numerically as vectors in a multi-dimensional space
- **Traditional databases** excel at structured data and transactional operations; **vector databases** excel at high-dimensional data and rapid similarity searches
- **Vector libraries** support read and update only; **vector databases** support full CRUD
- Traditional databases use **SQL queries**; vector databases use **similarity search** to retrieve data
- Relational databases organize data into **tables with rows and columns**

---

> *Author: Richa Arora*
