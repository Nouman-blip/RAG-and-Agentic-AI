# LangChain vs LlamaIndex

> **Estimated Reading Time: 10 minutes**

Both LangChain and LlamaIndex are frameworks for building LLM-powered applications like RAG pipelines, chatbots, document summarizers, and data extractors. This guide compares them across each stage of a typical RAG workflow.

---

## Quick Summary

| Feature | LangChain | LlamaIndex |
|---|---|---|
| Design Philosophy | Modular, flexible, integration-heavy | Simple, opinionated, great defaults |
| Document Loading | Many loaders, relies on external libs | `SimpleDirectoryReader` handles most formats natively |
| Chunking | Highly configurable splitters | `SentenceSplitter` works well out of the box |
| Embedding | Manual — embed then store separately | Auto — embed and store in one command |
| Vector Store | No single wrapper class | `VectorStoreIndex` wraps any backend |
| Metadata | May need manual setup | Auto-created and stored |
| Prompt Augmentation | Separate, easy to customize | Combined with LLM step, harder to customize |
| LLM Response | Manual — `llm.invoke(messages)` | Automatic via query engine or response synthesizer |

---

## RAG Pipeline Overview

```
Load & Chunk Docs → Embed Chunks → Store Vectors
     → Accept User Prompt → Embed Prompt
          → Retrieve Relevant Chunks → Augment Prompt
               → Pass to LLM → Get Response
```

---

## Step-by-Step Comparison

### 1. Loading Source Documents

**LangChain**
Provides many specialized loaders:
- `TextLoader` — plain text files
- `CSVLoader` — tabular CSV files
- `JSONLoader` — JSON data
- `WebBaseLoader` — web pages
- `DoclingLoader` — PDF, DOCX, PPTX, HTML
- `DirectoryLoader` — entire directories (uses `UnstructuredLoader` by default)

**LlamaIndex**
Provides `SimpleDirectoryReader` which natively handles markdown, PDF, Word, PowerPoint and more — for both single files and entire directories.
Extra connectors available at **LlamaHub**: `DatabaseReader`, `JSONReader`, `RssReader` and more.

> **Verdict:** LlamaIndex wins for simplicity out of the box. LangChain wins for flexibility and breadth of integrations.

---

### 2. Chunking Documents

**LangChain**
- `CharacterTextSplitter` — splits on a character sequence, limits by character length
- `TokenTextSplitter` — splits by token count
- `RecursiveCharacterTextSplitter` — recursively splits on a list of characters
- `MarkdownHeaderTextSplitter` — splits on markdown headings
- `SemanticChunker` — splits based on semantic similarity between sentences

**LlamaIndex**
- `SentenceSplitter` — token-based, comprehensive default splitting (similar to `RecursiveCharacterTextSplitter`)
- File-based node parsers for HTML, JSON, markdown, and code
- `SemanticSplitterNodeParser` — splits on semantic similarity
- `LangChainNodeParser` — wraps any LangChain splitter for use in LlamaIndex

> **Verdict:** Both are comprehensive. LlamaIndex's `SentenceSplitter` is a better default. LlamaIndex can also use LangChain splitters via a wrapper.

---

### 3. Embedding & Storing Vectors

**LangChain**
- Supports many embedding models (HuggingFace, OpenAI, etc.)
- Embeds chunks first, then stores in a vector store separately
- No single wrapper class — relies on integrations: `Chroma`, `FAISS`, `Milvus`, `PGVector`
- Offers more granular control over each vector store's unique features

**LlamaIndex**
- Supports many embedding models + wraps LangChain embedding models
- Embeds and stores in one command:
```python
index = VectorStoreIndex(nodes)
```
- `VectorStoreIndex` works in memory by default, swappable to Chroma, FAISS, etc.
- Metadata is automatically created and stored

> **Verdict:** LlamaIndex is simpler and more automated. LangChain gives more control.

---

### 4. Accepting the User Prompt

No significant implementation differences — both frameworks expect the prompt to be passed in from an external workflow or process.

---

### 5. Embedding the User Prompt

Both frameworks embed the user prompt using the same model stored inside the vector store object. Embedding is typically combined with retrieval, so there are no notable differences here.

---

### 6. Retrieving Relevant Chunks

Both support retrieval of the top-k most similar chunks. Both also offer advanced retrieval patterns — for example, LangChain's **parent document retriever** fetches the full document containing the matched chunk (requires two vector stores).

> For specialized retrieval needs, explore each framework's docs to find the best fit.

---

### 7. Augmenting the Prompt

**LangChain**
- Prompt augmentation is a separate, standalone step
- Easy to access and customize prompt templates

**LlamaIndex**
- Prompt augmentation is combined with the LLM response step (via **response synthesizer** or **query engine**)
- Default templates work well, but customization is more involved

---

### 8. Passing Augmented Prompt to the LLM

**LangChain**
Done manually:
```python
response = llm.invoke(messages)
```

**LlamaIndex**
Done automatically via:
- **Response Synthesizer** — takes user prompt + retrieved nodes, augments, and calls LLM internally
- **Query Engine** — single object that handles embedding, retrieval, augmentation, and LLM call all at once

---

## Conclusion

| | LangChain | LlamaIndex |
|---|---|---|
| Best for | Flexibility, custom pipelines, many integrations | Simplicity, quick setup, sensible defaults |
| Tradeoff | More manual setup required | Harder to customize internals |

Both frameworks are capable of handling most typical RAG workflows. Choose **LangChain** if you need fine-grained control and modular design. Choose **LlamaIndex** if you want to move fast with less boilerplate.

---

> *Author: Wojciech "Victor" Fulmyk*
