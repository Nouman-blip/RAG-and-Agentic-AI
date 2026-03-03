# Introduction to RAG

> **Estimated Reading Time: 10 minutes**

---

## What is RAG?

Retrieval-Augmented Generation (RAG) is a machine-learning technique that integrates information retrieval with generative AI to produce accurate and context-aware responses. By equipping generative models, such as large language models (LLMs), with access to external data sources, RAG enhances the model's ability to provide relevant and helpful answers to user prompts or queries.

---

## Why RAG?

LLMs without RAG cannot access external sources and are limited to the information provided in a prompt and their training data. Consequently, these LLMs are prone to generating responses that may be:

- Inaccurate
- Outdated
- Fabricated ("hallucinations")

Additionally, such responses often lack valid sources, making it difficult to trace their origin or verify accuracy.

In an RAG system, the key addition is **retrieved text**, sourced from an external data store. This retrieved text is combined with the original prompt to form an **augmented prompt**, which the LLM then processes to generate a grounded response.

### Key Benefits of RAG

| Benefit | Description |
|---|---|
| Enhanced response quality | Incorporating retrieved data delivers more accurate and detailed responses |
| Up-to-date information | External data provides current information beyond the model's static training data |
| Verification of sources | Citing external documents allows users to trace and verify information |

---

## What About Models with Long Context Lengths?

Models capable of handling context lengths up to 128,000 tokens or more are widely available today. To put this in perspective: roughly 100 tokens ≈ 75 words, meaning a 128,000-token model can process ~96,000 words of English text.

However, relying solely on long context lengths presents several limitations:

| Limitation | Description |
|---|---|
| Input Dependency | Users must already possess the necessary source information to include in the prompt |
| Limited Capacity | 128,000 tokens may still be insufficient for very long texts (e.g. War and Peace exceeds 560,000 words) |
| Redundancy Issues | Irrelevant or repeated information can dilute the LLM's focus — a "needle in a haystack" problem |
| Processing Time | More tokens = longer processing times |
| Cost Implications | Higher token usage increases both computational and financial costs |

### How RAG Addresses These Limitations

| Limitation | RAG Solution |
|---|---|
| Input Dependency | Connects to an external data store users don't need to provide themselves |
| Limited Capacity | Retrieves only the most relevant text, creating smaller augmented prompts |
| Redundancy Issues | Passes only the most pertinent information to the LLM |
| Processing Time | Shorter augmented prompts reduce response generation time |
| Cost Implications | Shorter prompts reduce response generation costs |

---

## How Does RAG Work?

The RAG process follows these core steps:

### Step-by-Step RAG Pipeline

```
Gather Sources
 └── Embed Sources (via embedding model)
      └── Store Vectors (in vector database)
           └── Obtain User Prompt
                └── Embed User Prompt (same embedding model)
                     └── Retrieve Relevant Data (via retriever)
                          └── Create Augmented Prompt
                               └── Obtain Response (via LLM)
```

### Detailed Steps

**1. Gather Sources**
Start with sources like office documents, company policies, or any other relevant information that may provide context for the user's future prompt.

**2. Embed Sources**
Pass the gathered information through an embedding model. The model converts each chunk of text into a vector — a fixed-length column of numbers representing semantic meaning.

**3. Store Vectors**
Store the embedded source vectors in a vector store — a specialized database optimized for storing and manipulating vector data.

**4. Obtain a User's Prompt**
Receive a prompt from the user, which may be standalone or incorporate prior conversation history.

**5. Embed the User's Prompt**
Embed the user's prompt using the **same embedding model** used for source documents to ensure compatibility.

**6. Retrieve Relevant Data**
Pass the prompt embedding to the retriever, which searches the vector store for relevant source embeddings and returns the matched text.

**7. Create an Augmented Prompt**
Combine the retrieved text with the user's original prompt to form an augmented prompt.

**8. Obtain a Response**
Feed the augmented prompt into an LLM, which processes it and produces a grounded response.

---

## RAG Details

### Gather Sources
- Preprocessing may be required before embedding — e.g. converting PDFs to plain text or using dynamic preprocessing libraries.

### Embed Sources
- **Chunking:** Large documents are split into smaller, manageable chunks for efficient retrieval.
- **Tokenization:** Text is split into tokens (words, parts of words, punctuation), each assigned a unique numerical ID.
- **Neural Network Processing:** Token IDs are processed by the embedding model's neural network, producing fixed-length vectors that capture semantic meaning.

### Store Vectors
- Simple systems use matrices; most production systems use specialized vector databases such as **ChromaDB**, **FAISS**, or **Milvus**.

### Obtain a User's Prompt
- If conversation history is included, tools like **LangChain** or **LlamaIndex** help augment the current prompt with relevant context.

### Retrieve Relevant Data
Common retrieval strategies include:
- Retrieving the single most relevant text chunk
- Fetching the entire document containing the relevant chunk
- Retrieving multiple relevant documents

### Create an Augmented Prompt
Common methods include:
- Simple concatenation of the user's prompt with retrieved text
- Structured templates where user input, retrieved text, and LLM instructions are placed in defined sections

### Obtain a Response
- Responses can be further refined using predefined templates to ensure consistent presentation tailored to the use case.

---

> **Note:** The many intricacies of each RAG step contribute to a wide range of implementation possibilities, enabling customization to suit various applications and needs.
