# 🤖 Agentic AI Protocols
### Complete Reference Guide — All Major Agent Communication Standards

---

## 📌 Table of Contents

1. [What Are AI Agent Protocols?](#what-are-ai-agent-protocols)
2. [Why Protocols Exist](#why-protocols-exist)
3. [All 6 Major Protocols](#all-6-major-protocols)
   - [ACP — Agent Communication Protocol](#1-acp--agent-communication-protocol)
   - [ANP — Agent Network Protocol](#2-anp--agent-network-protocol)
   - [AG-UI — Agent-User Interaction Protocol](#3-ag-ui--agent-user-interaction-protocol)
   - [A2A — Agent2Agent Protocol](#4-a2a--agent2agent-protocol)
   - [MCP — Model Context Protocol](#5-mcp--model-context-protocol)
   - [AP2 — Agent Payments Protocol](#6-ap2--agent-payments-protocol)
4. [How A2A + MCP + AP2 Work Together](#how-a2a--mcp--ap2-work-together)
5. [How to Choose a Protocol](#how-to-choose-a-protocol)
6. [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## What Are AI Agent Protocols?

AI agent protocols establish **standards of communication** among artificial intelligence agents and between AI agents and other systems.

They define:
- **Syntax** — how messages are structured
- **Sequence** — when and in what order messages are sent
- **Roles** — who speaks, who listens, who acts
- **Conventions** — how agents respond to each other

> ⚠️ **Important distinction:** Protocols standardize communication — they do NOT act as orchestrators. Managing workflow coordination, execution, and optimization is a separate concern.

---

## Why Protocols Exist

### The Problem

Agent-based AI systems often run in **silos**:
- Built by different providers
- Using diverse AI agent frameworks
- Employing distinct agentic architectures

Real-world integration becomes a challenge. Coupling these fragmented systems requires tailored connectors for every possible type of agent interaction — an unscalable approach.

### The Solution

Protocols turn **disparate multi-agent systems** into an **interlinked ecosystem** where agents share a standard way of:
- Discovering each other
- Understanding each other's capabilities
- Collaborating across tasks

### Current State of Maturity

> ⚠️ Many protocols are still in early stages — not yet widely deployed at scale. Organizations adopting them now are **early adopters** and must be prepared for breaking changes and evolving specifications.

---

## All 6 Major Protocols

---

### 1. ACP — Agent Communication Protocol

| Field | Detail |
|---|---|
| **Created by** | IBM (BeeAI) |
| **Type** | Agent-to-agent communication |
| **Default mode** | Asynchronous |
| **Best for** | Complex, long-running tasks |

**What it does:**
ACP is an open standard that enables AI agents to **collaborate freely** across teams, frameworks, technologies, and organizations. It transforms siloed agents into interoperable teammates.

**Position in the stack:**
ACP is considered the **next step after MCP** — while MCP standardizes tool and data access, ACP defines how agents themselves operate and communicate.

🔗 https://www.ibm.com/think/topics/agent-communication-protocol

---

### 2. ANP — Agent Network Protocol

| Field | Detail |
|---|---|
| **Type** | Open-source agent communication framework |
| **Analogy** | Like HTTP, but for AI agents |
| **Scope** | Internet-wide agent discovery and interaction |

**What it does:**
ANP allows agents to **locate, connect with, and interact** across the internet in an open and secure environment — laying groundwork for an AI-driven internet where agents are the primary operating entities.

**Core problem it solves:**
The lack of a standardized, secure, and efficient method for agent-to-agent communication at internet scale.

🔗 https://www.agent-network-protocol.com/guide/

---

### 3. AG-UI — Agent-User Interaction Protocol

| Field | Detail |
|---|---|
| **Type** | Event-based open protocol |
| **Focus** | AI agent ↔ user-facing application communication |
| **Design goal** | Ease of use and adaptability |

**What it does:**
AG-UI standardizes the connection between AI agents and **frontend applications**. It provides a consistent structure for exchanging:
- Agent state
- UI intents
- User interactions

**Developer benefit:**
Build and deploy agent-driven UI features without complex custom integrations — focus on core application functionality instead.

🔗 https://docs.ag-ui.com/introduction

---

### 4. A2A — Agent2Agent Protocol

| Field | Detail |
|---|---|
| **Created by** | Google → now managed by Linux Foundation |
| **Model** | Client-server |
| **Transport** | HTTPS |
| **Data format** | JSON-RPC 2.0 |
| **Real-time support** | Yes — SSE (Server-Sent Events) |

**What it does:**
A2A is an open standard for AI agent communication that follows a **three-step workflow:**

```
Step 1 — DISCOVERY
User or agent initiates a task → client agent looks up
remote agents to find the best fit

        ↓

Step 2 — AUTHENTICATION
Client agent identifies the right remote agent →
remote agent handles authorization and access control

        ↓

Step 3 — COMMUNICATION
Client agent sends the task →
remote agent processes it over HTTPS with JSON-RPC 2.0
```

🔗 https://www.ibm.com/think/topics/agent2agent-protocol

---

### 5. MCP — Model Context Protocol

| Field | Detail |
|---|---|
| **Created by** | Anthropic |
| **Type** | Tool and data access layer for AI models |
| **Data format** | JSON-RPC 2.0 |
| **Transport options** | stdio (local) / HTTP (remote) |

**What it does:**
MCP provides a standardized way for AI models to get the **context they need to carry out tasks** — connecting agents to APIs, databases, files, web searches, and other data sources.

**Three architectural elements:**

| Component | Role |
|---|---|
| **MCP Host** | Contains orchestration logic; connects MCP clients to servers; can host multiple clients |
| **MCP Client** | Converts user requests into protocol-ready format; 1-to-1 relationship with a server; manages sessions, parses responses, handles errors |
| **MCP Server** | Converts requests into server actions; typically GitHub repos in various languages; provides access to tools and LLM inferencing |

**Transport layer:**

```
Client ←──── stdio (lightweight, synchronous, local) ────→ Server
Client ←──── HTTP (remote requests, cloud-hosted)   ────→ Server
```

🔗 https://www.ibm.com/think/topics/model-context-protocol

---

### 6. AP2 — Agent Payments Protocol

| Field | Detail |
|---|---|
| **Created by** | Google (in collaboration with payments/tech sector) |
| **Announced** | September 2025 |
| **Extends** | A2A and MCP |
| **Core mechanism** | Cryptographically signed digital mandates |

**What it does:**
AP2 enables **secure, cross-platform payments initiated by AI agents** — for scenarios where autonomous agents make purchases on behalf of users, without a human clicking "Buy Now."

**Two types of mandates:**

#### Intent Mandate
Captures the user's request in a verifiable way — the authorization record for what the user asked for.

#### Cart Mandate
A tamper-proof record of exactly what was approved — items, price, and terms — created when a transaction is finalized.

**Two purchase scenarios:**

| Scenario | How It Works |
|---|---|
| **User-involved** (real-time) | User asks agent to buy → Intent Mandate created → agent presents cart → user confirms → Cart Mandate created → payment processed |
| **Delegated** (autonomous) | User provides pre-signed Intent Mandate with spending caps/criteria → agent acts when criteria are met → agent generates Cart Mandate → purchase completed |

**AP2 guarantees every transaction:**
- Reflects the user's intent
- Operates within defined boundaries
- Is traceable and secure

🔗 https://cloud.google.com/blog/products/ai-machine-learning/announcing-agents-to-payments-ap2-protocol

---

## How A2A + MCP + AP2 Work Together

These three protocols form the **framework for agentic commercial transactions.**

### Example: "Order me wireless noise-cancelling headphones under $250"

```
┌─────────────────────────────────────────────────────────┐
│  User: "Order me wireless headphones, under $250"       │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │        A2A AT WORK      │
          │  Shopping agent talks   │
          │  to retailer's product  │
          │  agent + payment agent  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │        MCP AT WORK      │
          │  Agent gathers product  │
          │  details, user prefs,   │
          │  and purchase history   │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │        AP2 AT WORK      │
          │  Agent creates Cart     │
          │  Mandate → user reviews │
          │  → payment processed    │
          └─────────────────────────┘
```

**Division of responsibility:**

| Protocol | Responsibility |
|---|---|
| **A2A** | Agent-to-agent communication and task coordination |
| **MCP** | Gathering context (data, tools, external services) |
| **AP2** | Secure, authorized payment execution |

---

## How to Choose a Protocol

With no standardized benchmarks yet, enterprises must evaluate protocols against their own needs. Key criteria:

### Efficiency
- Protocols should minimize latency
- Communication overhead must be kept to a minimum
- Look for: fast data transfer, rapid response times

### Reliability
- Must handle changing network conditions throughout agentic workflows
- Must have mechanisms for failure and disruption recovery
- **ACP** — async by default (good for long-running tasks)
- **A2A** — real-time streaming via SSE (good for continuous updates)

### Scalability
- Must support growing agent ecosystems without performance degradation
- Test by gradually or suddenly increasing agents and external tool connections

### Security
- Authentication, encryption, and access control are non-negotiable
- Protocols are increasingly incorporating safety guardrails

---

## Quick Reference Cheat Sheet

```
┌──────┬──────────────────────┬──────────────────────────────────┐
│ ID   │ Protocol             │ Core Purpose                     │
├──────┼──────────────────────┼──────────────────────────────────┤
│ ACP  │ Agent Communication  │ Agent-to-agent interoperability  │
│      │ Protocol             │ across orgs and frameworks       │
├──────┼──────────────────────┼──────────────────────────────────┤
│ ANP  │ Agent Network        │ Internet-scale agent discovery   │
│      │ Protocol             │ and connection (like HTTP)       │
├──────┼──────────────────────┼──────────────────────────────────┤
│ AG-UI│ Agent-User           │ Standard agent ↔ frontend UI     │
│      │ Interaction Protocol │ communication layer              │
├──────┼──────────────────────┼──────────────────────────────────┤
│ A2A  │ Agent2Agent          │ Discovery, auth, and secure      │
│      │ Protocol             │ task communication between agents│
├──────┼──────────────────────┼──────────────────────────────────┤
│ MCP  │ Model Context        │ Standardized tool and data       │
│      │ Protocol             │ access for AI models             │
├──────┼──────────────────────┼──────────────────────────────────┤
│ AP2  │ Agent Payments       │ Secure, autonomous, agent-       │
│      │ Protocol             │ initiated payment transactions   │
└──────┴──────────────────────┴──────────────────────────────────┘

KEY RELATIONSHIPS:
  MCP  → what tools/data the agent can ACCESS
  A2A  → how agents COMMUNICATE with each other
  ACP  → how agents COLLABORATE across organizations
  ANP  → how agents FIND each other on the internet
  AG-UI→ how agents TALK to user interfaces
  AP2  → how agents make PAYMENTS on your behalf

COMMERCIAL TRANSACTION STACK:
  A2A (coordinate) + MCP (gather context) + AP2 (pay) = Full agentic commerce
```

---

*Source: IBM Cognitive Class — Agentic AI Protocols Reading*
