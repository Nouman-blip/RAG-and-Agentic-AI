# 🚀 Build Smarter AI Apps: Empower LLMs with LangChain

> **Estimated time needed:** 60 minutes

---

## 📋 Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Core LangChain Concepts](#core-langchain-concepts)
3. [Prompt Engineering](#prompt-engineering)
4. [Document Processing](#document-processing)
5. [Memory & Conversation](#memory--conversation)
6. [Chains](#chains)
7. [Tools & Agents](#tools--agents)

---

## 🎯 Learning Objectives

After completing this guide, you will be able to:

- Use core features of LangChain (prompt templates, chains, agents)
- Explore LangChain's modular approach for dynamic adjustments
- Enhance LLM applications with Retrieval-Augmented Generation (RAG)
- Build contextually-aware, intelligent AI applications

---

## 🔧 Setup & Installation

### Install Required Libraries

```bash
%%capture
!pip install --force-reinstall --no-cache-dir tenacity==8.2.3 --user
!pip install "ibm-watsonx-ai==1.0.8" --user
!pip install "ibm-watson-machine-learning==1.0.367" --user
!pip install "langchain-ibm==0.1.7" --user
!pip install "langchain-community==0.2.10" --user
!pip install "langchain-experimental==0.0.62" --user
!pip install "langchainhub==0.1.18" --user
!pip install "langchain==0.2.11" --user
!pip install "pypdf==4.2.0" --user
!pip install "chromadb==0.4.24" --user
```

### Restart Kernel

```python
import os
os._exit(00)
```

### Import Required Libraries

```python
# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# IBM Watsonx imports
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
```

---

## 🧠 Core LangChain Concepts

### 1. Setting Up the Model

#### Configure Model Parameters

```python
model_id = 'meta-llama/llama-3-3-70b-instruct'

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.2,
}

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
    # "api_key": "your api key here"  # Uncomment for local use
}

project_id = "skills-network"

model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id
)
```

#### Wrap Model for LangChain

```python
llama_llm = WatsonxLLM(model=model)
```

#### Test Basic Generation

```python
msg = model.generate("In today's sales meeting, we ")
print(msg['results'][0]['generated_text'])
```

---

### 2. Chat Messages

LangChain supports different message types:

- **SystemMessage** - Prime AI behavior
- **HumanMessage** - User messages
- **AIMessage** - AI responses

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Simple chat
msg = llama_llm.invoke([
    SystemMessage(content="You are a helpful AI bot"),
    HumanMessage(content="I enjoy mystery novels, what should I read?")
])

# Chat with history
msg = llama_llm.invoke([
    SystemMessage(content="You are a supportive AI bot"),
    HumanMessage(content="I like high-intensity workouts, what should I do?"),
    AIMessage(content="You should try a CrossFit class"),
    HumanMessage(content="How often should I attend?")
])
```

---

## 💬 Prompt Engineering

### String Prompt Templates

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}

prompt.invoke(input_)
```

### Chat Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

input_ = {"topic": "cats"}
prompt.invoke(input_)
```

### MessagesPlaceholder

For inserting multiple messages dynamically:

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("msgs")
])

input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}
prompt.invoke(input_)
```

---

## 🔄 Output Parsers

### JSON Output Parser

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define data structure
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

# Create parser
output_parser = JsonOutputParser(pydantic_object=Joke)
format_instructions = output_parser.get_format_instructions()

# Create prompt
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)

# Create chain
chain = prompt | llama_llm | output_parser
chain.invoke({"query": "Tell me a joke."})
```

### CSV List Parser

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

prompt = PromptTemplate(
    template="Answer the user query. {format_instructions}\nList five {subject}.",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | llama_llm | output_parser
chain.invoke({"subject": "ice cream flavors"})
```

---

## 📄 Document Processing

### Document Object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Python is an interpreted high-level programming language.",
    metadata={
        'my_document_id': 234234,
        'my_document_source': "About Python",
        'my_document_create_time': 1680013019
    }
)
```

### Loading Documents

#### PDF Loader

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("https://example.com/paper.pdf")
document = loader.load()

# Access specific page
print(document[2])  # Page 2
print(document[1].page_content[:1000])  # First 1000 chars of page 1
```

#### Web Loader

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://python.langchain.com/v0.2/docs/introduction/")
web_data = loader.load()
print(web_data[0].page_content[:1000])
```

### Text Splitters

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separator="\n"
)

chunks = text_splitter.split_documents(document)
print(len(chunks))
print(chunks[5].page_content)
```

---

## 🔍 Embeddings & Vector Stores

### Create Embeddings

```python
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params
)

texts = [text.page_content for text in chunks]
embedding_result = watsonx_embedding.embed_documents(texts)
```

### Vector Store with Chroma

```python
from langchain.vectorstores import Chroma

docsearch = Chroma.from_documents(chunks, watsonx_embedding)

# Similarity search
query = "Langchain"
docs = docsearch.similarity_search(query)
print(docs[0].page_content)
```

---

## 🔎 Retrievers

### Vector Store-backed Retriever

```python
retriever = docsearch.as_retriever()
docs = retriever.invoke("Langchain")
print(docs[0])
```

### Parent Document Retriever

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

# Parent splitter (larger chunks)
parent_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
    separator='\n'
)

# Child splitter (smaller chunks for search)
child_splitter = CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    separator='\n'
)

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=watsonx_embedding
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

retriever.add_documents(document)
```

### RetrievalQA

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llama_llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=False
)

query = "what is this paper discussing?"
qa.invoke(query)
```

---

## 💭 Memory & Conversation

### Chat Message History

```python
from langchain.memory import ChatMessageHistory

history = ChatMessageHistory()
history.add_ai_message("hi!")
history.add_user_message("what is the capital of France?")

# View messages
print(history.messages)

# Get AI response
ai_response = llama_llm.invoke(history.messages)
history.add_ai_message(ai_response)
```

### Conversation Buffer Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=llama_llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Have a conversation
conversation.invoke(input="Hello, I am a little cat. Who are you?")
conversation.invoke(input="What can you do?")
conversation.invoke(input="Who am I?")
```

---

## ⛓️ Chains

### Traditional Approach: LLMChain

```python
from langchain.chains import LLMChain

template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}
YOUR RESPONSE:
"""

prompt_template = PromptTemplate(template=template, input_variables=['location'])
location_chain = LLMChain(llm=llama_llm, prompt=prompt_template, output_key='meal')

location_chain.invoke(input={'location': 'China'})
```

### Modern Approach: LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """Your job is to come up with a classic dish from the area that the users suggests.
{location}
YOUR RESPONSE:
"""

prompt = PromptTemplate.from_template(template)

# LCEL chain with pipe operator
location_chain_lcel = prompt | llama_llm | StrOutputParser()

result = location_chain_lcel.invoke({"location": "China"})
```

### Sequential Chains

#### Traditional Sequential Chain

```python
from langchain.chains import SequentialChain

# Chain 1: Location -> Meal
location_chain = LLMChain(llm=llama_llm, prompt=location_prompt, output_key='meal')

# Chain 2: Meal -> Recipe
dish_chain = LLMChain(llm=llama_llm, prompt=dish_prompt, output_key='recipe')

# Chain 3: Recipe -> Time
recipe_chain = LLMChain(llm=llama_llm, prompt=time_prompt, output_key='time')

# Combine all chains
overall_chain = SequentialChain(
    chains=[location_chain, dish_chain, recipe_chain],
    input_variables=['location'],
    output_variables=['meal', 'recipe', 'time'],
    verbose=True
)

overall_chain.invoke(input={'location': 'China'})
```

#### LCEL Sequential Chain

```python
from langchain_core.runnables import RunnablePassthrough

# Define individual chains
location_chain_lcel = PromptTemplate.from_template(location_template) | llama_llm | StrOutputParser()
dish_chain_lcel = PromptTemplate.from_template(dish_template) | llama_llm | StrOutputParser()
time_chain_lcel = PromptTemplate.from_template(time_template) | llama_llm | StrOutputParser()

# Combine using RunnablePassthrough
overall_chain_lcel = (
    RunnablePassthrough.assign(
        meal=lambda x: location_chain_lcel.invoke({"location": x["location"]})
    )
    | RunnablePassthrough.assign(
        recipe=lambda x: dish_chain_lcel.invoke({"meal": x["meal"]})
    )
    | RunnablePassthrough.assign(
        time=lambda x: time_chain_lcel.invoke({"recipe": x["recipe"]})
    )
)

result = overall_chain_lcel.invoke({"location": "China"})
```

---

## 🛠️ Tools & Agents

### Creating Tools

#### Using Tool Class

```python
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

python_calculator = Tool(
    name="Python Calculator",
    func=python_repl.run,
    description="Useful for calculations. Input should be valid Python code."
)

# Test the tool
python_calculator.invoke("a = 3; b = 1; print(a+b)")
```

#### Using @tool Decorator

```python
from langchain.tools import tool

@tool
def search_weather(location: str):
    """Search for the current weather in the specified location."""
    return f"The weather in {location} is currently sunny and 72°F."
```

### Creating Agents

#### ReAct Agent Setup

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# Create tools list
tools = [python_calculator, search_weather]

# Define ReAct prompt
prompt_template = """You are an agent who has access to the following tools:

{tools}

The available tools are: {tool_names}

To use a tool, use this format:
```
Thought: I need to figure out what to do
Action: tool_name
Action Input: the input to the tool
```

After using a tool:
```
Observation: result of the tool
```

When you have the answer:
```
Thought: I know the answer
Final Answer: the final answer
```

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Create agent
agent = create_react_agent(
    llm=llama_llm,
    tools=tools,
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
```

#### Using the Agent

```python
# Single query
result = agent_executor.invoke({"input": "What is the square root of 256?"})
print(result["output"])

# Multiple queries
queries = [
    "What's 345 * 789?",
    "Calculate the square root of 144",
    "What's the weather in Miami?",
]

for query in queries:
    result = agent_executor.invoke({"input": query})
    print(f"Query: {query}")
    print(f"Answer: {result['output']}\n")
```

---

## 🎓 Key Takeaways

### LCEL vs Traditional Chains

| Feature | Traditional | LCEL |
|---------|------------|------|
| **Syntax** | Explicit chain classes | Pipe operator `\|` |
| **Flexibility** | More rigid | Highly composable |
| **Readability** | More verbose | Cleaner, functional |
| **Debugging** | Standard | Better visualization |
| **Recommendation** | Legacy support | ✅ Recommended |

### Best Practices

1. **Use LCEL** for new development (pipe operator `|`)
2. **Start simple** - Add complexity as needed
3. **Use retrievers** for document-based applications
4. **Implement memory** for conversational agents
5. **Chain wisely** - Break complex tasks into steps
6. **Test tools** individually before adding to agents
7. **Use verbose mode** for debugging chains and agents

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [IBM Watsonx.ai](https://www.ibm.com/watsonx)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

---

*Happy building with LangChain! 🎉*
