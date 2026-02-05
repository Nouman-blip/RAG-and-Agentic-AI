# 🚀 Foundations of Generative AI and LangChain

> **Estimated time needed:** 10 minutes

---

## 📦 Installation & Setup

### Package Installation

Install the necessary Python libraries required for the course:

```bash
%%capture
!pip install "ibm-watsonx-ai==1.0.8" --user
!pip install "langchain==0.2.11" --user
!pip install "langchain-ibm==0.1.7" --user
!pip install "langchain-core==0.2.43" --user
```

### Suppress Warnings

Keep your output clean by suppressing warnings:

```python
import warnings
warnings.filterwarnings('ignore')
```

---

## 🤖 IBM Watsonx LLM Integration

### WatsonxLLM Setup

Facilitates interaction with IBM's Watsonx large language models:

```python
from langchain_ibm import WatsonxLLM

granite_llm = WatsonxLLM(
    model_id="ibm/granite-3-2-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={
        "max_new_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.2
    }
)
```

### Custom LLM Model Function

Invokes IBM Watsonx LLM with custom prompts and parameters:

```python
def llm_model(prompt_txt, params=None):
    model_id = "ibm/granite-3-2-8b-instruct"
    default_params = {
        "max_new_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.2
    }
    if params:
        default_params.update(params)
    
    granite_llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=default_params
    )
    
    response = granite_llm.invoke(prompt_txt)
    return response
```

### GenParams - Text Generation Parameters

Control text generation with various parameters:

```python
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Get example values
GenParams().get_example_values()

# Use in parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}
```

**Available parameters:**
- `max_new_tokens` - Maximum tokens to generate
- `min_new_tokens` - Minimum tokens to generate
- `temperature` - Controls randomness (0-1)
- `top_p` - Nucleus sampling threshold
- `top_k` - Top-k sampling parameter

---

## 💡 Prompting Techniques

### 1. Basic Prompt

The simplest form of prompting - provide text and let the model continue:

```python
params = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}

prompt = "The wind is"

response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response: {response}\n")
```

---

### 2. Zero-shot Prompt

Perform tasks without examples or prior specific training:

```python
prompt = """Classify the following statement as true or false: 
            'The Eiffel Tower is located in Berlin.'
        Answer:
"""

response = llm_model(prompt, params)
print(f"prompt: {prompt}\n")
print(f"response: {response}\n")
```

**Use case:** Testing the model's ability to understand instructions and apply knowledge to new contexts.

---

### 3. One-shot Prompt

Provide a single example to establish a pattern:

```python
params = {
    "max_new_tokens": 20,
    "temperature": 0.1,
}

prompt = """Here is an example of translating a sentence from English to French:

        English: "How is the weather today?"
        French: "Comment est le temps aujourd'hui?" 
        
        Now, translate the following sentence from English to French:            
        English: "Where is the nearest supermarket?"           
"""

response = llm_model(prompt, params)
```

**Use case:** Improving understanding of desired output format and style.

---

### 4. Few-shot Prompt

Provide multiple examples (2-5) for clearer pattern recognition:

```python
params = {
    "max_new_tokens": 10,
}

prompt = """Here are few examples of classifying emotions in statements:
            
            Statement: 'I just won my first marathon!'
            Emotion: Joy
            
            Statement: 'I can't believe I lost my keys again.'
            Emotion: Frustration
            
            Statement: 'My best friend is moving to another country.'
            Emotion: Sadness
            
            Now, classify the emotion in the following statement:
            Statement: 'That movie was so scary I had to cover my eyes.'
"""

response = llm_model(prompt, params)
```

**Use case:** Establishing clearer patterns for better understanding of expected outputs.

---

### 5. Chain-of-Thought (CoT) Prompting

Break down complex problems into step-by-step reasoning:

```python
params = {
    "max_new_tokens": 512,
    "temperature": 0.5,
}

prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today 
            and got a new delivery of 8 apples. How many apples are there now?'

        Break down each step of your calculation
"""

response = llm_model(prompt, params)
```

**Use case:** Improving problem-solving abilities and reducing errors in multi-step reasoning tasks.

---

### 6. Self-consistency

Generate multiple solutions and determine the most consistent result:

```python
params = {
    "max_new_tokens": 512,
}

prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

        Provide three independent calculations and explanations, 
        then determine the most consistent result.
"""

response = llm_model(prompt, params)
```

**Use case:** Improving accuracy by leveraging multiple approaches to the same problem.

---

## 🔗 LangChain Components

### PromptTemplate

Create reusable prompt structures with dynamic values:

```python
from langchain_core.prompts import PromptTemplate

template = """Tell me a {adjective} joke about {content}."""
prompt = PromptTemplate.from_template(template)

# Format the prompt
formatted_prompt = prompt.format(
    adjective="funny",
    content="chickens"
)
```

**Use case:** Define consistent formats with placeholders for variable content.

---

### RunnableLambda

Wrap Python functions into LangChain runnable components:

```python
from langchain_core.runnables import RunnableLambda

# Define a function to ensure proper formatting
def format_prompt(variables):
    return prompt.format(**variables)

# Use in a chain
joke_chain = (
    RunnableLambda(format_prompt)
    | llm
    | StrOutputParser()
)
```

**Use case:** Create transformation steps in chains for formatting or processing data.

---

### StrOutputParser

Extract clean string outputs from LLM responses:

```python
from langchain_core.output_parsers import StrOutputParser

# Create a chain that returns a string
chain = (
    RunnableLambda(format_prompt)
    | llm
    | StrOutputParser()
)

# Run the chain
response = chain.invoke({"variable": "value"})
```

**Use case:** Ensure clean string outputs as the final step in a chain.

---

## ⛓️ LCEL Pattern (LangChain Expression Language)

Build LangChain applications using the pipe operator (`|`) for flexible composition:

### Basic LCEL Pattern

```python
chain = (
    RunnableLambda(format_prompt)  # Format input
    | llm                         # Process with LLM
    | StrOutputParser()           # Parse output
)

# Run the chain
result = chain.invoke({"variable": "value"})
```

### Complex LCEL Example

```python
template = """
    Answer the {question} based on the {content}.
    Respond "Unsure about answer" if not sure.

Answer:
"""
prompt = PromptTemplate.from_template(template)

qa_chain = (
    RunnableLambda(format_prompt)
    | llm
    | StrOutputParser()
)

answer = qa_chain.invoke({
    "question": "Which planets are rocky?",
    "content": "The inner planets are rocky."
})
```

### Benefits of LCEL

✅ **Better composability** - Easily combine components  
✅ **Clearer visualization** - See data flow at a glance  
✅ **More flexibility** - Construct complex chains efficiently

---

## 📚 Quick Reference Summary

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Basic Prompt** | Simple text continuation | Quick responses without specific structure |
| **Zero-shot** | Task without examples | Testing model's instruction understanding |
| **One-shot** | Single example provided | Establishing output format |
| **Few-shot** | Multiple examples (2-5) | Complex pattern recognition |
| **Chain-of-Thought** | Step-by-step reasoning | Multi-step problem solving |
| **Self-consistency** | Multiple independent solutions | Improving accuracy on complex tasks |

---

## 🎯 Best Practices

1. **Start simple** - Begin with basic prompts and add complexity as needed
2. **Be specific** - Clear instructions lead to better results
3. **Iterate** - Refine prompts based on model responses
4. **Use examples** - Few-shot prompting improves performance
5. **Chain wisely** - Use LCEL for maintainable, composable chains
6. **Control parameters** - Adjust temperature and tokens for optimal results

---

*Happy coding! 🎉*
