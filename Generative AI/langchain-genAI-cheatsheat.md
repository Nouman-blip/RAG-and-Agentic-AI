# Cheat Sheet: Build GenAI Application With LangChain

> **Estimated time needed: 5 minutes**

---

## Quick Reference Table

| Package / Method | Description |
|---|---|
| `mkdir` & `cd` | Create and navigate into a new project directory |
| Virtual environment | Set up a Python virtual environment for package management |
| `pip install ibm-watsonx-ai` | Install the IBM watsonx AI library for LLM interactions |
| `Credentials` | Authenticate with IBM watsonx AI |
| Model parameters | Define parameters for model inference |
| `ModelInference` | Initialize an AI model for text generation |
| `model.generate()` | Use an AI model to generate text based on a prompt |
| `PromptTemplate` | Define reusable prompt templates for different models |
| LangChain chaining | Pipe a prompt template into an AI model for structured output |
| Tokenization | Specialized token formatting for different AI models |
| `JsonOutputParser` | Parse and structure AI-generated responses using LangChain |
| Flask API integration | Create an API endpoint for AI model interactions |

---

## 1. Project Setup

### Create & Navigate into Project Directory
```bash
mkdir genai_flask_app
cd genai_flask_app
```

### Set Up a Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install IBM watsonx AI Library
```bash
pip install ibm-watsonx-ai
```

---

## 2. Authentication — Credentials
```python
from ibm_watsonx_ai import Credentials

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    # api_key = ""
)
```

---

## 3. Model Parameters
```python
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}
```

---

## 4. Model Inference
```python
from ibm_watsonx_ai.foundation_models import ModelInference

model = ModelInference(
    model_id="ibm/granite-3-3-8b-instruct",
    params=params,
    credentials=credentials,
    project_id="skills-network"
)
```

---

## 5. Generating an AI Response
```python
text = """
Only reply with the answer. What is the capital of Canada?
"""
print(model.generate(text)['results'][0]['generated_text'])
```

---

## 6. LangChain Prompt Templates
```python
from langchain.prompts import PromptTemplate

llama3_template = PromptTemplate(
    template='''system
{system_prompt}user
{user_prompt}assistant
''',
    input_variables=["system_prompt", "user_prompt"]
)
```

---

## 7. LangChain Chaining
```python
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model
    return chain.invoke({'system_prompt': system_prompt, 'user_prompt': user_prompt})
```

---

## 8. Tokenization & Prompt Formatting
```python
# Llama 3 formatted prompt
text = """
system
You are an expert assistant who provides concise and accurate answers.
user
What is the capital of Canada?
assistant
"""
```

---

## 9. JSON Output Parser
```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 to 100")
    response: str = Field(description="Generated AI response")

json_parser = JsonOutputParser(pydantic_object=AIResponse)
```

---

## 10. Enhancing AI Outputs
```python
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'format_prompt': json_parser.get_format_instructions()
    })
```

---

## 11. Flask API Integration
```python
from flask import Flask, request, jsonify
from model import get_model_response

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    model_name = data.get('model')
    user_message = data.get('message')

    if not user_message or not model_name:
        return jsonify({"error": "Missing message or model selection"}), 400

    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a concise response."

    try:
        response = get_model_response(model_name, system_prompt, user_message)
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Full Pipeline at a Glance
```
Setup (venv + install)
 └── Credentials → ModelInference
      └── PromptTemplate → LangChain Chain
           └── JsonOutputParser → Structured Output
                └── Flask API Endpoint → Client Response
```

---

> **Tip:** Always use `JsonOutputParser` with a Pydantic schema in production
> to ensure consistent, structured responses from your AI model.2 / 2
