# Foundations of Generative AI & LangChain

Estimated time: 10 minutes

This cheat sheet covers:

- IBM Watsonx LLM
- Prompting techniques (basic, zero/one/few-shot)
- LangChain basics
- LCEL chaining

---

## 1. Install Required Packages

Installs Watsonx, LangChain, and core dependencies.

```bash
%%capture
!pip install "ibm-watsonx-ai==1.0.8" --user
!pip install "langchain==0.2.11" --user
!pip install "langchain-ibm==0.1.7" --user
!pip install "langchain-core==0.2.43" --user
2. Suppress Warnings
import warnings
warnings.filterwarnings('ignore')
3. Initialize Watsonx Granite Model
Creates a connection to IBM Granite LLM.

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
4. Helper Function
Reusable function to invoke the LLM.

def llm_model(prompt_txt, params=None):
    default_params = {
        "max_new_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.2
    }
    if params:
        default_params.update(params)

    granite_llm = WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=default_params
    )

    response = granite_llm.invoke(prompt_txt)
    return response
5. Generation Parameters
Shows IBM GenText parameters.

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Example values
GenParams().get_example_values()

parameters = {
    GenParams.MAX_NEW_TOKENS: 256,
    GenParams.TEMPERATURE: 0.5,
}
6. Prompting Techniques
6.1 Basic Prompt
params = {
    "max_new_tokens": 128,
    "min_new_tokens": 10,
    "temperature": 0.5,
    "top_p": 0.2,
    "top_k": 1
}
prompt = "The wind is"

response = llm_model(prompt, params)
print(f"Prompt: {prompt}")
print(f"Response: {response}")
6.2 Zero-Shot Prompt
No examples provided; model must follow instructions.

prompt = """Classify the following statement as true or false:
'The Eiffel Tower is located in Berlin.'
Answer:
"""

response = llm_model(prompt, params)
print(response)
6.3 One-Shot Prompt
One example is provided for pattern learning.

params = {"max_new_tokens": 20, "temperature": 0.1}
prompt = """Here is an example of translating English to French:

English: "How is the weather today?"
French: "Comment est le temps aujourd'hui?"

Now translate:
English: "Where is the nearest supermarket?"
"""

response = llm_model(prompt, params)
print(response)
6.4 Few-Shot Prompt
Multiple examples provided to show a pattern.

params = {"max_new_tokens": 10}
prompt = """Here are a few examples of classifying emotions:

Statement: I just won my first marathon!
Emotion: Joy

Statement: I can’t believe I lost my keys again.
Emotion: Frustration

Statement: My best friend is moving to another country.
Emotion: Sadness

Now classify:
Statement: That movie was so scary I had to cover my eyes.
"""

response = llm_model(prompt, params)
print(response)
7. LangChain LCEL Example
Combines PromptTemplate, RunnableLambda, and StrOutputParser.

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

template = """Tell me a {adjective} joke about {content}."""
prompt = PromptTemplate.from_template(template)

def format_prompt(variables):
    return prompt.format(**variables)

chain = (
    RunnableLambda(format_prompt)
    | granite_llm
    | StrOutputParser()
)

response = chain.invoke({"adjective": "funny", "content": "chickens"})
print(response)
✅ Notes
WatsonxLLM: Connects to IBM Granite models.

Params: Adjust max_new_tokens, temperature, top_p for creativity and length.

Prompting strategies: Zero-shot, One-shot, Few-shot improve control.

LCEL chains: Flexible, composable pipelines using | operator.


---


