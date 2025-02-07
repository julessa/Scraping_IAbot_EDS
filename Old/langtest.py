import openai
from langsmith import wrappers, traceable
from langchain_openai import ChatOpenAI


# Auto-trace LLM calls in-context
client = wrappers.wrap_openai(openai.Client())
llm = ChatOpenAI()  # Instanciation du modèle LangChain

@traceable  # Auto-trace cette fonction
def pipeline_openai(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

@traceable  # Auto-trace cette fonction aussi
def pipeline_langchain(user_input: str):
    response = llm.invoke(user_input)
    return response.content  # On extrait seulement la réponse du modèle

# Exécution des pipelines
response_openai = pipeline_openai("Hello, world!")
response_langchain = pipeline_langchain("Hello, world!")

# Affichage des résultats
print(f"Pipeline OpenAI: {response_openai}")
print(f"Pipeline LangChain: {response_langchain}")