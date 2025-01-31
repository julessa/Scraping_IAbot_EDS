import openai
from openai import OpenAI

client = OpenAI(api_key="ta-clé-api")


try:
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Test"}])
    print("✅ OpenAI fonctionne :", response.choices[0].message.content)
except openai.RateLimitError:
    print("🚨 Ton compte est temporairement bloqué pour excès de requêtes.")
