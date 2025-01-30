import openai

openai.api_key = "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"  # Remplace par ta clé API

try:
    # Utilisation de l'API chat avec le modèle gpt-3.5-turbo
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(completion['choices'][0]['message']['content'])

except openai.error.RateLimitError:
    print("Quota dépassé, vérifie tes crédits sur OpenAI.")
except openai.error.OpenAIError as e:
    print(f"Erreur OpenAI : {e}")
