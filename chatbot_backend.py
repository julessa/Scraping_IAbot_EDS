from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

# Définition de l'application FastAPI
app = FastAPI()

# Ajouter CORS pour permettre les requêtes du frontend (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" par ["http://localhost:3000"] si nécessaire
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger la clé API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("🚨 Clé API OpenAI non trouvée. Définissez OPENAI_API_KEY dans vos variables d'environnement.")

# Modèle de requête pour l'API
class ChatRequest(BaseModel):
    question: str

# Route de l'API pour le chatbot
@app.post("/chat")
async def chat(request: ChatRequest):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request.question}],
        api_key=openai_api_key
    )
    return {"response": response["choices"][0]["message"]["content"]}
