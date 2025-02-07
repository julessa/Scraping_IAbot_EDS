import openai
import os
import sys

# ✅ Forcer UTF-8 pour éviter les erreurs d'encodage
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ✅ Vérification et chargement de la clé API OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ Clé API OpenAI non trouvée ! Assurez-vous qu'elle est définie dans les variables d'environnement.")

# ✅ Initialisation du client OpenAI (nouvelle API)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

try:
    # ✅ Test avec le modèle `text-embedding-3-small`
    response = client.embeddings.create(
        input="Test embedding",
        model="text-embedding-3-small"  # Modèle plus récent et accessible
    )
    
    # ✅ Affichage du résultat de l'embedding
    print("✅ OpenAI fonctionne, embedding généré :", response.data[0].embedding)

except openai.RateLimitError:
    print("🚨 Erreur : Tu as dépassé ton quota OpenAI. Vérifie ton crédit sur : https://platform.openai.com/account/usage")

except openai.AuthenticationError:
    print("🚨 Erreur : Clé API invalide. Vérifie que tu utilises une clé correcte.")

except openai.BadRequestError as e:
    print("🚨 Erreur OpenAI (requête invalide) :", e)

except Exception as e:
    print("❌ Erreur inconnue :", e)
