import json
import os
import time
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from tqdm import tqdm
from openai import OpenAIError

# 📂 Nom du fichier pour sauvegarder l'historique
HISTORY_FILE = "chat_history.json"

# 📌 Charger l'historique des conversations si le fichier existe
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 📌 Sauvegarder l'historique mis à jour
def save_chat_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

# 📌 Initialiser l'historique
chat_history = load_chat_history()

# 1️⃣ Vérifier la clé API d'OpenAI
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    print("❌ Erreur : Clé API OpenAI manquante. Définissez la variable d'environnement OPENAI_API_KEY.")
    exit()

# 2️⃣ Charger les données du JSON
try:
    with open("filtered_output.json", "r", encoding="utf-8") as f:
        data = json.load(f)
except FileNotFoundError:
    print("❌ Erreur : Fichier JSON introuvable.")
    exit()

# 3️⃣ Filtrer les doublons et transformer en documents LangChain avec affichage de progression
docs = []
for entry in tqdm(data, desc="Indexation des événements..."):
    if not entry.get("duplicate", False):
        text = f"{entry.get('date', 'Date inconnue')} : {entry.get('event', 'Événement non spécifié')}"
        docs.append(Document(page_content=text))

# 4️⃣ Fractionner les textes pour l'indexation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 5️⃣ Vérifier si FAISS est bien installé
try:
    import faiss
except ModuleNotFoundError:
    print("❌ Erreur : FAISS n'est pas installé. Exécutez `pip install faiss-cpu` ou `pip install faiss-gpu`.")
    exit()

# 6️⃣ Vérifier si l'index FAISS existe déjà
index_path = "faiss_index"
if os.path.exists(index_path):
    print("🔄 Chargement de l'index FAISS existant...")
    try:
        vector_store = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de FAISS : {e}")
        exit()
else:
    print("⚡ Aucun index trouvé, création d'un nouvel index...")
    vector_store = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    vector_store.save_local(index_path)

# 7️⃣ Créer un retriever pour interroger les données historiques
retriever = vector_store.as_retriever()

# 8️⃣ Configurer le modèle OpenAI via LangChain
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # ou "gpt-4"
except OpenAIError as e:
    print(f"❌ Erreur de configuration OpenAI : {e}")
    exit()

# 9️⃣ Construire la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 🔟 Fonction pour interroger le chatbot avec gestion de l'historique
def chat_with_bot(query, max_retries=3, wait_time=5):
    """
    Interroge le chatbot avec gestion du rate limit et sauvegarde des conversations.
    """
    for attempt in range(max_retries):
        try:
            response = qa_chain.invoke({"query": query})  
            time.sleep(1)  # ✅ Ajout d'une pause pour limiter la fréquence des requêtes
            answer = response.get("result", "⚠️ Aucune réponse disponible.")

            # 📌 Ajouter la conversation à l'historique
            chat_history.append({"question": query, "response": answer})
            save_chat_history(chat_history)  # Sauvegarde après chaque interaction

            return answer
        except openai.RateLimitError:
            print(f"⚠️ API Rate Limit atteinte. Attente de {wait_time} secondes avant de réessayer...")
            time.sleep(wait_time)  
        except OpenAIError as e:
            print(f"❌ Erreur OpenAI : {e}")
            return "⚠️ Une erreur est survenue avec l'API OpenAI."
        except Exception as e:
            print(f"❌ Erreur inattendue : {e}")
            return "⚠️ Une erreur inattendue est survenue."

    return "⚠️ Désolé, trop de requêtes envoyées à OpenAI. Réessayez plus tard."

# 🔥 Boucle interactive pour discuter avec le chatbot
print("\n🤖 Chatbot Historique - Tapez 'exit' ou 'quit' pour quitter.\n")

# 📌 Afficher l'historique si disponible
if chat_history:
    print("📜 Historique des conversations précédentes :")
    for i, conv in enumerate(chat_history[-5:], start=1):  # Affiche les 5 dernières
        print(f"{i}. Vous: {conv['question']}")
        print(f"   Bot: {conv['response']}\n")

while True:
    user_input = input("Vous: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Merci d'avoir utilisé le chatbot historique ! À bientôt.")
        break
    
    bot_response = chat_with_bot(user_input)
    print(f"Bot: {bot_response}\n")
