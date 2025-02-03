import streamlit as st
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
import json
import os
import requests
from streamlit_lottie import st_lottie
import sqlite3

# Connexion à la base SQLite
conn = sqlite3.connect("vector_db.sqlite")
cur = conn.cursor()
conn.enable_load_extension(True)  # Active le chargement des extensions

# Charger l'extension VSS pour SQLite
cur.execute("SELECT load_extension('vss0')")

# Création de la table vectorielle (1536 dimensions pour OpenAI)
cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS documents
    USING vss(embedding(1536))
""")

conn.commit()
print("✅ SQLite-VSS configuré avec succès !")
# Configuration du thème de Streamlit
st.set_page_config(page_title="Chatbot Historique ⚔️", page_icon="⚔️", layout="centered")

# Charger les styles CSS personnalisés
st.markdown(
    """
    <style>
    body {
        background-color: #f3f4f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 10px;
        font-size: 18px;
        display: block;
        margin: auto;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: black;
    }
    .container {
        max-width: 700px;
        margin: auto;
        padding: 20px;
        background: white;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    /* ✅ Centrage de l'animation Lottie */
    .lottie-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Fonction pour charger une animation Lottie
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with open("soldiers_animation.json", "r") as f:
    lottie_soldiers = json.load(f)

# 🎨 Interface Streamlit avec animation bien centrée sous le titre
st.markdown(
    """
    <div style="text-align: center;">
        <h1>⚔️ Chatbot Historique ⚔️</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Afficher l'animation centrée sous le titre
st_lottie(lottie_soldiers, speed=1, width=400, height=300, key="soldiers")

# Vérification et définition de la clé API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 Clé API OpenAI non trouvée. Assurez-vous de définir OPENAI_API_KEY dans vos variables d'environnement.")
    st.stop()

# 1️⃣ Charger les données du JSON
with open("combined_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2️⃣ Filtrer les doublons et transformer en documents LangChain
docs = []
for entry in data:
    if not entry["duplicate"]:  # Ignorer les doublons
        text = f"{entry['date']} : {entry['event']}"
        docs.append(Document(page_content=text))

# 3️⃣ Fractionner les textes pour l'indexation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 4️⃣ Vérifier si l'index FAISS existe déjà
index_path = "faiss_index"
if os.path.exists(index_path):
    vector_store = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    vector_store.save_local(index_path)

# 5️⃣ Créer un retriever pour interroger les données historiques
retriever = vector_store.as_retriever()

# 6️⃣ Configurer le modèle OpenAI via LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 7️⃣ Construire la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 8️⃣ Fonction pour interroger le chatbot
def chat_with_bot(query):
    response = qa_chain.invoke({"query": query})
    return response

# Zone de saisie utilisateur
query = st.text_input("Posez votre question sur l'histoire de France :")

if st.button("Envoyer"):
    if query:
        with st.spinner("Recherche en cours..."):
            response = chat_with_bot(query)
            st.markdown("### Réponse :")
            st.write(response)
    else:
        st.warning("Veuillez entrer une question.")
