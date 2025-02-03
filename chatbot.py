import streamlit as st
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import json
import os
import requests
from streamlit_lottie import st_lottie

# Pour la visualisation
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import shutil  # Pour supprimer le dossier d'index si nécessaire

# Importation du client de Chroma
import chromadb

# Configuration du thème de Streamlit
st.set_page_config(page_title="Chatbot Historique ⚔️", page_icon="⚔️", layout="centered")

# Chargement des styles CSS personnalisés
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

# Affichage du titre et de l'animation
st.markdown(
    """
    <div style="text-align: center;">
        <h1>⚔️ Chatbot Historique ⚔️</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st_lottie(lottie_soldiers, speed=1, width=400, height=300, key="soldiers")

# Vérification de la clé API OpenAI
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

st.write(f"Nombre de documents après filtrage : {len(docs)}")

# 3️⃣ Fractionner les textes pour l'indexation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
st.write(f"Nombre de documents après découpage : {len(split_docs)}")

# Afficher un exemple de document indexé (si disponible)
if len(split_docs) > 0:
    st.write("Exemple de document indexé :", split_docs[0].page_content)

# 4️⃣ Initialiser le vector store avec Chroma
persist_directory = "chroma_db"
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Créer explicitement un client Chroma en mode embarqué
chroma_client = chromadb.Client()  # Utilise les paramètres par défaut pour un client local

# Option pour forcer la recréation de l'index
force_recreate = st.checkbox("Forcer la recréation de l'index", value=False)
if force_recreate and os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    st.write("Dossier d'index supprimé. L'index sera recréé.")

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    st.write("Chargement de l'index existant...")
    vector_store = Chroma(
        client=chroma_client,
        persist_directory=persist_directory,
        embedding_function=embedding
    )
else:
    st.write("Création d'un nouvel index...")
    vector_store = Chroma.from_documents(
        split_docs,
        embedding,
        persist_directory=persist_directory,
        client=chroma_client
    )
    vector_store.persist()
    st.write("Index créé et sauvegardé.")

# Afficher le contenu du dossier d'index pour vérifier
if os.path.exists(persist_directory):
    files = os.listdir(persist_directory)
    st.write("Fichiers présents dans le dossier d'index :", files)
else:
    st.write("Le dossier d'index n'existe pas.")

# 5️⃣ Créer un retriever pour interroger les données historiques
retriever = vector_store.as_retriever()

# 6️⃣ Configurer le modèle OpenAI via LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# 7️⃣ Construire la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 8️⃣ Fonction pour interroger le chatbot
def chat_with_bot(query):
    response = qa_chain.invoke({"query": query})
    return response

# Interface pour interroger le chatbot
query = st.text_input("Posez votre question sur l'histoire de France :")
if st.button("Envoyer"):
    if query:
        with st.spinner("Recherche en cours..."):
            response = chat_with_bot(query)
            st.markdown("### Réponse :")
            st.write(response)
    else:
        st.warning("Veuillez entrer une question.")

st.markdown("---")
st.markdown("## Visualisation des Embeddings")

# Bouton pour lancer la visualisation des embeddings
if st.button("Visualiser les embeddings"):
    try:
        # Récupérer toutes les entrées en fixant un grand limit (ici 10000)
        result = vector_store._collection.get(limit=10000, include=["embeddings", "documents"])
        embeddings = result.get("embeddings", [])
        documents = result.get("documents", [])
        
        st.write(f"Nombre d'entrées récupérées : {len(embeddings)}")
        
        # Conversion en tableau numpy
        embeddings_np = np.array(embeddings)
        if embeddings_np.size == 0:
            st.write("Aucun embedding trouvé dans la base.")
        else:
            # Réduction de dimension en 2D avec PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_np)
            # Préparer un DataFrame pour Plotly
            df = pd.DataFrame({
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "document": documents
            })
            # Visualisation interactive avec Plotly
            fig = px.scatter(df, x="x", y="y", hover_data=["document"], title="Visualisation des embeddings")
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour la visualisation : {e}")

 