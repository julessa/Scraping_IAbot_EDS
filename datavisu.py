import os
# Désactiver le mode multi‑tenant (si vous souhaitez l'utiliser en mono‑tenant)
os.environ["CHROMADB_DISABLE_MULTITENANT"] = "true"

import streamlit as st
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma  # Version intégrée dans LangChain
import json
import requests
from streamlit_lottie import st_lottie

# Pour la visualisation
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import shutil  # Pour supprimer le dossier d'index si nécessaire

# --- Configuration de Streamlit ---
st.set_page_config(page_title="Chatbot Historique ⚔️", page_icon="⚔️", layout="centered")
st.markdown(
    """
    <style>
    body { background-color: #f3f4f6; }
    .stButton>button { color: white; background-color: #4CAF50; border-radius: 10px; font-size: 18px; display: block; margin: auto; }
    .stTextInput>div>div>input { background-color: #ffffff; color: black; }
    .container { max-width: 700px; margin: auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); text-align: center; }
    .lottie-container { display: flex; justify-content: center; align-items: center; }
    </style>
    """, unsafe_allow_html=True
)

# --- Chargement de l'animation Lottie ---
with open("soldiers_animation.json", "r") as f:
    lottie_soldiers = json.load(f)
st.markdown("""<div style="text-align: center;"><h1>⚔️ Chatbot Historique ⚔️</h1></div>""", unsafe_allow_html=True)
st_lottie(lottie_soldiers, speed=1, width=400, height=300, key="soldiers")

# --- Vérification de la clé API OpenAI ---
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 Clé API OpenAI non trouvée. Veuillez la définir dans vos variables d'environnement.")
    st.stop()

# --- 1. Charger les données du JSON ---
with open("combined_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- 2. Filtrer les doublons et transformer en documents LangChain ---
docs = []
for entry in data:
    if not entry["duplicate"]:
        text = f"{entry['date']} : {entry['event']}"
        docs.append(Document(page_content=text))
st.write(f"Nombre de documents après filtrage : {len(docs)}")

# --- 3. Fractionner les textes pour l'indexation ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
st.write(f"Nombre de documents après découpage : {len(split_docs)}")
if split_docs:
    st.write("Exemple de document indexé :", split_docs[0].page_content)

# --- 4. Création ou chargement de l'index persistant ---
# Définissez ici le dossier de persistance pour enregistrer l'index sur disque.
persist_directory = "chroma_db"
force_recreate = st.checkbox("Forcer la recréation de l'index", value=False)

@st.cache_resource(show_spinner=False)
def get_vector_store(_split_docs, _embedding, persist_directory, force_recreate):
    # Si force_recreate est activé et que le dossier existe, on le supprime pour repartir de zéro
    if force_recreate and persist_directory and os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        st.write("Dossier d'index supprimé. L'index sera recréé.")
    # Si le dossier de persistance existe et contient des fichiers, on le charge
    if persist_directory and os.path.exists(persist_directory) and os.listdir(persist_directory):
        st.write("Chargement de l'index existant...")
        return Chroma(
            persist_directory=persist_directory, 
            embedding_function=_embedding, 
            chroma_db_impl="duckdb+parquet"
        )
    else:
        st.write("Création d'un nouvel index persistant (backend DuckDB+Parquet)...")
        vector_store = Chroma.from_documents(
            _split_docs, 
            _embedding, 
            persist_directory=persist_directory, 
            chroma_db_impl="duckdb+parquet"
        )
        vector_store.persist()
        st.write("Index créé et sauvegardé.")
        return vector_store

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vector_store = get_vector_store(split_docs, embedding, persist_directory, force_recreate)

# Vérification du contenu du dossier d'index
if persist_directory:
    if os.path.exists(persist_directory):
        files = os.listdir(persist_directory)
        st.write("Fichiers présents dans le dossier d'index :", files)
    else:
        st.write("Le dossier d'index n'existe pas.")

# --- 5. Création du retriever ---
retriever = vector_store.as_retriever()

# --- 6. Configuration du modèle OpenAI via LangChain ---
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# --- 7. Construction de la chaîne de questions-réponses ---
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def chat_with_bot(query):
    return qa_chain.invoke({"query": query})

# --- Interface du chatbot ---
query = st.text_input("Posez votre question sur l'histoire de France :")
if st.button("Envoyer"):
    if query:
        with st.spinner("Recherche en cours..."):
            response = chat_with_bot(query)
            st.markdown("### Réponse :")
            st.write(response)
    else:
        st.warning("Veuillez entrer une question.")

# --- 8. Visualisation des embeddings ---
st.markdown("---")
st.markdown("## Visualisation des Embeddings")
if st.button("Visualiser les embeddings"):
    try:
        result = vector_store._collection.get(limit=10000, include=["embeddings", "documents"])
        embeddings = result.get("embeddings", [])
        documents = result.get("documents", [])
        st.write(f"Nombre d'entrées récupérées : {len(embeddings)}")
        embeddings_np = np.array(embeddings)
        if embeddings_np.size == 0:
            st.write("Aucun embedding trouvé dans la base.")
        else:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_np)
            df = pd.DataFrame({
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "document": documents
            })
            fig = px.scatter(df, x="x", y="y", hover_data=["document"], title="Visualisation des embeddings")
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données pour la visualisation : {e}")
