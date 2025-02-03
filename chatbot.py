import os
# Désactivez le mode multi‑tenant dès le début
os.environ["CHROMADB_DISABLE_MULTITENANT"] = "true"

import streamlit as st
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import json
import requests
from streamlit_lottie import st_lottie

# Pour la visualisation
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Configuration de Streamlit
st.set_page_config(page_title="Chatbot Historique ⚔️", page_icon="⚔️", layout="centered")
st.markdown("""<style>
body { background-color: #f3f4f6; }
.stButton>button { color: white; background-color: #4CAF50; border-radius: 10px; font-size: 18px; display: block; margin: auto; }
.stTextInput>div>div>input { background-color: #ffffff; color: black; }
.container { max-width: 700px; margin: auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); text-align: center; }
.lottie-container { display: flex; justify-content: center; align-items: center; }
</style>""", unsafe_allow_html=True)

with open("soldiers_animation.json", "r") as f:
    lottie_soldiers = json.load(f)
st.markdown("""<div style="text-align: center;"><h1>⚔️ Chatbot Historique ⚔️</h1></div>""", unsafe_allow_html=True)
st_lottie(lottie_soldiers, speed=1, width=400, height=300, key="soldiers")

# Vérification de la clé API
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 Clé API OpenAI non trouvée.")
    st.stop()

# Charger les données JSON
with open("combined_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = []
for entry in data:
    if not entry["duplicate"]:
        text = f"{entry['date']} : {entry['event']}"
        docs.append(Document(page_content=text))
st.write(f"Nombre de documents après filtrage : {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)
st.write(f"Nombre de documents après découpage : {len(split_docs)}")
if split_docs:
    st.write("Exemple de document indexé :", split_docs[0].page_content)

# Utiliser le mode en mémoire : ne pas spécifier persist_directory
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
st.write("Création d'un nouvel index (mode en mémoire)...")
vector_store = Chroma.from_documents(split_docs, embedding)
# Pas besoin d'appeler persist() en mode mémoire

retriever = vector_store.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

def chat_with_bot(query):
    response = qa_chain.invoke({"query": query})
    return response

query = st.text_input("Posez votre question sur l'histoire de France :")
if st.button("Envoyer"):
    if query:
        with st.spinner("Recherche..."):
            response = chat_with_bot(query)
            st.markdown("### Réponse :")
            st.write(response)
    else:
        st.warning("Veuillez entrer une question.")

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
        st.error(f"Erreur lors de la récupération des données : {e}")
 