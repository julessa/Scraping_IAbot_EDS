import os
import json
import time  # Import nécessaire pour time.sleep
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from datetime import datetime
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# Désactivez le mode multi‑tenant dès le début
os.environ["CHROMADB_DISABLE_MULTITENANT"] = "true"

# Configuration de la page
st.set_page_config(page_title="Chatbot Historique ⚔️", page_icon="⚔️", layout="centered")

# Styles CSS personnalisés
st.markdown(
    """
    <style>
        body {
            background-color: #f3f4f6;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 10px;
            font-size: 18px;
            padding: 10px 20px;
            border: none;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: black;
            font-size: 16px;
            padding: 8px;
        }
        .container {
            max-width: 700px;
            margin: auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            padding: 20px;
        }
        .lottie-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Chargement de l'animation depuis le fichier JSON
with open("soldiers_animation.json", "r") as f:
    lottie_soldiers = json.load(f)

# Vérification de la clé API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 Clé API OpenAI non trouvée.")
    st.stop()

# Chargement des données JSON contenant les événements historiques
with open("combined_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Filtrer les doublons et transformer les entrées en documents LangChain
docs = []
for entry in data:
    if "duplicate" in entry and not entry["duplicate"]:
        text = f"{entry['date']} : {entry['event']}"
        docs.append(Document(page_content=text))

# Découpage des documents en morceaux
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Initialisation (ou chargement) de la base vectorielle Chroma
persist_directory = "chroma_db"
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
else:
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vector_store.persist()

# Création du retriever pour interroger la base de données
retriever = vector_store.as_retriever()

# Configuration du modèle OpenAI via LangChain
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.0
)

# Construction de la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# --- MENU LATÉRAL ---
st.sidebar.header("Instructions")
st.sidebar.info(
    "Posez vos questions sur l'histoire de France via le formulaire principal. "
    "L'historique des interactions est affiché ci-dessous."
)

with st.sidebar.expander("Historique des interactions récents"):
    with st.spinner("Chargement de l'historique..."):
        results = vector_store.similarity_search("Réponse", k=5)
        if results:
            for doc in results:
                st.markdown(doc.page_content)
                st.markdown("---")
        else:
            st.info("Aucune interaction trouvée.")

st.sidebar.markdown("### À propos")
st.sidebar.info(
    "Ce chatbot utilise LangChain et l'API OpenAI pour répondre à vos questions sur l'histoire de France. "
    "Les interactions sont sauvegardées et affichées dans l'historique."
)

# --- ZONE PRINCIPALE ---
with st.container():
    st.markdown('<div class="header"><h1>⚔️ Chatbot Historique ⚔️</h1></div>', unsafe_allow_html=True)
    st_lottie(lottie_soldiers, speed=1, width=400, height=300, key="soldiers")

# Fonction pour interroger le chatbot et enregistrer l'historique
def chat_with_bot(query):
    response = qa_chain.run(query)
    # Pause d'1 seconde pour limiter le risque de blocage par l'API OpenAI
    time.sleep(1)
    # Ajout d'un timestamp pour chaque interaction
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_entry = f"**[{timestamp}] Question:** {query}\n\n**Réponse:** {response}"
    history_doc = Document(page_content=history_entry)
    # Ajout de l'entrée d'historique à Chroma et sauvegarde
    vector_store.add_documents([history_doc])
    vector_store.persist()
    return response

# Formulaire pour saisir la question avec bouton centré
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Posez votre question sur l'histoire de France :", "")
    col1, col2, col3 = st.columns([1, 1, 1])
    submit_button = col2.form_submit_button("Envoyer")

if submit_button:
    if query:
        with st.spinner("Recherche..."):
            response = chat_with_bot(query)
        st.markdown("### Réponse :")
        st.write(response)
    else:
        st.warning("Veuillez entrer une question.")

# Option de téléchargement de l'historique complet dans la sidebar
with st.sidebar.expander("Télécharger l'historique complet"):
    with st.spinner("Préparation du fichier..."):
        results_all = vector_store.similarity_search("Réponse", k=100)
        history_text = "\n\n".join([doc.page_content for doc in results_all])
    st.download_button("Télécharger l'historique", history_text, "historique.txt", "text/plain")
