import os
import json
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from datetime import datetime
import re

# Désactivez le mode multi‑tenant dès le début
os.environ["CHROMADB_DISABLE_MULTITENANT"] = "true"

# Configuration de la page
st.set_page_config(page_title="Chatbot Historique ⚔️", page_icon="⚔️", layout="centered")

# Étape 1 : Demander l'acceptation des CGV et de la politique de confidentialité
accept_cgv = st.checkbox("J'accepte les conditions générales de vente et la politique de confidentialité.")
if not accept_cgv:
    st.warning("Vous devez accepter les conditions générales de vente pour interagir avec le chatbot.")
    st.stop()

# Lien vers la politique de confidentialité
st.markdown("[Politique de confidentialité](URL_de_votre_politique_de_confidentialité)", unsafe_allow_html=True)

# Styles CSS personnalisés pour simuler un chat de type WhatsApp
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
            text-align: center;
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
        /* Ajout du style pour les messages avec des couleurs plus foncées */
        .user-message {
            background-color: #0066cc;  /* Bleu plus foncé */
            color: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            text-align: left;
            width: fit-content;
            max-width: 80%;
            display: inline-block;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .bot-message {
            background-color: #28a745;  /* Vert plus foncé */
            color: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
            text-align: left;
            width: fit-content;
            max-width: 80%;
            display: inline-block;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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

# Filtrer les documents entre 1914 et 1945
def filter_documents_by_date(documents, start_year=1918, end_year=1945):
    filtered_docs = []
    
    for doc in documents:
        date_str = doc['date']
        match = re.search(r'\b(\d{4})\b', date_str)
        
        if match:
            year = int(match.group(1))
            if start_year <= year <= end_year:
                filtered_docs.append(doc)
        else:
            print(f"Date mal formatée ou sans année : {date_str}")
    
    return filtered_docs

# Filtrer les documents par période (1918-1945)
filtered_data = filter_documents_by_date(data)

# Filtrer les doublons et transformer les entrées en documents LangChain
docs = []
for entry in filtered_data:
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
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    vector_store = Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory=persist_directory)
    vector_store.persist()

# Création du retriever pour interroger la base de données
retriever = vector_store.as_retriever()

# Configuration du modèle OpenAI via LangChain
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key, temperature=0.0)

# Construction de la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Liste des mots-clés à filtrer pour interdire des sujets modernes
forbidden_keywords = ["maître gims", "trello", "Steve jobs", "napoléon", "révolution française", "1789", "rois de france"]

# Fonction pour interroger le chatbot et enregistrer l'historique
def chat_with_bot(query):
    # Vérification des mots-clés interdits
    if any(keyword in query.lower() for keyword in forbidden_keywords):
        return "Désolé, je ne peux répondre qu'aux questions concernant les guerres mondiales (1914-1945)."
    
    # Vérification d'une date valide dans la question
    date_match = re.search(r'\b(\d{1,2})\s([a-zA-Zéàû]+)\s(\d{4})\b', query)  # Détecter les dates dans le format "18 juin 1917"
    if date_match:
        year = int(date_match.group(3))
        if year < 1918 or year > 1945:
            return "Désolé, je ne peux répondre qu'aux événements entre 1918 et 1945."
    else:
        return "Désolé, je ne peux répondre qu'aux questions sur les événements entre 1918 et 1945."
    
    # Si la question est dans la période valide, interroger le modèle LangChain
    response = qa_chain.run(query)
    time.sleep(1)  # Pause d'1 seconde pour limiter le risque de blocage par l'API OpenAI

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
        # Affichage conversationnel avec des bulles distinctes pour l'utilisateur et le bot
        st.markdown(f'<div class="user-message">{query}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
    else:
        st.warning("Veuillez entrer une question.")

# Option de téléchargement de l'historique complet dans la sidebar
with st.sidebar.expander("Télécharger l'historique complet"):
    with st.spinner("Préparation du fichier..."):
        results_all = vector_store.similarity_search("Réponse", k=100)
        history_text = "\n\n".join([doc.page_content for doc in results_all])
    st.download_button("Télécharger l'historique", history_text, "historique.txt", "text/plain")
