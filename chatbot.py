import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import json
import os
import requests
from streamlit_lottie import st_lottie

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

# Chargement de l'animation depuis le fichier JSON
with open("soldiers_animation.json", "r") as f:
    lottie_soldiers = json.load(f)

# Afficher le titre et l'animation
st.markdown(
    """
    <div style="text-align: center;">
        <h1>⚔️ Chatbot Historique ⚔️</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st_lottie(lottie_soldiers, speed=1, width=400, height=300, key="soldiers")

# Vérification et définition de la clé API OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 Clé API OpenAI non trouvée. Assurez-vous de définir OPENAI_API_KEY dans vos variables d'environnement.")
    st.stop()

# 1️⃣ Charger les données du JSON
with open("combined_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2️⃣ Filtrer les doublons et transformer en Documents LangChain
docs = []
for entry in data:
    if "duplicate" in entry and not entry["duplicate"]:
        text = f"{entry['date']} : {entry['event']}"
        docs.append(Document(page_content=text))

# 3️⃣ Fractionner les textes pour l'indexation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 4️⃣ Initialiser Chroma
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

# 5️⃣ Créer un retriever pour interroger les données historiques
retriever = vector_store.as_retriever()

# 6️⃣ Configurer le modèle OpenAI via LangChain
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key,
    temperature=0.0
)

# 7️⃣ Construire la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# 8️⃣ Fonction pour interroger le chatbot et enregistrer l'historique dans Chroma
def chat_with_bot(query):
    response = qa_chain.run(query)
    
    # Créer un document avec la question et la réponse
    history_entry = f"Question: {query}\nRéponse: {response}"
    history_doc = Document(page_content=history_entry)
    
    # Ajouter l'entrée d'historique à Chroma
    vector_store.add_documents([history_doc])
    
    # Sauvegarder les changements dans Chroma
    vector_store.persist()

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

# 9️⃣ Afficher l'historique des questions et réponses depuis Chroma
if st.button("Afficher l'historique"):
    with st.spinner("Chargement de l'historique..."):
        results = vector_store.similarity_search("Réponse", k=5)  # Affiche les 5 dernières interactions
        st.markdown("### Historique des questions et réponses :")
        for result in results:
            st.write(result.page_content)
            st.markdown("---")
