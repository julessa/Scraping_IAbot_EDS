import json
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from tqdm import tqdm

# 1️⃣ Charger les données du JSON
with open("combined_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2️⃣ Filtrer les doublons et transformer en documents LangChain avec affichage de progression
docs = []
for entry in tqdm(data, desc="Indexation des événements..."):
    if not entry["duplicate"]:  # Ignorer les doublons
        text = f"{entry['date']} : {entry['event']}"
        docs.append(Document(page_content=text))

# 3️⃣ Fractionner les textes pour l'indexation
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 4️⃣ Vérifier si l'index FAISS existe déjà
index_path = "faiss_index"
if os.path.exists(index_path):
    print("🔄 Chargement de l'index FAISS existant...")
    vector_store = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
else:
    print("⚡ Aucun index trouvé, création d'un nouvel index...")
    vector_store = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    vector_store.save_local(index_path)

# 5️⃣ Créer un retriever pour interroger les données historiques
retriever = vector_store.as_retriever()

# 6️⃣ Configurer le modèle OpenAI via LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo")  # ou "gpt-4"

# 7️⃣ Construire la chaîne de questions-réponses
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# 8️⃣ Fonction pour interroger le chatbot
def chat_with_bot(query):
    response = qa_chain.invoke({"query": query})  # Utilisation de invoke() au lieu de run()
    return response

# 🔥 Tester le chatbot
print("Bot:", chat_with_bot("Que s'est-il passé le 28 juin 1914 ?"))
print("Bot:", chat_with_bot("Quels sont les événements majeurs de la Révolution française ?"))
print("Bot:", chat_with_bot("Que s'est-il passé en mai 1968 en France ?"))
