import json
import os
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Charger et nettoyer les données du fichier JSON
def load_and_clean_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filtrer les doublons
    cleaned_data = [entry for entry in data if not entry.get("duplicate", False)]
    return cleaned_data

# 2. Créer un jeu de données question-réponse
def generate_qa_pairs(cleaned_data):
    qa_pairs = []
    
    for entry in cleaned_data:
        # Créer des questions à partir des événements
        question = f"Que s'est-il passé le {entry['date']} ?"
        answer = entry['event']
        qa_pairs.append({"question": question, "answer": answer})
    
    return qa_pairs

# 3. Convertir les événements en documents LangChain et les indexer dans un index séparé pour l'entraînement
def prepare_chroma_index_for_training(qa_pairs, persist_directory, embedding, chunk_size=500, chunk_overlap=50):
    # Créer des documents LangChain à partir des paires question-réponse
    docs = []
    for qa in qa_pairs:
        text = f"Question: {qa['question']} Réponse: {qa['answer']}"
        docs.append(Document(page_content=text))
    
    # Fractionner les documents pour une meilleure indexation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)

    # Créer un index Chroma pour les paires question-réponse
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

    return vector_store

# 4. Ajouter l'historique des conversations dans un index distinct
def add_to_chat_history(query, response, chat_history_vector_store, embedding):
    history_entry = f"Question: {query}\nRéponse: {response}"
    history_doc = Document(page_content=history_entry)
    chat_history_vector_store.add_documents([history_doc])
    chat_history_vector_store.persist()

# Fonction de visualisation des documents dans Chroma (historique ou entraînement)
def display_chroma_documents(vector_store, num_docs=5):
    documents = vector_store.similarity_search("", k=num_docs)
    for doc in documents:
        print(f"**Contenu du document :** {doc.page_content}")
        print("-" * 50)

# Main
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Répertoires d'index
    training_data_directory = "training_data_db"  # Répertoire pour les paires question-réponse
    chat_history_directory = "chat_history_db"   # Répertoire pour l'historique des conversations

    # Charger les données et générer les paires question-réponse
    cleaned_data = load_and_clean_data("combined_data.json")
    qa_pairs = generate_qa_pairs(cleaned_data)

    # Initialiser les embeddings OpenAI
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Créer un index Chroma pour les questions-réponses (formation)
    training_vector_store = prepare_chroma_index_for_training(qa_pairs, training_data_directory, embedding)
    print("Chroma index created for training data.")

    # Créer un index séparé pour l'historique des conversations
    chat_history_vector_store = Chroma(persist_directory=chat_history_directory, embedding_function=embedding)

    # Exemple d'ajout d'une question-réponse à l'historique (après une interaction avec le chatbot)
    query = "Quel événement a eu lieu le 14 juillet 1789 ?"
    response = "La prise de la Bastille a eu lieu le 14 juillet 1789."
    add_to_chat_history(query, response, chat_history_vector_store, embedding)
    print("Added to chat history.")

    # Optionnel : Visualisation des documents indexés
    print("\n--- Documents d'entraînement (Question-Réponse) ---")
    display_chroma_documents(training_vector_store, num_docs=5)

    print("\n--- Historique des conversations ---")
    display_chroma_documents(chat_history_vector_store, num_docs=5)
