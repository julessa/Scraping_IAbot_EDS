import json
import os
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Charger et préparer les données du fichier JSON
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
        # Afficher chaque paire dans le terminal
        print(f"Question: {question}")
        print(f"Réponse: {answer}")
        print("-" * 50)
    return qa_pairs


# 3. Convertir les événements en documents LangChain et les indexer dans Chroma
def prepare_chroma_index(qa_pairs, persist_directory, embedding, chunk_size=500, chunk_overlap=50):
    # Créer des documents LangChain à partir des paires question-réponse
    docs = []
    for qa in qa_pairs:
        text = f"Question: {qa['question']} Réponse: {qa['answer']}"
        docs.append(Document(page_content=text))
    
    # Fractionner les documents pour une meilleure indexation
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)

    # Vérifier si le répertoire Chroma existe déjà
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

# 4. Améliorer la récupération de données en Chroma avec des métadonnées et réévaluation des embeddings
def improve_chroma_retrieval(vector_store, new_data, embedding):
    new_docs = []
    for entry in new_data:
        text = f"{entry['date']} : {entry['event']}"
        new_docs.append(Document(page_content=text))
    
    # Ajouter de nouveaux documents et réindexer
    vector_store.add_documents(new_docs)
    vector_store.persist()
    return vector_store

# Main
if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    persist_directory = "chroma_db"

    # Charger les données et générer les paires question-réponse
    cleaned_data = load_and_clean_data("combined_data.json")
    qa_pairs = generate_qa_pairs(cleaned_data)
    print(f"Generated {len(qa_pairs)} question-answer pairs.")

    # Initialiser les embeddings OpenAI
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Préparer et indexer les données dans Chroma
    vector_store = prepare_chroma_index(qa_pairs, persist_directory, embedding)
    print("Chroma index created successfully.")

    # Si vous souhaitez améliorer l'index avec de nouvelles données ou l'historique
    vector_store = improve_chroma_retrieval(vector_store, cleaned_data, embedding)
    print("Chroma retrieval enhanced with new data.")
