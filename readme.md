# Chatbot Historique ⚔️

Ce projet est un chatbot historique permettant de répondre à des questions sur l'histoire de France en s'appuyant sur une base de données indexée avec FAISS et utilisant OpenAI via LangChain. Il dispose d'une interface interactive réalisée avec Streamlit.

## Installation et Configuration

### 1. Cloner le dépôt
```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows

Set-ExecutionPolicy Unrestricted -Scope Process # SI erreur windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API OpenAI
Créer un fichier `.env` à la racine du projet et y ajouter :
```bash
OPENAI_API_KEY="votre_cle_api_openai"
```
Ou exporter la variable directement :
```bash
export OPENAI_API_KEY="votre_cle_api_openai"  # macOS/Linux
set OPENAI_API_KEY="votre_cle_api_openai"  # Windows
```

## Lancer l'application

Exécutez la commande suivante pour démarrer l'interface Streamlit :
```bash
streamlit run chatbot_frontend.py
```
Cela ouvrira automatiquement l'application dans votre navigateur.

## Fonctionnalités

- **Répondre à des questions sur l'histoire de France** en utilisant une base de données préchargée.
- **Interface interactive** avec Streamlit.
- **Indexation optimisée** des événements historiques avec FAISS.
- **Animation Lottie** représentant des soldats pour une meilleure immersion visuelle.

## Structure du projet
```
/
|-- chatbot_frontend.py  # Code de l'interface et du chatbot
|-- combined_data.json  # Base de données historique
|-- faiss_index/  # Index FAISS (créé après la première exécution)
|-- soldiers_animation.json  # Animation Lottie
|-- requirements.txt  # Dépendances Python
|-- .env  # Clé API OpenAI (à créer)
```

## Problèmes courants et solutions

### 1. **Erreur : OpenAI API key non définie**
   - Vérifiez que la clé API est bien définie dans le fichier `.env` ou en tant que variable d'environnement.
   - Relancez votre terminal après avoir défini la clé.

### 2. **L'animation ne s'affiche pas**
   - Vérifiez que `soldiers_animation.json` est bien dans le dossier racine.
   - Assurez-vous que `streamlit-lottie` est installé :
     ```bash
     pip install streamlit-lottie
     ```

### 3. **L'index FAISS est corrompu**
   - Supprimez le dossier `faiss_index/` et relancez le script pour reconstruire l'index.

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à soumettre une issue ou une pull request.

## Licence
Ce projet est sous licence MIT. Voir `LICENSE` pour plus d'informations.

