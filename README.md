# Système de Questions-Réponses sur les Recommandations des Employés

Ce projet implémente un système de questions-réponses basé sur le mécanisme de Retrieval-Augmented Generation (RAG) pour interroger un dataset de recommandations d'employés. Le système permet de poser des questions en langage naturel sur les évaluations et recommandations des employés, et reçoit des réponses contextuelles pertinentes.

## Fonctionnalités

- Chargement et traitement des données d'employés à partir d'un fichier JSON  
- Conversion des données en documents structurés pour le RAG  
- Création d'un index vectoriel pour la recherche sémantique  
- Interface en ligne de commande pour poser des questions en français  
- Réponses contextuelles basées sur les données d'employés  

## Structure du Dataset

Le dataset (`data/suggestions.json`) contient des informations sur les employés, incluant :

- Nom de l'employé  
- Évaluation professionnelle  
- Score de performance  
- Suggestions de formation et de développement  

Chaque employé peut avoir plusieurs suggestions, chacune avec un type (programme de formation, meilleures pratiques, étude de cas), un contenu détaillé et une source.

## Prérequis

- Python 3.8+  
- OpenAI API Key  

## Installation

1. Cloner le dépôt :
   ```bash
   git clone <repository-url>
   cd clean-devo-rag
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Créer un fichier `.env` à la racine du projet avec votre clé API OpenAI :
   ```bash
   OPENAI_API_KEY=votre_clé_api_openai
   ```

## Utilisation

Pour lancer l'application :

```bash
python app.py
```

Vous pouvez ensuite poser des questions en français sur les employés, leurs évaluations et les recommandations qui leur sont faites. Par exemple :

- "Quelles formations sont recommandées pour Marie Dupont ?"  
- "Quels employés ont un score inférieur à 65 ?"  
- "Quelles sont les recommandations pour les employés qui manquent de leadership ?"  
- "Quel type de formation est le plus recommandé dans l'ensemble des données ?"  

Pour quitter l'application, tapez simplement `exit`.

## Architecture Technique

L'application utilise LangChain pour implémenter le pipeline RAG :

1. **Chargement des données** : Les données JSON sont chargées et converties en documents LangChain.  
2. **Découpage des documents** : Les documents sont découpés en chunks de taille appropriée pour l'indexation.  
3. **Création d'embeddings** : OpenAI Embeddings est utilisé pour vectoriser les documents.  
4. **Indexation vectorielle** : FAISS est utilisé comme base de données vectorielle pour stocker et rechercher efficacement les documents similaires.  
5. **Chaîne de requête** : Une chaîne LangChain est configurée pour :
   - Recevoir une question  
   - Récupérer les documents pertinents  
   - Générer une réponse contextualisée avec ChatGPT  

## Extensions Possibles

- Interface web avec Streamlit ou Gradio  
- Support pour d'autres langues  
- Visualisation des données d'employés  
- Intégration avec d'autres sources de données RH  
- **Remplacer FAISS par ChromaDB** : ChromaDB est une alternative plus récente et conviviale à FAISS. Elle offre une gestion plus souple des métadonnées, permet de sauvegarder les données de manière persistante, et s’intègre facilement dans des environnements en production. C’est un bon choix si tu veux suivre précisément l’origine des documents, mettre à jour les embeddings à la volée, ou gérer plusieurs utilisateurs.

