import unittest
import json
import os
import sys
from unittest.mock import patch, MagicMock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_suggestions_data, create_documents_from_json, setup_rag_pipeline
from langchain_core.documents import Document

class TestRagApplication(unittest.TestCase):
    """Tests unitaires pour l'application RAG de suggestions d'employés."""
    
    def setUp(self):
        """Préparation des données de test."""
        self.test_data = [
            {
                "employe": "Test Employé",
                "evaluation": "Test évaluation",
                "score": 75,
                "suggestions": [
                    {
                        "type": "programme de formation",
                        "content": "Test contenu de formation",
                        "source": "https://test.com/formation"
                    }
                ]
            }
        ]
        
        # Créer un fichier JSON temporaire pour les tests
        self.test_file_path = 'tests/test_data.json'
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, ensure_ascii=False)
    
    def tearDown(self):
        """Nettoyage après les tests."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
    
    def test_load_suggestions_data(self):
        """Test de la fonction de chargement des données."""
        # Patch polars.read_json pour retourner un DataFrame avec nos données de test
        with patch('polars.read_json') as mock_read_json:
            mock_df = MagicMock()
            mock_df.to_dicts.return_value = self.test_data
            mock_read_json.return_value = mock_df
            
            data = load_suggestions_data(self.test_file_path)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]['employe'], "Test Employé")
            self.assertEqual(data[0]['score'], 75)
            self.assertEqual(len(data[0]['suggestions']), 1)
    
    def test_create_documents_from_json(self):
        """Test de la fonction de création de documents à partir des données JSON."""
        documents = create_documents_from_json(self.test_data)
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].metadata['employe'], "Test Employé")
        
        # Vérifier que le contenu du document contient les informations attendues
        self.assertIn("Employé: Test Employé", documents[0].page_content)
        self.assertIn("Score: 75", documents[0].page_content)
        self.assertIn("Suggestions:", documents[0].page_content)
        self.assertIn("Type: programme de formation", documents[0].page_content)
        self.assertIn("Source: https://test.com/formation", documents[0].page_content)
    
    @patch('app.OpenAIEmbeddings')
    @patch('app.FAISS')
    @patch('app.ChatOpenAI')
    def test_setup_rag_pipeline(self, mock_chat_openai, mock_faiss, mock_embeddings):
        """Test de la configuration du pipeline RAG avec des mocks."""
        # Configuration des mocks
        mock_embeddings.return_value = MagicMock()
        mock_vectorstore = MagicMock()
        mock_faiss.from_documents.return_value = mock_vectorstore
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chat_openai.return_value = MagicMock()
        
        # Patch temporaire de la fonction load_suggestions_data
        with patch('app.load_suggestions_data') as mock_load_data:
            mock_load_data.return_value = self.test_data
            
            # Appel de la fonction à tester
            rag_chain = setup_rag_pipeline()
            
            # Vérifications
            mock_load_data.assert_called_once()
            mock_embeddings.assert_called_once()
            mock_faiss.from_documents.assert_called_once()
            mock_vectorstore.as_retriever.assert_called_once()
            mock_chat_openai.assert_called_once()
            self.assertIsNotNone(rag_chain)


if __name__ == '__main__':
    unittest.main()
