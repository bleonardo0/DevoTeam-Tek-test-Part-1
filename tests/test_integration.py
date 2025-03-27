import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_suggestions_data, create_documents_from_json
import config


class TestDataProcessing(unittest.TestCase):
    """Tests pour les fonctions de traitement des données."""
    
    def setUp(self):
        """Préparation des données de test."""
        # Chemin vers le fichier de test
        self.test_file_path = 'tests/test_data_mini.json'
        
        # Charger les données de test
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
    
    def test_load_test_data(self):
        """Test du chargement des données de test."""
        # Patch polars.read_json pour retourner un DataFrame avec nos données de test
        with patch('polars.read_json') as mock_read_json:
            # Charger les données de test
            with open(self.test_file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
                
            mock_df = MagicMock()
            mock_df.to_dicts.return_value = test_data
            mock_read_json.return_value = mock_df
            
            data = load_suggestions_data(self.test_file_path)
            
            # Vérifier que les données sont correctement chargées
            self.assertEqual(len(data), 2)  # Deux employés dans le fichier de test
            self.assertEqual(data[0]['employe'], "Jean Test")
            self.assertEqual(data[1]['employe'], "Marie Test")
            
            # Vérifier les scores
            self.assertEqual(data[0]['score'], 72)
            self.assertEqual(data[1]['score'], 85)
            
            # Vérifier les suggestions
            self.assertEqual(len(data[0]['suggestions']), 2)
            self.assertEqual(len(data[1]['suggestions']), 2)
            
            # Vérifier le type de la première suggestion
            self.assertEqual(data[0]['suggestions'][0]['type'], "programme de formation")
    
    def test_create_documents(self):
        """Test de la création de documents à partir des données de test."""
        documents = create_documents_from_json(self.test_data)
        
        # Vérifier le nombre de documents créés
        self.assertEqual(len(documents), 2)
        
        # Vérifier les métadonnées
        self.assertEqual(documents[0].metadata['employe'], "Jean Test")
        self.assertEqual(documents[1].metadata['employe'], "Marie Test")
        
        # Vérifier le contenu des documents
        self.assertIn("Jean Test", documents[0].page_content)
        self.assertIn("Score: 72", documents[0].page_content)
        self.assertIn("communication efficace", documents[0].page_content)
        
        self.assertIn("Marie Test", documents[1].page_content)
        self.assertIn("Score: 85", documents[1].page_content)
        self.assertIn("technologies cloud", documents[1].page_content)


class TestRagWithRealData(unittest.TestCase):
    """Tests avec les données réelles (à exécuter uniquement si les données existent)."""
    
    def setUp(self):
        """Vérifier si le fichier de données réel existe."""
        self.real_data_path = config.SUGGESTIONS_DATA_PATH
        self.has_real_data = os.path.exists(self.real_data_path)
        
        if not self.has_real_data:
            self.skipTest("Le fichier de données réel n'existe pas")
    
    def test_load_real_data(self):
        """Test du chargement des données réelles."""
        # Patch polars.read_json pour retourner un DataFrame avec nos données réelles
        with patch('polars.read_json') as mock_read_json:
            # Charger les données réelles
            with open(self.real_data_path, 'r', encoding='utf-8') as f:
                real_data = json.load(f)
                
            mock_df = MagicMock()
            mock_df.to_dicts.return_value = real_data
            mock_read_json.return_value = mock_df
            
            data = load_suggestions_data(self.real_data_path)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            
            # Vérifier la structure des données
            first_employee = data[0]
            self.assertIn('employe', first_employee)
            self.assertIn('evaluation', first_employee)
            self.assertIn('score', first_employee)
            self.assertIn('suggestions', first_employee)
            
            # Vérifier la structure des suggestions
            first_suggestion = first_employee['suggestions'][0]
            self.assertIn('type', first_suggestion)
            self.assertIn('content', first_suggestion)
            self.assertIn('source', first_suggestion)
    
    def test_create_documents_from_real_data(self):
        """Test de la création de documents à partir des données réelles."""
        # Patch polars.read_json pour retourner un DataFrame avec nos données réelles
        with patch('polars.read_json') as mock_read_json:
            # Charger les données réelles
            with open(self.real_data_path, 'r', encoding='utf-8') as f:
                real_data = json.load(f)
                
            mock_df = MagicMock()
            mock_df.to_dicts.return_value = real_data
            mock_read_json.return_value = mock_df
            
            data = load_suggestions_data(self.real_data_path)
            documents = create_documents_from_json(data)
            
            self.assertIsInstance(documents, list)
            self.assertEqual(len(documents), len(data))
            
            # Vérifier que tous les documents sont des instances de Document
            for doc in documents:
                self.assertIn('employe', doc.metadata)


if __name__ == '__main__':
    unittest.main()
