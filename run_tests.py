#!/usr/bin/env python3
"""
Script pour exécuter tous les tests unitaires du projet.
"""
import unittest
import os
import sys

if __name__ == "__main__":
    # Ajouter le répertoire parent au chemin de recherche des modules
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Découvrir et exécuter tous les tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    # Exécuter les tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Sortir avec un code d'erreur si des tests ont échoué
    sys.exit(not result.wasSuccessful())
