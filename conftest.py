import sys
import os

# C:\Users\Yun\Desktop\API
BASE_DIR = os.path.dirname(__file__)

# Priorité au dossier app pour les imports
sys.path.insert(0, os.path.join(BASE_DIR, "app"))
sys.path.insert(0, os.path.join(BASE_DIR, "automl"))
sys.path.insert(0, BASE_DIR)