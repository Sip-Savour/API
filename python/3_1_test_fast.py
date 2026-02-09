import time
import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
from predict import fast_predict


# ================= TEST =================
tests = [
    ("Poulet (White)", "chicken white citrus butter", "white"),
    ("Steak (Red)", "steak grilled pepper smoke", "red"),
    ("Dessert (Piege)", "cake marzipan honey", "white"), 
]

print("=== DÉBUT DES TESTS AVEC FALLBACK ===")
print(f"{'TEST':<15} | {'DÉCISION IA':<20} | {'STRATÉGIE':<25} | {'RÉSULTAT'}")
print("-" * 100)

for nom, desc, color in tests:
    decision, bouteille, duree, strategie = fast_predict(desc, color)

    if bouteille is not None:
        res = f"{bouteille['title'][:30]}..."
    else:
        res = "Aucun résultat"

    print(f"{nom:<15} | {str(decision)[:20]:<20} | {strategie:<25} | {res}")

print("-" * 100)
