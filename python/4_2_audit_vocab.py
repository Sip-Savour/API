import pandas as pd
import joblib
import os
import re
from collections import Counter

# ================= CONFIGURATION =================
DATA_DIR = "data/"
GENERATED_DIR = "generated_files/pkl/"
INPUT_DB = DATA_DIR + "wines_db_full.csv"
KEYWORDS_FILE = GENERATED_DIR + "keywords_list.pkl"  # Votre liste actuelle (générée par 1_prepare)
OUTPUT_FILE = DATA_DIR + "audit_vocabulary.csv"
TOP_N_WORDS = 1500  # Combien de mots on analyse

# Liste de mots à ignorer (Stop Words)
# On enlève l'anglais basique ET le vocabulaire générique du vin qui n'apporte pas d'info sur le goût
STOP_WORDS = set([
    # Anglais basique
    "the", "and", "a", "of", "with", "is", "in", "it", "to", "that", "this", 
    "for", "on", "as", "are", "by", "an", "at", "be", "has", "from", "its",
    "but", "not", "or", "have", "some", "very", "more", "now", "up", "can",
    # Mots génériques du vin (N'aident pas à différencier)
    "wine", "flavors", "notes", "palate", "finish", "nose", "aromas", "drink", 
    "years", "bottle", "glass", "show", "shows", "well", "good", "made", 
    "blend", "vineyard", "vintage", "character", "style", "bottling", "opens",
    "through", "offers", "hint", "hints", "touch", "bit", "gives", "named"
])

def audit_vocabulary():
    print(f"📚 Audit du vocabulaire des descriptions...")

    # 1. Chargement des données
    if not os.path.exists(INPUT_DB):
        print(f"❌ Erreur : {INPUT_DB} introuvable.")
        return

    try:
        df = pd.read_csv(INPUT_DB, usecols=['description'], on_bad_lines='skip', low_memory=False)
    except:
        print("⚠️ Erreur lecture CSV standard, tentative mode robuste...")
        df = pd.read_csv(INPUT_DB, on_bad_lines='skip', low_memory=False)

    print(f"   > Analyse de {len(df)} descriptions...")

    # 2. Chargement des mots-clés actuels (ceux déjà dans votre système)
    current_keywords = set()
    if os.path.exists(KEYWORDS_FILE):
        try:
            current_keywords = set(joblib.load(KEYWORDS_FILE))
            print(f"   > Vous utilisez actuellement {len(current_keywords)} mots-clés.")
        except:
            print("   ⚠️ Impossible de lire la liste actuelle.")
    else:
        print("   ⚠️ Aucune liste de mots-clés existante trouvée.")

    # 3. Traitement du texte (Tokenization)
    print("   > Comptage des mots (cela peut prendre quelques secondes)...")
    
    # On joint tout le texte pour aller plus vite
    text_blob = " ".join(df['description'].dropna().astype(str).tolist()).lower()
    
    # Regex : on ne garde que les mots de 3 lettres ou plus (a-z)
    words = re.findall(r'\b[a-z]{3,}\b', text_blob)
    
    # 4. Filtrage
    filtered_words = [w for w in words if w not in STOP_WORDS]
    
    # Comptage
    counter = Counter(filtered_words)
    most_common = counter.most_common(TOP_N_WORDS)

    # 5. Préparation de l'export
    data_export = []
    new_suggestions = 0

    for word, freq in most_common:
        # Statut : Est-ce que ce mot est déjà une colonne dans votre 1_prepare.py ?
        if word in current_keywords:
            status = "DÉJÀ UTILISÉ"
        else:
            status = "NOUVEAU (À AJOUTER ?)"
            new_suggestions += 1
            
        data_export.append({
            "Mot": word,
            "Frequence": freq,
            "Statut": status
        })

    # 6. Sauvegarde CSV
    df_export = pd.DataFrame(data_export)
    df_export.to_csv(OUTPUT_FILE, index=False, sep=";", encoding="utf-8-sig")

    print("\n" + "="*50)
    print(f"✅ EXPORT TERMINÉ : '{OUTPUT_FILE}'")
    print(f"   - Mots analysés : {TOP_N_WORDS}")
    print(f"   - Suggestions potentielles : {new_suggestions}")
    print("="*50)
    print("👉 Ouvrez ce fichier CSV.")
    print("👉 Les mots marqués 'NOUVEAU' avec une haute fréquence sont les meilleurs candidats")
    print("   pour enrichir votre liste 'KEYWORDS_COLUMNS' dans 1_prepare.py !")

if __name__ == "__main__":
    audit_vocabulary()