import pandas as pd
import joblib
import os
import re
from collections import Counter

# ================= CONFIGURATION =================
DATA_DIR = "data/"
GENERATED_DIR = "generated_files/pkl/"
INPUT_DB = DATA_DIR + "wines_db_full.csv"
KEYWORDS_FILE = GENERATED_DIR + "keywords_list.pkl"
OUTPUT_FILE = DATA_DIR + "audit_vocabulary.csv"

# J'augmente un peu le nombre de mots analysés au départ (ex: 2000)
# pour être sûr d'en trouver assez de "nouveaux" une fois les anciens filtrés.
TOP_N_WORDS = 2000 

STOP_WORDS = set([
    "the", "and", "a", "of", "with", "is", "in", "it", "to", "that", "this", 
    "for", "on", "as", "are", "by", "an", "at", "be", "has", "from", "its",
    "but", "not", "or", "have", "some", "very", "more", "now", "up", "can",
    "wine", "flavors", "notes", "palate", "finish", "nose", "aromas", "drink", 
    "years", "bottle", "glass", "show", "shows", "well", "good", "made", 
    "blend", "vineyard", "vintage", "character", "style", "bottling", "opens",
    "through", "offers", "hint", "hints", "touch", "bit", "gives", "named"
])

def audit_vocabulary():
    print(f"📚 Audit du vocabulaire (Mots INCONNUS uniquement)...")

    # 1. Chargement des données
    if not os.path.exists(INPUT_DB):
        print(f"❌ Erreur : {INPUT_DB} introuvable.")
        return

    try:
        df = pd.read_csv(INPUT_DB, on_bad_lines='skip', low_memory=False)
    except:
        print("⚠️ Erreur lecture CSV standard.")
        return

    # 2. Chargement des mots-clés actuels (ceux à EXCLURE)
    current_keywords = set()
    if os.path.exists(KEYWORDS_FILE):
        try:
            current_keywords = set(joblib.load(KEYWORDS_FILE))
            print(f"   > Filtre actif : {len(current_keywords)} mots déjà connus seront ignorés.")
        except:
            print("   ⚠️ Impossible de lire la liste actuelle.")
    else:
        print("   ⚠️ Aucune liste de mots-clés existante (tout sera considéré nouveau).")

    # 3. Traitement du texte
    print("   > Tokenization et comptage...")
    text_blob = " ".join(df['description'].dropna().astype(str).tolist()).lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_blob)
    
    # 4. Filtrage des Stop Words
    filtered_words = [w for w in words if w not in STOP_WORDS]
    
    # Comptage
    counter = Counter(filtered_words)
    most_common = counter.most_common(TOP_N_WORDS)

    # 5. Préparation de l'export (FILTRAGE STRICT)
    data_export = []
    
    for word, freq in most_common:
        # --- MODIFICATION ICI ---
        # Si le mot est déjà connu, on PASSE au suivant (on ne l'enregistre pas)
        if word in current_keywords:
            continue 
            
        # Si on arrive ici, c'est forcément un nouveau mot
        data_export.append({
            "Mot": word,
            "Frequence": freq,
            "Statut": "NOUVEAU" # Plus besoin de préciser, ils sont tous nouveaux
        })

    # 6. Sauvegarde CSV
    if data_export:
        df_export = pd.DataFrame(data_export)
        df_export.to_csv(OUTPUT_FILE, index=False, sep=";", encoding="utf-8-sig")

        print("\n" + "="*50)
        print(f"✅ EXPORT TERMINÉ : '{OUTPUT_FILE}'")
        print(f"   - Mots analysés (Top N) : {TOP_N_WORDS}")
        print(f"   - Mots déjà connus (filtrés) : {len(most_common) - len(df_export)}")
        print(f"   - Nouvelles suggestions : {len(df_export)}")
        print("="*50)
    else:
        print("⚠️ Aucun nouveau mot trouvé dans le Top N !")

if __name__ == "__main__":
    audit_vocabulary()