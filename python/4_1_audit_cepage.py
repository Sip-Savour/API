import pandas as pd
import json
import os

# CONFIGURATION
DATA_DIR = "data/"
INPUT_DB = DATA_DIR+"wines_db_full.csv"
JSON_FILE = DATA_DIR+"wine_colors.json"
OUTPUT_FILE = DATA_DIR+"audit_cepages.csv"  # Le fichier de sortie demandé

def audit_and_export():
    print(f"📊 Démarrage de l'audit des cépages...")

    # 1. Chargement de la base de données
    if not os.path.exists(INPUT_DB):
        print(f"❌ Erreur : {INPUT_DB} introuvable. Lancez 1_prepare.py d'abord.")
        return

    # Chargement robuste (parfois le séparateur change ou des lignes sont mal formées)
    try:
        df = pd.read_csv(INPUT_DB, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    # Vérification colonne
    if 'variety' not in df.columns:
        print("❌ Colonne 'variety' absente du CSV.")
        return

    # Compte des fréquences (du plus fréquent au plus rare)
    counts = df['variety'].value_counts()
    print(f"   > {len(counts)} cépages différents trouvés dans {len(df)} bouteilles.")

    # 2. Chargement du JSON existant (pour comparaison)
    known_colors = {}
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                known_colors = json.load(f)
            print(f"   > Comparaison avec wine_colors.json ({len(known_colors)} connus).")
        except:
            print("   ⚠️ Erreur lecture JSON, on suppose vide.")
    else:
        print("   ⚠️ wine_colors.json introuvable.")

    # 3. Préparation des données pour l'export
    data_export = []
    
    missing_count = 0
    
    for variety, count in counts.items():
        # On regarde si on connait la couleur
        color = known_colors.get(variety, "NON DÉFINI")
        
        status = "OK" if color != "NON DÉFINI" else "A FAIRE"
        if status == "A FAIRE":
            missing_count += 1

        data_export.append({
            "Frequence": count,
            "Cepage": variety,
            "Couleur_Actuelle": color,
            "Statut": status
        })

    # 4. Écriture du fichier CSV
    df_export = pd.DataFrame(data_export)
    
    # On sauvegarde en UTF-8 avec BOM pour qu'Excel ouvre les accents correctement
    df_export.to_csv(OUTPUT_FILE, index=False, sep=";", encoding="utf-8-sig")

    print("\n" + "="*50)
    print(f"✅ EXPORT TERMINÉ : '{OUTPUT_FILE}'")
    print(f"   - Total cépages : {len(df_export)}")
    print(f"   - Déjà configurés : {len(df_export) - missing_count}")
    print(f"   - Manquants ('A FAIRE') : {missing_count}")
    print("="*50)
    print("👉 Ouvrez ce fichier dans Excel, filtrez sur 'A FAIRE' et complétez votre JSON !")

if __name__ == "__main__":
    audit_and_export()
