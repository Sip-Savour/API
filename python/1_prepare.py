import pandas as pd
import numpy as np
import os
import joblib

# ================= CONFIGURATION =================
DATA_DIR = "data/"
GENERATED_ML_DIR = "generated_files/automl/"
GENERATED_PKL_DIR = "generated_files/pkl/"
INPUT_CSV = DATA_DIR + "winemag-data_first150k.csv"
OUTPUT_CSV = DATA_DIR + "wines_db_full.csv"
BASENAME  = "wine_train"

GENERATED_DIR = GENERATED_ML_DIR


KEYWORD_GROUPS = {
    # --- COULEUR & FRUITS ---
    "red_fruit":    ["red", "cherry", "raspberry", "strawberry", "cranberry", "pomegranate", "currant", "rhubarb", "watermelon", "sour cherry"],
    "black_fruit":  ["black", "blackberry", "cassis", "plum", "dark fruit", "blueberry", "bramble", "boysenberry", "black cherry", "blackcurrant"],
    "dried_fruit":  ["raisin", "prune", "fig", "date", "dried fruit", "cooked fruit", "candied", "jammy"],
    "citrus":       ["citrus", "lemon", "lime", "grapefruit", "orange", "mandarin", "tangerine", "zest", "rind", "yuzu", "bergamot"],
    "tropical":     ["tropical", "pineapple", "melon", "mango", "papaya", "passion", "lychee", "guava", "banana", "kiwi", "mangoes"],
    "tree_fruit":   ["apple", "pear", "peach", "apricot", "nectarine", "quince", "yellow fruit"],
    "gooseberry":   ["gooseberry"], 

    # --- SUCRE & STYLE ---
    "dry":          ["dry", "bone dry"], 
    "sweet":        ["sweet", "sugar", "honey", "lush", "syrup", "botrytis", "late harvest", "dessert", "off-dry", "maple"],
    
    # --- STRUCTURE ---
    "acidity":      ["acid", "acidity", "tart", "crisp", "bright", "lively", "fresh", "freshness", "zesty", "sour", "racy", "zippy", "electric", "nervous"],
    "tannins":      ["tannin", "tannins", "tannic", "firm", "chewy", "astringent", "grip", "structured", "muscular", "abrasive", "harsh", "gripping", "austere"], 
    "body_full":    ["bodied", "full", "heavy", "dense", "thick", "rich", "richness", "concentrated", "big", "fat", "oily", "viscous", "lush", "opulent", "extract", "fuller"],
    "body_light":   ["light", "elegant", "delicate", "thin", "airy", "lean", "watery", "dilute"],
    "texture_soft": ["smooth", "soft", "silky", "velvety", "creamy", "round", "supple", "polished", "plush", "seamless"],
    
    # --- BOISÉ, FUMÉ & ÉPICES ---
    "oak":          ["oak", "wood", "cedar", "barrel", "cask", "vanilla", "coconut", "woody", "sandalwood", "sawdust", "pine", "resin"], 
    "smoke_tobacco":["smoke", "smoky", "ash", "ashy", "charcoal", "tobacco", "cigar", "nicotine", "burnt", "charred", "roasted", "campfire", "incense", "soot"], 
    "pastry":       ["brioche", "dough", "yeast", "biscuit", "bread", "toast", "toasty", "butter", "cream", "butterscotch", "caramel", "toffee", "marzipan", "nougat", "praline", "cookie", "graham", "marshmallow"], 
    "spices":       ["spice", "spicy", "pepper", "peppery", "cinnamon", "clove", "nutmeg", "licorice", "anise", "cardamom", "ginger", "allspice", "asian spice"],
    "nutty":        ["nutty", "almond", "hazelnut", "walnut", "pecan", "chestnut", "oxidized", "sherry"],
    "cocoa":        ["chocolate", "cocoa", "mocha", "coffee", "espresso", "dark chocolate", "milk chocolate"],

    # --- VÉGÉTAL & HERBACÉ ---
    "herbal":       ["herb", "herbal", "green", "grass", "grassy", "leafy", "stem", "vegetal", "hay", "straw", "bramble", "fern", "weedy"],
    "aromatic_herb":["mint", "eucalyptus", "menthol", "sage", "thyme", "fennel", "dill", "rosemary", "lavender", "bay leaf", "basil", "oregano"],
    "vegetable":    ["bell pepper", "jalapeno", "capsicum", "olive", "green olive", "black olive", "tomato leaf", "asparagus", "green bean", "olives"],
    "floral":       ["floral", "flower", "blossom", "rose", "violet", "jasmine", "honeysuckle", "acacia", "chamomile", "white flower", "potpourri", "flowery"],
    
    # --- ERROIR, MINÉRAL & CHIMIQUE ---
    "earth":        ["earth", "earthy", "dirt", "soil", "dusty", "loam", "mushroom", "truffle", "forest floor", "underbrush", "compost", "wet leaves", "soils"],
    "mineral":      ["mineral", "minerality", "stone", "slate", "flint", "chalk", "chalky", "saline", "salty", "crushed rock", "limestone", "wet stone", "oyster shell", "granite", "stones", "sulfur", "gunpowder"], 
    "inorganic":    ["graphite", "pencil", "lead", "petrol", "diesel", "gasoline", "rubber", "tar", "asphalt", "plastic", "vinyl", "kerosene", "tarry"],
    "savory":       ["savory", "meaty", "bacon", "game", "leather", "animal", "cured meat", "sausage", "blood", "iron", "beef", "bouillon", "soy", "umami", "gamy"],
    "funky":        ["barnyard", "sweaty", "horse", "brett", "band-aid", "yeasty", "cheese", "wax", "beeswax", "lanolin", "wet wool", "funk", "cheesy"],

    # --- QUALITÉ & AGE ---
    "clean":        ["clean", "pure", "precise", "focused", "linear", "crystalline"],
    "complex":      ["complex", "complexity", "layered", "nuanced", "depth", "multidimensional", "intricate"],
    "age":          ["old", "aged", "mature", "developed", "tertiary", "evolved", "peak"],
    "finish_long":  ["long finish", "length", "lingering", "persistent", "endless", "persistence"]
}


def prepare():
    print(f"--- Chargement & Nettoyage ---")
    
    os.makedirs(GENERATED_PKL_DIR, exist_ok=True)
    os.makedirs(GENERATED_ML_DIR, exist_ok=True)

    if not os.path.exists(INPUT_CSV):
        print(f"Fichier {INPUT_CSV} manquant. Vérifiez le dossier data/.")
        return

    df = pd.read_csv(INPUT_CSV, on_bad_lines='skip', low_memory=False)
    
    if 'title' not in df.columns:
        print("   > Génération des titres (Winery + Variety)...")
        df['winery'] = df['winery'].fillna("Inconnu")
        df['variety'] = df['variety'].fillna("")
        df['title'] = df['winery'].astype(str) + " " + df['variety'].astype(str)

    # Nettoyage
    df = df.dropna(subset=['description', 'variety'])
    df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0).astype(int)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    
    print(f"   > Dataset de travail : {len(df)} vins.")

    # --- CRÉATION DES META-FEATURES ---
    print("--- Génération des 'Meta-Features' ---")
    
    descriptions = df['description'].str.lower()
    
    # On prépare les noms des colonnes finales
    final_columns = list(KEYWORD_GROUPS.keys())
    
    # Matrice de résultat
    X_matrix = np.zeros((len(df), len(final_columns)), dtype=int)
    
    for i, (col_name, synonyms) in enumerate(KEYWORD_GROUPS.items()):
        # ASTUCE : Regex 'OR' (\b = mot entier uniquement)
        # Exemple : "\bacid\b|\bacidity\b|\btart\b"
        pattern = '|'.join([f"\\b{word}\\b" for word in synonyms])
        
        # Détection vectorisée
        presence = descriptions.str.contains(pattern, regex=True).astype(int).values
        X_matrix[:, i] = presence
        
        count = np.sum(presence)
        if count > 0:
            print(f"   - Méta-colonne '{col_name}' : trouvée dans {count} vins")

    print(f"   > Matrice générée : {X_matrix.shape}")
    
    # SAUVEGARDE DES MOTS CLES
    joblib.dump(final_columns, GENERATED_PKL_DIR + "keywords_list.pkl")

    # SAUVEGARDE DU DICTIONNAIRE COMPLET (POUR L'API) - TRES IMPORTANT
    joblib.dump(KEYWORD_GROUPS, GENERATED_PKL_DIR + "keyword_groups.pkl")
    print(f"   > Mappage complet sauvegardé : keyword_groups.pkl")

    # --- SAUVEGARDE ---
    print(f"--- 3. Écriture des fichiers finaux ---")
    np.savetxt(f"{GENERATED_DIR + BASENAME}.data", X_matrix, fmt='%d')
    df['variety'].to_csv(f"{GENERATED_DIR + BASENAME}.solution", index=False, header=False)
    
    # Sauvegarde du CSV propre pour le KNN
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    prepare()