#!/bin/bash
#SBATCH --job-name=5_api
#SBATCH --output=automl_output.log
#SBATCH --error=automl_error.log
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Aymeric.Mabire.Etu@univ-lemans.fr

# =========================================================
# 1. NAVIGATION VERS LE CODE (Chemin Absolu)
# =========================================================
# On force le script à aller là où est api.py
cd /info/etu/m1/s2203089/API/python

echo "📍 Dossier de travail actuel : $(pwd)"

# Petite vérification pour le log
if [ -f "api.py" ]; then
    echo "✅ Fichier api.py trouvé."
else
    echo "❌ ERREUR : api.py est introuvable ici !"
    ls -la
    exit 1
fi

# =========================================================
# 2. ACTIVATION ENVIRONNEMENT
# =========================================================
# On cherche le venv. 
# Si votre venv est dans le dossier API (un cran au-dessus), on fait ../
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Venv activé (local)."
elif [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
    echo "✅ Venv activé (parent)."
else
    echo "⚠️ ATTENTION : Venv non trouvé automatiquement."
    # Mettez ici le chemin absolu si les deux précédents échouent
    # source /info/etu/m1/s2203089/API/python/venv/bin/activate
fi

# =========================================================
# 3. LANCEMENT API
# =========================================================
echo "🚀 Lancement Uvicorn sur le noeud $(hostname)..."

# Lancement bloquant (pas de nohup, pas de &)
uvicorn api:app --host 0.0.0.0 --port 8000
