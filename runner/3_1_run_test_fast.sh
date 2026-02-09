#!/bin/bash
#SBATCH --job-name=3_Test_Fast         # Nom du job
#SBATCH --output=logs/test_output.log    # Fichier de sortie standard
#SBATCH --error=logs/test_error.log      # Fichier d'erreur
#SBATCH --time=48:00:00               # Temps max d'exécution (hh:mm:ss)
#SBATCH --ntasks=1                    # Nombre de tâches (1 pour python)
#SBATCH --cpus-per-task=16             # Nombre de threads / CPU à utiliser
#SBATCH --mem=128G                     # Mémoire allouée
#SBATCH --mail-type=END,FAIL               # Quand envoyer un mail (END, FAIL, ALL)
#SBATCH --mail-user=Aymeric.Mabire.Etu@univ-lemans.fr  # Ton adresse mail
# --- Charger l'environnement Python ---

# --- Exécution du script Python ---
echo "=== Lancement AutoML ==="
python app/3_1_test_fast.py
echo "=== Fin AutoML ==="
