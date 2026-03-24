#!/bin/bash
#SBATCH --job-name=4_2_Audit_Vocabulaire         # Nom du job
#SBATCH --output=logs/audit/audit_vocabulaire_output.log    # Fichier de sortie standard
#SBATCH --error=logs/audit/audit_vocabulaire_error.log      # Fichier d'erreur
#SBATCH --time=48:00:00               # Temps max d'exécution (hh:mm:ss)
#SBATCH --ntasks=1                    # Nombre de tâches (1 pour python)
#SBATCH --cpus-per-task=16             # Nombre de threads / CPU à utiliser
#SBATCH --mem=128G                     # Mémoire allouée
#SBATCH --mail-type=END,FAIL               # Quand envoyer un mail (END, FAIL, ALL)
#SBATCH --mail-user=Aymeric.Mabire.Etu@univ-lemans.fr  # Ton adresse mail
# --- Charger l'environnement Python ---

# --- Exécution du script Python ---
echo "=== Lancement AutoML ==="
python app/config/4_2_audit_vocab.py
echo "=== Fin AutoML ==="
