#!/bin/bash

# Script de création d'archive pour le build Docker de l'API Vin
# Crée une archive tar contenant tous les fichiers nécessaires.
# Valide la présence des fichiers avant la création pour garantir la cohérence du build.

echo "Création de l'archive tar pour l'API..."

# Liste des fichiers requis pour la création de l'image Docker
REQUIRED_FILES=(
    "alembic"
    "app"
    "automl"
    "data"
    "generated_files"
    "alembic.ini"
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    "sommelier.db"
)

echo "Validation de la présence des fichiers requis..."
MISSING_FILES=()

# Vérifie que chaque fichier/dossier existe
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$file" ]; then
        MISSING_FILES+=("$file")
        echo "❌ Manquant : $file"
    else
        echo "✅ Trouvé : $file"
    fi
done

# Arrêt avec erreur si des fichiers manquent
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo "❌ Erreur : Fichiers requis manquants détectés :"
    printf '%s\n' "${MISSING_FILES[@]}"
    exit 1
fi

# Crée l'archive tar avec tous les fichiers requis
echo "Création de l'archive API.tar..."
tar -cf API.tar "${REQUIRED_FILES[@]}"

# Valide la création de l'archive et affiche les résultats
if [ $? -eq 0 ]; then
    echo "✅ Archive créée avec succès !"
    echo "Contenu de l'archive :"
    tar -tf API.tar
    echo ""
    echo "Taille de l'archive :"
    ls -lh API.tar
else
    echo "❌ Une erreur est survenue pendant la création de l'archive"
    exit 1
fi