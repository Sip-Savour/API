# API Sip&Savour

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## **API Sip&Savour** est une API de machine learning permettant de prédire les vins optimaux en fonction des indicateurs de gouts necessaires et du type de vin souhaiter.

## 📦 Installation

### Depuis les sources

```bash
git clone git@github.com:Sip-Savour/API.git
cd API/
```

### Dépendances

```bash
pip install -r requirements.txt
```

**Dépendances principales :**

- numpy >= 1.21
- pandas >= 1.3
- scikit-learn >= 1.0
- optuna >= 3.0
- xgboost >= 1.5
- lightgbm >= 3.3
- catboost >= 1.0

## 🔄 Ordre d'execution des programmes:

```
1 - Prepare.py
Permet de préparer les données pour l'entrainement
2 - train_recom.py
Permet de preparer le fichier de recommendation KNN
3- train.py
Permet d'entrainer le modèle de prédiction
4- test_fast.py
Permet de tester les resultats du modèles et sa vitesse d'execution
5- migration.py
Effectue la migration des donnees csv en une base de données.
Initialise également la base de donnée
6- api.py
Lancement de l'api sur le port 8000
```

---

## 📁 Format des données

| No                                | country                           | description                                          | designation                                                                 | points                                                  | price                             | province                                    | region_1                                     | region_2                                                                       | variety                                  |
| --------------------------------- | --------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------- | --------------------------------- | ------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------- |
| Number of the wine in the dataset | The country that the wine is from | A few sentences from a sommelier describing the wine | The vineyard within the winery where the grapes that made the wine are from | The number of points rated the wine on a scale of 1-100 | The cost for a bottle of the wine | The province or state that the wine is from | The wine growing area in a province or state | Sometimes there are more specific regions specified within a wine growing area | The type of grapes used to make the wine |

## 📂 Structure du projet

```
API/
├── automl
├── data
│   ├── archive.zip
│   ├── audit_cepages.csv
│   ├── audit_vocabulary.csv
│   ├── data_train.csv
│   ├── sommelier.db
│   ├── wine_colors.json
│   ├── winemag-data-130k-v2.csv
│   ├── winemag-data-130k-v2.json
│   ├── winemag-data_first150k.csv
│   └── wines_db_full.csv
├── generated_files
│   ├── automl
│   └── pkl
├── python
│   ├── 1_prepare.py
│   ├── 2_1_train_recom.py
│   ├── 2_2_train.py
│   ├── 3_1_test_fast.py
│   ├── 4_1_audit_cepage.py
│   ├── 4_2_audit_vocab.py
│   ├── 5_migration.py
│   ├── api.py
│   ├── database.py
│   └── **pycache**
├── README.md
├── requirements.txt
├── runner
└── tests
```

---

## 🧪 Exemple de requêtes de recherches

```
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": "steak grilled pepper smoke", "color": "red"}'
```

---

## 🧪 Exemple de résultats

```
{
  "cepage": "Syrah",
  "bottle": {
    "title": "Domaine X Syrah 2015",
    "description": "A peppery and smoky wine...",
    "variety": "Syrah"
  }
}
```

---

## 🤝 Contribuer

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit (`git commit -m 'Ajout fonctionnalité X'`)
4. Push (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

---

## 📄 Licence

MIT License - voir [LICENSE](LICENSE) pour plus de détails.

---

## 👥 Auteurs

- **Aymeric** - _Développement_

## 🙏 Remerciements

- [wine-dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews/data?select=winemag-data_first150k.csv)
