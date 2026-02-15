# 🏦 API de Classification de Transactions Bancaires
## POC by Nathan F. (utilisation de Mistral AI pour la génération du code & Readme)
Cette API permet de prédire la catégorie d'une transaction bancaire à partir de son libellé et de son montant, en utilisant un modèle de machine learning entraîné avec TensorFlow.

---
## 📋 Prérequis
- Python 3.12 (testé et validé) ✅
---

## 🛠 Installation
Pour installer les dépendances, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

---

## 🚀 Lancer l'API
**⚠️ Attention :**
Il est **nécessaire** de lancer au préalable l'entraînement du modèle (`tensorflow_train.py`). Celui-ci générera les fichiers **nécessaires** pour l'API.

Pour démarrer l'API localement, exécutez la commande suivante dans le terminal :

```bash
uvicorn fast-api:app --reload
```

- L'API sera accessible à l'adresse : [http://127.0.0.1:8000](http://127.0.0.1:8000) 🌐
- La documentation interactive (Swagger UI) est disponible ici : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 📚

---

## 📡 Utilisation de l'API

### Endpoint : `POST /predire`

**Description** : Prédit la catégorie d'une transaction bancaire.

**Requête** :
- **URL** : `http://127.0.0.1:8000/predire`
- **Méthode** : `POST`
- **Headers** :
  - `Content-Type: application/json`
- **Body** (JSON) :
  ```json
  {
    "libelle": "Paiement Amazon Livraison",
    "montant": 49.99
  }
  ```

**Exemple avec `curl`** :

```bash
curl -X POST "http://127.0.0.1:8000/predire" -H "Content-Type: application/json" -d '{"libelle":"Paiement Amazon Livraison","montant":49.99}'
```

**Réponse attendue** :
```json
{
  "categorie": "Shopping",
  "score_confiance": 95.47
}
```

---

## 📌 Remarques

- Vérifiez que les fichiers `.h5` et `.pkl` sont bien présents dans le dossier avant de lancer l'API.
- Pour arrêter l'API, utilisez `CTRL+C` dans le terminal.

