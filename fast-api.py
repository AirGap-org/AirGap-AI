from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
from scipy.sparse import hstack

# Charger le modèle et les objets
model = tf.keras.models.load_model('modele_transactions.keras')
tfidf = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = FastAPI()

class Transaction(BaseModel):
    libelle: str
    montant: float

@app.post("/predire")
def predire(transaction: Transaction):
    # Vectorisation du libellé
    X_libelle = tfidf.transform([transaction.libelle])
    # Normalisation du montant
    X_montant = scaler.transform([[transaction.montant]])
    # Combinaison des features
    X = hstack([X_libelle, X_montant]).toarray()
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    # Prédiction
    probabilites = model.predict(X_tensor, verbose=0)[0]
    categorie_encodee = np.argmax(probabilites)
    score_confiance = float(probabilites[categorie_encodee] * 100)
    categorie = label_encoder.inverse_transform([categorie_encodee])[0]
    return {"categorie": categorie, "score_confiance": score_confiance}

# Pour lancer l'API : uvicorn app:app --reload
