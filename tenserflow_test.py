import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input

from tenserflow_train import predire_categorie

model = tf.keras.models.load_model('modele_transactions.h5')
tfidf = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

df_nouveau = pd.read_csv('__data/aldwin_out.csv')

# Prédire pour chaque transaction
for index, row in df_nouveau.iterrows():
    categorie, score = predire_categorie(model, tfidf, scaler, label_encoder, row["montant"], row["libelle"])
    print(
        f"Transaction: {row['libelle']} ({row['montant']}€) → Catégorie prédite: {categorie} (confiance: {score:.2f}%)")
