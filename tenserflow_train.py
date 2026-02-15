import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib


def predire_categorie(model, tfidf, scaler, label_encoder, montant, libelle):
    # Vectorisation du libellé
    X_libelle = tfidf.transform([libelle])
    # Normalisation du montant
    X_montant = scaler.transform([[float(montant)]])
    # Combinaison des features
    X = hstack([X_libelle, X_montant]).toarray()
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    # Prédiction
    probabilites = model.predict(X_tensor, verbose=0)[0]
    categorie_encodee = np.argmax(probabilites)
    score_confiance = probabilites[categorie_encodee] * 100
    return label_encoder.inverse_transform([categorie_encodee])[0], score_confiance


# Charger le dataset
df = pd.read_csv('__data/airgap_dataset.csv')

# Préparation des données
X_libelle = df['libelle']
X_montant = df[['montant']].values
y = df['categorie']

# Vectorisation des libellés
tfidf = TfidfVectorizer()
X_libelle_vectorized = tfidf.fit_transform(X_libelle)

# Normalisation du montant
scaler = StandardScaler()
X_montant_scaled = scaler.fit_transform(X_montant)

# Combinaison des features
X = hstack([X_libelle_vectorized, X_montant_scaled])

# Encodage des catégories
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Conversion en tenseurs TensorFlow
X_train_tensor = tf.convert_to_tensor(X_train.toarray(), dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test.toarray(), dtype=tf.float32)

# Construction du modèle
input_layer = Input(shape=(X_train_tensor.shape[1],))
dense_layer = Dense(64, activation='relu')(input_layer)
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(dense_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train_tensor, y_train, validation_data=(X_test_tensor, y_test), epochs=10, batch_size=32)

# Évaluation
loss, accuracy = model.evaluate(X_test_tensor, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Sauvegarde du modele
model.save('modele_transactions.keras')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')