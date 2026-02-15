import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selectionpandas import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def predire_categorie(model, tfidf, label_encoder, montant, libelle):
    """
    Prédit la catégorie d'une transaction à partir de la date, du montant et du libellé.
    """
    # Prétraitement du libellé (identique à l'entraînement)
    X_libelle = tfidf.transform([libelle])

    # Combinaison des features (ici, seulement libellé et montant)
    X = hstack([X_libelle, [[float(montant)]]])

    # Prédiction
    probabilites = model.predict_proba(X)[0]
    categorie_encodee = model.predict(X)[0]

    score_confiance = probabilites[categorie_encodee] * 100  # Convertir en pourcentage

    return label_encoder.inverse_transform([categorie_encodee])[0], score_confiance


if __name__ == '__main__':
    df = pd.read_csv("../__data/airgap_dataset.csv")

    tfidf = TfidfVectorizer()
    X_libelle = tfidf.fit_transform(df['libelle'])

    label_encoder = LabelEncoder()
    df['categorie_encodee'] = label_encoder.fit_transform(df['categorie'])

    # y = donnée a prédire
    y = df['categorie_encodee']

    # X = features
    X = hstack([X_libelle, df[['montant']].values])

    # diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

    # model = RandomForestClassifier(n_estimators=100, random_state=50)
    model = XGBClassifier(n_estimators=200, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

    # tentatives de prédiction sur de nouvelles données

    df_aldwin = pd.read_csv("data_prep/export/aldwin_out.csv", delimiter=',', decimal='.')
    transactions_aleatoires = df_aldwin.sample(n=20)
    for index, row in transactions_aleatoires.iterrows():
        categorie, score = predire_categorie(model, tfidf, label_encoder, row["montant"],
                                             row["libelle"])
        print(
            f"Transaction: {row['libelle']} ({row['montant']}€) → Catégorie prédite: {categorie} (confiance: {score:.2f}%)")
