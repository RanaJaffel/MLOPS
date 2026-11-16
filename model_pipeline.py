
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib


def prepare_data(data_path='Churn_Modelling.csv'):

    # 1. Charger les données
    data = pd.read_csv(data_path)

    # 2. Supprimer les colonnes inutiles
    data = data.drop(['Surname', 'Geography'], axis=1)

    # 3. Encoder la colonne Gender
    encoder = LabelEncoder()
    data['Gender'] = encoder.fit_transform(data['Gender'])

    # 4. Supprimer les valeurs manquantes
    data = data.dropna()

    # 5. Séparer les caractéristiques et la cible
    X = data.drop(['Exited', 'RowNumber', 'CustomerId'], axis=1)
    y = data['Exited']

    # 6. Séparer les données en train et test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # 7. Normaliser les données
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 8. Sauvegarder le scaler
    joblib.dump(scaler, 'scaler.joblib')

    # 9. Retourner les données préparées
    return x_train_scaled, x_test_scaled, y_train, y_test



def train_model(x_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=1
    )

    model.fit(x_train, y_train)

    return model



def evaluate_model(model, x_test, y_test):

    # Faire les prédictions
    y_pred = model.predict(x_test)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")

    return accuracy




def save_model(model, model_path='model.joblib'):

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")




def load_model(model_path='model.joblib'):

    model = joblib.load(model_path)

    print(f"Model loaded from {model_path}")

    return model



