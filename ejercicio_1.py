import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ruta del dataset (mismo directorio que este archivo)
DATA_PATH = Path(__file__).parent / "Titanic-Dataset.csv"


# [gians] cachea el resultado de la función
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Carga el CSV y devuelve un DataFrame. Lanza excepción si falla."""
    # paso. cargar dataset con pandas
    return pd.read_csv(path)


def ejercicio_1():
    st.header("Ejercicio 1 — Titanic Dataset")

    if not DATA_PATH.exists():
        st.error(f"No se encuentra el archivo de datos en: {DATA_PATH}")
        st.write(
            "Asegúrate de colocar `Titanic-Dataset.csv` en la misma carpeta que `app.py`."
        )
        return

    with st.spinner("Cargando datos..."):
        try:
            df = load_data(DATA_PATH)

            st.subheader("Vista inicial del dataset")
            st.dataframe(df.head(50))

            # paso. eliminar columnas innecesarias
            df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

            # paso. rellenar valores nulos
            # para variables numéricas (Age, Fare) usar la media
            df["Age"].fillna(df["Age"].mean(), inplace=True)
            df["Fare"].fillna(df["Fare"].mean(), inplace=True)

            # para variables categóricas (Embarked) usar la moda
            df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

            # Separar features (X) y variable objetivo (y)
            y = df["Survived"].values  # Variable dependiente
            X = df.drop(
                ["Survived", "PassengerId"], axis=1
            ).values  # Variables independientes

            # Codificar variables categóricas

            # Codificar la columna 'Sex' con LabelEncoder (es binaria)
            le_sex = preprocessing.LabelEncoder()
            X[:, 1] = le_sex.fit_transform(X[:, 1])

            # Codificar la columna 'Embarked' con OneHotEncoder
            ct = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        OneHotEncoder(categories="auto", sparse_output=False),
                        [6],
                    )
                ],
                remainder="passthrough",
            )
            X = ct.fit_transform(X)

            # Reordenar columnas después del OneHotEncoder
            # Las nuevas columnas de Embarked están al principio, las movemos al final
            X = np.hstack([X[:, 3:], X[:, 0:3]])

            # Escalado de variables numéricas (Age y Fare están en las posiciones 3 y 6)

            sc_X = StandardScaler()
            # Escalamos solo Age y Fare
            X[:, [2, 5]] = sc_X.fit_transform(X[:, [2, 5]])

            # Convertir de nuevo a DataFrame para visualización
            columns = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked_C",
                "Embarked_Q",
                "Embarked_S",
            ]
            df = pd.DataFrame(X, columns=columns)
            df.insert(0, "Survived", y)  # Añadir Survived al inicio

            st.subheader("Vista previa del dataset después del preprocesamiento")
            st.markdown("""
            Se han realizado las siguientes transformaciones:
            1. Codificación de variables categóricas:
               - Sex: LabelEncoder (female=0, male=1) por ser binaria
               - Embarked: OneHotEncoder (3 columnas dummy: C, Q, S)
            2. Escalado de variables numéricas con StandardScaler:
               - Age: Media 0 y desviación estándar 1
               - Fare: Media 0 y desviación estándar 1
            """)
            st.markdown("**Primeros 5 registros después del preprocesamiento:**")
            st.dataframe(df.head(5))

            # División en conjuntos de entrenamiento y prueba

            # Separar features y target
            X = df.drop("Survived", axis=1)
            y = df["Survived"]

            # Realizar la división 70-30
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30, random_state=42
            )

            st.subheader("Dimensiones de los conjuntos de entrenamiento y prueba")
            st.markdown(f"""
            **Conjunto de entrenamiento (70%):**
            - X_train shape: {X_train.shape}
            - y_train shape: {y_train.shape}
            
            **Conjunto de prueba (30%):**
            - X_test shape: {X_test.shape}
            - y_test shape: {y_test.shape}
            """)

        except Exception as e:
            st.exception(e)
            return
