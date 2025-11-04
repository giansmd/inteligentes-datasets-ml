import streamlit as st
import pandas as pd
from pathlib import Path
import io
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn import preprocessing

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
        st.write("Asegúrate de colocar `Titanic-Dataset.csv` en la misma carpeta que `app.py`.")
        return

    with st.spinner("Cargando datos..."):
        try:
            df = load_data(DATA_PATH)
            # paso. eliminar columnas innecesarias
            df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
            
            # paso. rellenar valores nulos
            # para variables numéricas (Age, Fare) usar la media
            df['Age'].fillna(df['Age'].mean(), inplace=True)
            df['Fare'].fillna(df['Fare'].mean(), inplace=True)
            
            # para variables categóricas (Embarked) usar la moda
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

            # Separar features (X) y variable objetivo (y)
            y = df['Survived'].values  # Variable dependiente
            X = df.drop(['Survived', 'PassengerId'], axis=1).values  # Variables independientes

            # Primero usamos LabelEncoder para convertir categorías a números
            
            # Codificar la columna 'Sex' (está en la posición 1 de X)
            le_sex = preprocessing.LabelEncoder()
            X[:, 1] = le_sex.fit_transform(X[:, 1])
            
            # Codificar la columna 'Embarked' (está en la posición 6 de X)
            le_embarked = preprocessing.LabelEncoder()
            X[:, 6] = le_embarked.fit_transform(X[:, 6])

            # Luego usamos OneHotEncoder solo para la columna 'Sex'
            ct = ColumnTransformer([
                ('one_hot_encoder', OneHotEncoder(categories='auto'), [1])
            ], remainder='passthrough')
            
            X = np.array(ct.fit_transform(X), dtype=np.float64)
            
            # Convertir de nuevo a DataFrame para visualización
            columns = ['Sex_0', 'Sex_1'] + ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            df = pd.DataFrame(X, columns=columns)
            df.insert(0, 'Survived', y)  # Añadir Survived al inicio

        except Exception as e:
            st.exception(e)
            return

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(100))

def ejercicio_2():
    st.header("Ejercicio 2 — Exploración (placeholder)")
    st.markdown("En esta página añadiremos visualizaciones y análisis exploratorio: conteos por clase, sobrevivientes por sexo/edad, gráficos, etc.")
    st.info("Pendiente: implementar visualizaciones con Altair/Matplotlib/Seaborn y controles interactivos.")


def ejercicio_3():
    st.header("Ejercicio 3 — Modelado (placeholder)")
    st.markdown("En esta página añadiremos un ejemplo de modelo (p. ej. clasificación con sklearn), selección de características y evaluación.")
    st.info("Pendiente: implementar pipeline simple con división train/test, entrenamiento y métricas.")


def main():
    st.title("Proyecto: Titanic — 3 ejercicios")

    page = st.sidebar.selectbox("Selecciona una página:", [
        "Ejercicio 1 - Lectura",
        "Ejercicio 2 - Exploración",
        "Ejercicio 3 - Modelado"
    ])

    if page == "Ejercicio 1 - Lectura":
        ejercicio_1()
    elif page == "Ejercicio 2 - Exploración":
        ejercicio_2()
    else:
        ejercicio_3()


if __name__ == '__main__':
    main()
