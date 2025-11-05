import streamlit as st
import pandas as pd
from pathlib import Path
import io
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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
            from sklearn.preprocessing import StandardScaler

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
            from sklearn.model_selection import train_test_split

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


def ejercicio_2():
    st.header("Ejercicio 2 — Student Performance Dataset")

    # Ruta del dataset de estudiantes
    STUDENT_DATA_PATH = Path(__file__).parent / "student-mat.csv"

    if not STUDENT_DATA_PATH.exists():
        st.error(f"No se encuentra el archivo de datos en: {STUDENT_DATA_PATH}")
        st.write(
            "Asegúrate de colocar `student-mat.csv` en la misma carpeta que `app.py`."
        )
        return

    with st.spinner("Cargando datos..."):
        try:
            # Leer el dataset con pandas
            df_students = pd.read_csv(STUDENT_DATA_PATH, sep=",")

            # columnas:
            # school,sex,age,address,famsize,Pstatus,Medu,Fedu,Mjob,Fjob,reason,guardian,traveltime,studytime,failures,schoolsup,famsup,paid,activities,nursery,higher,internet,romantic,famrel,freetime,goout,Dalc,Walc,health,absences,G1,G2,G3
            
            st.subheader("Vista inicial del dataset de estudiantes")
            st.dataframe(df_students.head(50))

            # 1. Análisis y eliminación de duplicados
            st.subheader("1. Análisis de duplicados")
            num_duplicados = df_students.duplicated().sum()
            st.write(f"Número de filas duplicadas encontradas: {num_duplicados}")
            
            if num_duplicados > 0:
                df_students = df_students.drop_duplicates()
                st.success(f"Se eliminaron {num_duplicados} filas duplicadas")

            # 2. Análisis de valores inconsistentes
            st.subheader("2. Análisis de valores inconsistentes")

            # 2.1 Verificar rangos de edad (deberían ser razonables para estudiantes)
            edad_invalida = df_students[~df_students['age'].between(13, 25)].shape[0]
            st.write(f"Registros con edad fuera de rango (13-25 años): {edad_invalida}")
            if edad_invalida > 0:
                df_students = df_students[df_students['age'].between(13, 25)]

            # 2.2 Verificar calificaciones (G1, G2, G3 deberían estar entre 0 y 20)
            notas_invalidas = df_students[
                ~(df_students['G1'].between(0, 20) & 
                  df_students['G2'].between(0, 20) & 
                  df_students['G3'].between(0, 20))
            ].shape[0]
            st.write(f"Registros con calificaciones fuera de rango (0-20): {notas_invalidas}")
            if notas_invalidas > 0:
                df_students = df_students[
                    df_students['G1'].between(0, 20) & 
                    df_students['G2'].between(0, 20) & 
                    df_students['G3'].between(0, 20)
                ]

            # 2.3 Verificar valores categóricos
            st.write("\nValores únicos en columnas categóricas:")
            categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                              'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                              'famsup', 'paid', 'activities', 'nursery', 'higher', 
                              'internet', 'romantic']
            
            for col in categorical_cols:
                unique_values = df_students[col].unique()
                st.write(f"{col}: {sorted(unique_values)}")

            # 2.4 Verificar valores nulos
            st.write("\nValores nulos por columna:")
            null_counts = df_students.isnull().sum()
            if null_counts.sum() > 0:
                st.write(null_counts[null_counts > 0])
            else:
                st.write("No se encontraron valores nulos en el dataset")

            # Mostrar dimensiones finales del dataset
            st.subheader("Dimensiones finales del dataset")
            st.write(f"Forma del dataset después de la limpieza: {df_students.shape}")
            
            # 3. Codificación de variables categóricas con OneHotEncoder
            st.subheader("3. Codificación de variables categóricas")

            # Definir las columnas categóricas y numéricas
            categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                                'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                                'famsup', 'paid', 'activities', 'nursery', 'higher', 
                                'internet', 'romantic']

            # Obtener las columnas numéricas (las que no son categóricas)
            numeric_cols = df_students.columns.difference(categorical_cols).tolist()

            # Aplicar OneHotEncoder a las columnas categóricas
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_data = encoder.fit_transform(df_students[categorical_cols])

            # Obtener los nombres de las características codificadas
            encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

            # Crear un DataFrame con las características codificadas
            df_categorical = pd.DataFrame(
                encoded_data,
                columns=encoded_feature_names,
                index=df_students.index
            )

            # Combinar con las variables numéricas
            df_encoded = pd.concat([df_students[numeric_cols], df_categorical], axis=1)

            # Mostrar información sobre la codificación
            st.markdown("""
            Se ha realizado la codificación One-Hot de las siguientes variables categóricas:
            - Variables escolares: school, reason
            - Variables personales: sex, address
            - Variables familiares: famsize, Pstatus, Mjob, Fjob, guardian
            - Variables de apoyo: schoolsup, famsup, paid, activities, nursery
            - Otras variables: higher, internet, romantic

            Las variables numéricas se mantienen sin cambios.
            """)

            # Vista previa del dataset con variables codificadas
            st.subheader("Vista previa del dataset con variables codificadas")
            st.write(f"Dimensiones del dataset codificado: {df_encoded.shape}")
            st.write("Número de características después de la codificación:", df_encoded.shape[1])
            st.dataframe(df_encoded.head(50))

        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")


def ejercicio_3():
    st.header("Ejercicio 3 — ")
    st.markdown(
        "En esta página añadiremos un ejemplo de modelo (p. ej. clasificación con sklearn), selección de características y evaluación."
    )
    st.info(
        "Pendiente: implementar pipeline simple con división train/test, entrenamiento y métricas."
    )


def main():
    st.title("Proyecto de Datasets Inteligentes con ML")

    page = st.sidebar.selectbox(
        "Selecciona una página:",
        [
            "Ejercicio 1",
            "Ejercicio 2",
            "Ejercicio 3",
        ],
    )

    if page == "Ejercicio 1":
        ejercicio_1()
    elif page == "Ejercicio 2":
        ejercicio_2()
    elif page == "Ejercicio 3":
        ejercicio_3()


if __name__ == "__main__":
    main()
