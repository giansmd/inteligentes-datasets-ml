import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


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

            st.subheader("Vista inicial del dataset de estudiantes")
            st.dataframe(df_students.head(50))

            # Análisis de correlación entre notas
            st.subheader("Análisis de correlación entre notas")
            
            # Crear matriz de correlación para las notas
            notas_cols = ['G1', 'G2', 'G3']
            corr_matrix = df_students[notas_cols].corr()
            
            # Mostrar la matriz de correlación
            st.write("Matriz de correlación entre notas:")
            st.dataframe(corr_matrix.style.format("{:.3f}"))
            
            # Visualización con heatmap
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Crear el heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       vmin=-1, 
                       vmax=1, 
                       center=0,
                       fmt='.3f')
            plt.title('Correlación entre notas G1, G2 y G3')
            st.pyplot(fig)
            
            # Interpretación de la correlación
            st.markdown("""
            **Interpretación de la correlación:**
            - La correlación va de -1 a 1, donde:
                - 1 indica una correlación positiva perfecta
                - -1 indica una correlación negativa perfecta
                - 0 indica que no hay correlación
            - Valores cercanos a 1 indican una fuerte correlación positiva
            - Se espera ver una correlación positiva entre las notas, ya que típicamente
              un buen rendimiento en evaluaciones previas (G1, G2) suele indicar
              un buen rendimiento en la evaluación final (G3)
            """)
            
            # 1. Análisis y eliminación de duplicados
            st.subheader("1. Análisis de duplicados")
            num_duplicados = df_students.duplicated().sum()
            st.write(f"Número de filas duplicadas encontradas: {num_duplicados}")

            if num_duplicados > 0:
                df_students = df_students.drop_duplicates()
                st.success(f"Se eliminaron {num_duplicados} filas duplicadas")

            # 2. Análisis de valores inconsistentes
            st.subheader("2. Análisis de valores inconsistentes")

            # 2.1 Verificar rangos de edad
            edad_invalida = df_students[~df_students["age"].between(13, 25)].shape[0]
            st.write(f"Registros con edad fuera de rango (13-25 años): {edad_invalida}")
            if edad_invalida > 0:
                df_students = df_students[df_students["age"].between(13, 25)]

            # 2.2 Verificar calificaciones
            notas_invalidas = df_students[
                ~(
                    df_students["G1"].between(0, 20)
                    & df_students["G2"].between(0, 20)
                    & df_students["G3"].between(0, 20)
                )
            ].shape[0]
            st.write(
                f"Registros con calificaciones fuera de rango (0-20): {notas_invalidas}"
            )
            if notas_invalidas > 0:
                df_students = df_students[
                    df_students["G1"].between(0, 20)
                    & df_students["G2"].between(0, 20)
                    & df_students["G3"].between(0, 20)
                ]

            # 2.3 Verificar valores categóricos
            st.write("\nValores únicos en columnas categóricas:")
            categorical_cols = [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "Mjob",
                "Fjob",
                "reason",
                "guardian",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ]

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
            categorical_cols = [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "Mjob",
                "Fjob",
                "reason",
                "guardian",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
            ]

            # Obtener las columnas numéricas
            numeric_cols = df_students.columns.difference(categorical_cols).tolist()

            # Aplicar OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_data = encoder.fit_transform(df_students[categorical_cols])

            # Obtener los nombres de las características codificadas
            encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

            # Crear DataFrame con características codificadas
            df_categorical = pd.DataFrame(
                encoded_data, columns=encoded_feature_names, index=df_students.index
            )

            # Combinar con variables numéricas
            df_encoded = pd.concat([df_students[numeric_cols], df_categorical], axis=1)

            st.markdown("""
            Se ha realizado la codificación One-Hot de las siguientes variables categóricas:
            - Variables escolares: school, reason
            - Variables personales: sex, address
            - Variables familiares: famsize, Pstatus, Mjob, Fjob, guardian
            - Variables de apoyo: schoolsup, famsup, paid, activities, nursery
            - Otras variables: higher, internet, romantic

            Las variables numéricas se mantienen sin cambios.
            """)

            # 4. Normalización de variables numéricas seleccionadas
            st.subheader("4. Normalización de variables numéricas")

            # Definir las columnas a normalizar
            columns_to_normalize = ["G1", "G2", "absences", "age"]

            # Crear y ajustar el StandardScaler
            scaler = StandardScaler()
            df_encoded[columns_to_normalize] = scaler.fit_transform(
                df_encoded[columns_to_normalize]
            )

            st.markdown("""
            Se han normalizado las siguientes variables numéricas usando StandardScaler:
            - G1 y G2 (notas previas)
            - absences (número de ausencias)
            - age (edad del estudiante)
            
            La variable G3 no se normaliza ya que será nuestra variable dependiente.
            
            Después de la normalización, estas variables tienen:
            - Media = 0
            - Desviación estándar = 1
            """)

            # 5. Separación de características (X) y variable objetivo (y)
            st.subheader("5. Separación de variables independientes y dependiente")

            # Separar la variable objetivo (G3) del resto de características
            y = df_encoded["G3"]
            X = df_encoded.drop("G3", axis=1)

            st.markdown("""
            Se han separado los datos en:
            - X: Variables independientes (características)
            - y: Variable dependiente (G3 - nota final)
            
            Las características se han convertido a un array numpy de tipo float64 para
            asegurar la compatibilidad con los algoritmos de machine learning.
            """)

            # Mostrar las dimensiones de X e y
            st.write(f"Dimensiones de X (características): {X.shape}")
            st.write(f"Dimensiones de y (variable objetivo): {y.shape}")

            # Vista previa de los datos separados
            st.subheader("Vista previa del dataset final")

            # Crear DataFrame temporal para visualización
            df_preview = pd.DataFrame(X, columns=df_encoded.drop("G3", axis=1).columns)
            df_preview["G3 (target)"] = y

            st.write("\nPrimeros registros del dataset procesado:")
            st.dataframe(df_preview.head(50))

            # 6. División en conjuntos de entrenamiento y prueba
            st.subheader("6. División train-test")

            # Realizar la división 80-20
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42
            )

            st.markdown("""
            Se han dividido los datos en:
            - Conjunto de entrenamiento (80%)
            - Conjunto de prueba (20%)
            
            Se utilizó random_state=42 para asegurar reproducibilidad.
            """)

            st.markdown(f"""
            **Conjunto de entrenamiento (80%):**
            - X_train shape: {X_train.shape}
            - y_train shape: {y_train.shape}
            
            **Conjunto de prueba (20%):**
            - X_test shape: {X_test.shape}
            - y_test shape: {y_test.shape}
            """)

        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")
