import streamlit as st
import pandas as pd
from pathlib import Path
import io

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
        except Exception as e:
            st.exception(e)
            return

    st.subheader("Vista previa del dataset")
    st.dataframe(df.head(100))

    st.markdown("**Dimensiones:**")
    st.write(f"Filas: {df.shape[0]} — Columnas: {df.shape[1]}")

    st.markdown("**Resumen de columnas y tipos**")
    st.write(df.dtypes)

    st.markdown("**Información (df.info())**")
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    st.text(s)

    st.markdown("**Estadísticas descriptivas (solo numéricas)**")
    st.write(df.describe())

    if st.checkbox("Mostrar dataset completo (puede ser lento)"):
        st.dataframe(df)

    st.markdown("---")
    st.markdown("Puedes descargar una muestra de 100 filas:")
    csv_sample = df.head(100).to_csv(index=False).encode('utf-8')
    st.download_button("Descargar muestra CSV (100 filas)", data=csv_sample, file_name="titanic_sample.csv", mime='text/csv')


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
