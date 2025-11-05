import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def ejercicio_3():
    st.header("Ejercicio 3 — Iris Dataset Classification")

    # Cargar el dataset
    iris = load_iris()

    dataframe_inicial = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    dataframe_inicial["target"] = iris.target

    st.write("### Dataset Iris original")
    st.dataframe(dataframe_inicial.head(150))

    X = iris.data
    y = iris.target

    #añadir el nombre de las columnas
    dataframe_x = pd.DataFrame(data=X, columns=iris.feature_names)
    dataframe_y = pd.DataFrame(data=y, columns=["target"])

    # usamos standard scaler para escalar las características
    scaler = StandardScaler()
    dataframe_x_scaled = scaler.fit_transform(dataframe_x)

    # convertimos dataframe_x_scaled a dataframe de pandas
    dataframe_x_scaled = pd.DataFrame(data=dataframe_x_scaled, columns=iris.feature_names)

    # juntamos ambos
    dataframe = pd.concat([dataframe_x_scaled, dataframe_y], axis=1)

    st.write("### Dataset Iris con características escaladas")
    st.dataframe(dataframe.head(150))

    # dividimos en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe_x_scaled, dataframe_y, test_size=0.3, random_state=0
    )

    # mostramos los shapes
    st.write(f"Dimensiones de X_train: {X_train.shape}")
    st.write(f"Dimensiones de X_test: {X_test.shape}")

    # mostramos un gráfico de dispersión para sepal length vs petal length
    # se usa el color para diferenciar las clases
    st.write("### Gráfico de dispersión: Sepal Length vs Petal Length")
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        dataframe_x_scaled["sepal length (cm)"],
        dataframe_x_scaled["petal length (cm)"],
        c=dataframe_y["target"],
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    st.pyplot(plt)

    # estadisticas descriptivas del dataset estandarizado
    st.write("### Estadísticas descriptivas del dataset estandarizado")
    st.dataframe(dataframe.describe())
