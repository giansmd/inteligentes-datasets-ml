import streamlit as st
from ejercicio_1 import ejercicio_1
from ejercicio_2 import ejercicio_2


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
