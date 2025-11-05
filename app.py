import streamlit as st
from ejercicio_1 import ejercicio_1
from ejercicio_2 import ejercicio_2
from ejercicio_3 import ejercicio_3


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
