import streamlit as st
import pandas as pd

from models.sentiment import analisar_sentimento
from models.fuzzy import calcular_interesse
from models.genetic import executar_ga



st.set_page_config(
    page_title="Sistema Inteligente de Recomendação",
    page_icon="🧠",
    layout="centered"
)



st.title("🧠 Sistema Inteligente de Recomendação")

st.write(
    "Sistema utilizando PLN + Fuzzy + Algoritmo Genético"
)



texto = st.text_area(
    "Digite uma avaliação do produto"
)

compras = st.slider(
    "Frequência de compras",
    0,
    100,
    50
)



if st.button("Analisar"):

   

    sentimento, probs = analisar_sentimento(texto)

    st.subheader("📊 Resultado do Naive Bayes")

    if sentimento == "positivo":

        st.success(
            f"Sentimento identificado: {sentimento}"
        )

        valor_sentimento = 90

    elif sentimento == "negativo":

        st.error(
            f"Sentimento identificado: {sentimento}"
        )

        valor_sentimento = 10

    else:

        st.warning(
            f"Sentimento identificado: {sentimento}"
        )

        valor_sentimento = 50

    st.write("Probabilidades:")
    st.write(probs)

   

    interesse = calcular_interesse(
        valor_sentimento,
        compras
    )

    st.subheader("🧠 Resultado Fuzzy")

    st.metric(
        "Nível de Interesse",
        f"{interesse:.2f}"
    )

    if interesse < 30:

        st.error(
            "Cliente com baixo interesse."
        )

    elif interesse < 70:

        st.warning(
            "Cliente com interesse moderado."
        )

    else:

        st.success(
            "Cliente com alto interesse."
        )

    

    recomendados, fitness = executar_ga(
        interesse
    )

    st.subheader("🧬 Algoritmo Genético")

    st.write("Produtos recomendados:")

    for produto in recomendados:

        st.info(produto)

    st.write(
        f"Score otimizado: {fitness}"
    )

   

    novo_registro = pd.DataFrame([{
        "avaliacao": texto,
        "sentimento": sentimento,
        "interesse": round(interesse, 2),
        "produtos": ", ".join(recomendados)
    }])

    novo_registro.to_csv(
        "data/historico.csv",
        mode="a",
        header=False,
        index=False,
        sep=";",
        encoding="utf-8"
    )

    

    st.subheader("📌 Resumo Final")

    st.write(f"""
    • Sentimento detectado: **{sentimento}**

    • Interesse calculado pelo Fuzzy:
    **{interesse:.2f}**

    • Quantidade de produtos recomendados:
    **{len(recomendados)}**
    """)



st.subheader("📜 Histórico de Análises")

try:

    historico = pd.read_csv(
        "data/historico.csv",
        sep=";",
        encoding="utf-8"
    )

    if len(historico.columns) == 4:

        historico.columns = [
            "Avaliação",
            "Sentimento",
            "Interesse",
            "Produtos"
        ]

        st.table(historico)

    else:

        st.error(
            "Erro no histórico CSV. "
            "Limpe o arquivo historico.csv"
        )

except Exception as erro:

    st.error(
        f"Erro ao carregar histórico: {erro}"
    )