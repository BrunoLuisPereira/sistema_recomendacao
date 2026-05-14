"""
Camada I — Percepção e Sentimento (PLN + Naive Bayes)

Pipeline:
  1. Carrega reviews do Olist e rotula: 1-2 = negativo, 3 = neutro, 4-5 = positivo
  2. Pré-processa o texto (tokenização, stop words PT-BR, stemming RSLP)
  3. Vetoriza com TF-IDF
  4. Treina MultinomialNB
  5. Retorna DataFrame com probabilidade de sentimento positivo por produto
"""

import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Garante que os recursos NLTK estejam disponíveis
for resource in ["punkt", "punkt_tab", "stopwords", "rslp"]:
    try:
        if resource in ["punkt", "punkt_tab"]:
            nltk.data.find(f"tokenizers/{resource}")
        else:
            nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STEMMER = RSLPStemmer()
STOP_PT = set(stopwords.words("portuguese"))


def preprocessar(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", texto)
    tokens = word_tokenize(texto, language="portuguese")
    tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_PT and len(t) > 2]
    return " ".join(tokens)


def rotular(score: int) -> str:
    if score <= 2:
        return "negativo"
    if score == 3:
        return "neutro"
    return "positivo"


def carregar_dados():
    reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")
    items = pd.read_csv("data/olist_order_items_dataset.csv")

    # Junta order_id → product_id
    order_product = items[["order_id", "product_id"]].drop_duplicates("order_id")
    reviews = reviews.merge(order_product, on="order_id", how="inner")

    # Combina título + mensagem como texto final
    reviews["review_text"] = (
        reviews["review_comment_title"].fillna("") + " " + reviews["review_comment_message"].fillna("")
    ).str.strip()

    # Remove reviews sem nenhum texto
    reviews = reviews[reviews["review_text"].str.len() > 3].copy()
    reviews["sentiment"] = reviews["review_score"].apply(rotular)
    return reviews[["product_id", "review_text", "review_score", "sentiment"]]


def treinar_modelo(df: pd.DataFrame, salvar=True):
    print(f"\n[Camada I] Total de reviews com texto: {len(df)}")
    print(df["sentiment"].value_counts())

    print("\n[Camada I] Pré-processando textos...")
    df = df.copy()
    df["texto_proc"] = df["review_text"].apply(preprocessar)

    X = df["texto_proc"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    modelo = MultinomialNB(alpha=0.5)
    modelo.fit(X_train_vec, y_train)

    y_pred = modelo.predict(X_test_vec)
    print("\n[Camada I] Resultado no conjunto de teste:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    classes = modelo.classes_
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)

    if salvar:
        os.makedirs("results", exist_ok=True)
        with open("results/modelo_nb.pkl", "wb") as f:
            pickle.dump({"modelo": modelo, "vectorizer": vectorizer}, f)
        print("\n[Camada I] Modelo salvo em results/modelo_nb.pkl")

    return modelo, vectorizer


def calcular_sentimento_por_produto(df: pd.DataFrame, modelo, vectorizer) -> pd.DataFrame:
    """
    Para cada produto retorna:
      - prob_positivo: média da probabilidade de sentimento positivo
      - n_reviews: quantidade de reviews usadas
    """
    df = df.copy()
    df["texto_proc"] = df["review_text"].apply(preprocessar)
    X_vec = vectorizer.transform(df["texto_proc"])
    probs = modelo.predict_proba(X_vec)

    classes = list(modelo.classes_)
    idx_pos = classes.index("positivo")

    df["prob_positivo"] = probs[:, idx_pos]

    resultado = (
        df.groupby("product_id")
        .agg(
            prob_positivo=("prob_positivo", "mean"),
            n_reviews=("product_id", "count"),
        )
        .reset_index()
    )
    return resultado


def executar() -> pd.DataFrame:
    """Ponto de entrada principal da Camada I."""
    df = carregar_dados()
    modelo, vectorizer = treinar_modelo(df)
    sentimento_produtos = calcular_sentimento_por_produto(df, modelo, vectorizer)
    sentimento_produtos.to_csv("results/sentimento_produtos.csv", index=False)
    print(f"\n[Camada I] Sentimento calculado para {len(sentimento_produtos)} produtos.")
    return sentimento_produtos


if __name__ == "__main__":
    executar()
