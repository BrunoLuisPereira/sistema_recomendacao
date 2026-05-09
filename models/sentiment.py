import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


df = pd.read_csv("dataset/reviews.csv")


modelo = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)


modelo.fit(df["texto"], df["classe"])

def analisar_sentimento(texto):
    sentimento = modelo.predict([texto])[0]

    probabilidades = modelo.predict_proba([texto])[0]

    classes = modelo.classes_

    resultado_probs = dict(zip(classes, probabilidades))

    return sentimento, resultado_probs