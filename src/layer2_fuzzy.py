"""
Camada II — Inferência e Tratamento de Incerteza (Sistema de Inferência Fuzzy Mamdani)

Entradas:
  - sentimento  : probabilidade média de sentimento positivo [0, 1]
  - preco_rel   : preço relativo dentro da categoria [0, 1] (0=barato, 1=caro)

Saída:
  - atratividade: score de atratividade do produto [0, 10]

Variáveis linguísticas e regras Mamdani implementadas com scikit-fuzzy.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

warnings.filterwarnings("ignore")


def construir_sistema_fuzzy():
    # ── Universos de discurso ──────────────────────────────────────────────────
    sentimento = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "sentimento")
    preco_rel  = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "preco_rel")
    atrativ    = ctrl.Consequent(np.arange(0, 10.1,  0.1),  "atratividade")

    # ── Funções de pertinência — Sentimento ───────────────────────────────────
    sentimento["negativo"] = fuzz.trapmf(sentimento.universe, [0,    0,    0.25, 0.45])
    sentimento["neutro"]   = fuzz.trimf( sentimento.universe, [0.35, 0.50, 0.65])
    sentimento["positivo"] = fuzz.trapmf(sentimento.universe, [0.55, 0.75, 1.0,  1.0 ])

    # ── Funções de pertinência — Preço Relativo ───────────────────────────────
    preco_rel["barato"] = fuzz.trapmf(preco_rel.universe, [0,    0,    0.25, 0.45])
    preco_rel["medio"]  = fuzz.trimf( preco_rel.universe, [0.35, 0.50, 0.65])
    preco_rel["caro"]   = fuzz.trapmf(preco_rel.universe, [0.55, 0.75, 1.0,  1.0 ])

    # ── Funções de pertinência — Atratividade ─────────────────────────────────
    atrativ["muito_baixa"] = fuzz.trapmf(atrativ.universe, [0, 0, 1.5, 3.0])
    atrativ["baixa"]       = fuzz.trimf( atrativ.universe, [2.0, 3.5, 5.0])
    atrativ["media"]       = fuzz.trimf( atrativ.universe, [4.0, 5.0, 6.0])
    atrativ["alta"]        = fuzz.trimf( atrativ.universe, [5.0, 6.5, 8.0])
    atrativ["muito_alta"]  = fuzz.trapmf(atrativ.universe, [7.0, 8.5, 10, 10])

    # ── Base de regras ─────────────────────────────────────────────────────────
    regras = [
        # Sentimento positivo
        ctrl.Rule(sentimento["positivo"] & preco_rel["barato"], atrativ["muito_alta"]),
        ctrl.Rule(sentimento["positivo"] & preco_rel["medio"],  atrativ["alta"]),
        ctrl.Rule(sentimento["positivo"] & preco_rel["caro"],   atrativ["media"]),
        # Sentimento neutro
        ctrl.Rule(sentimento["neutro"] & preco_rel["barato"], atrativ["alta"]),
        ctrl.Rule(sentimento["neutro"] & preco_rel["medio"],  atrativ["media"]),
        ctrl.Rule(sentimento["neutro"] & preco_rel["caro"],   atrativ["baixa"]),
        # Sentimento negativo
        ctrl.Rule(sentimento["negativo"] & preco_rel["barato"], atrativ["baixa"]),
        ctrl.Rule(sentimento["negativo"] & preco_rel["medio"],  atrativ["muito_baixa"]),
        ctrl.Rule(sentimento["negativo"] & preco_rel["caro"],   atrativ["muito_baixa"]),
    ]

    sistema = ctrl.ControlSystem(regras)
    return sistema


def normalizar_preco_por_categoria(df_produtos: pd.DataFrame) -> pd.DataFrame:
    """Normaliza o preço dentro de cada categoria (min-max por categoria)."""
    df = df_produtos.copy()

    def minmax(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - mn) / (mx - mn)

    df["preco_rel"] = df.groupby("product_category_name")["avg_price"].transform(minmax)
    df["preco_rel"] = df["preco_rel"].clip(0, 1).fillna(0.5)
    return df


def calcular_atratividade(df_sentimento: pd.DataFrame, df_produtos: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe o DataFrame de sentimento (Camada I) e o catálogo de produtos,
    retorna o catálogo enriquecido com o score de atratividade fuzzy.
    """
    df_prod = normalizar_preco_por_categoria(df_produtos)
    df = df_prod.merge(df_sentimento, on="product_id", how="left")
    df["prob_positivo"] = df["prob_positivo"].fillna(0.5)
    df["n_reviews"] = df["n_reviews"].fillna(0)

    sistema = construir_sistema_fuzzy()
    sim = ctrl.ControlSystemSimulation(sistema)

    scores = []
    for _, row in df.iterrows():
        sim.input["sentimento"] = float(np.clip(row["prob_positivo"], 0.001, 0.999))
        sim.input["preco_rel"]  = float(np.clip(row["preco_rel"],     0.001, 0.999))
        sim.compute()
        scores.append(sim.output["atratividade"])

    df["score_fuzzy"] = scores
    print(f"\n[Camada II] Score fuzzy calculado para {len(df)} produtos.")
    print(f"  Média: {df['score_fuzzy'].mean():.2f}  |  Min: {df['score_fuzzy'].min():.2f}  |  Max: {df['score_fuzzy'].max():.2f}")
    return df


def plotar_funcoes_pertinencia():
    """Gera figura com as funções de pertinência das 3 variáveis."""
    sistema = construir_sistema_fuzzy()
    sim = ctrl.ControlSystemSimulation(sistema)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Sistema Fuzzy — Funções de Pertinência", fontsize=13, fontweight="bold")

    # Sentimento
    ax = axes[0]
    ax.set_title("Sentimento (entrada)")
    x = np.arange(0, 1.01, 0.01)
    ax.plot(x, fuzz.trapmf(x, [0, 0, 0.25, 0.45]),  label="Negativo",  color="tomato")
    ax.plot(x, fuzz.trimf( x, [0.35, 0.50, 0.65]),   label="Neutro",    color="gold")
    ax.plot(x, fuzz.trapmf(x, [0.55, 0.75, 1.0, 1.0]),label="Positivo", color="seagreen")
    ax.set_xlabel("Prob. positivo"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Preço relativo
    ax = axes[1]
    ax.set_title("Preço Relativo (entrada)")
    ax.plot(x, fuzz.trapmf(x, [0, 0, 0.25, 0.45]),    label="Barato", color="seagreen")
    ax.plot(x, fuzz.trimf( x, [0.35, 0.50, 0.65]),     label="Médio",  color="gold")
    ax.plot(x, fuzz.trapmf(x, [0.55, 0.75, 1.0, 1.0]), label="Caro",   color="tomato")
    ax.set_xlabel("Preço normalizado"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Atratividade
    ax = axes[2]
    ax.set_title("Atratividade (saída)")
    y = np.arange(0, 10.1, 0.1)
    ax.plot(y, fuzz.trapmf(y, [0, 0, 1.5, 3.0]),    label="Muito Baixa", color="darkred")
    ax.plot(y, fuzz.trimf( y, [2.0, 3.5, 5.0]),      label="Baixa",       color="tomato")
    ax.plot(y, fuzz.trimf( y, [4.0, 5.0, 6.0]),      label="Média",       color="gold")
    ax.plot(y, fuzz.trimf( y, [5.0, 6.5, 8.0]),      label="Alta",        color="limegreen")
    ax.plot(y, fuzz.trapmf(y, [7.0, 8.5, 10, 10]),   label="Muito Alta",  color="darkgreen")
    ax.set_xlabel("Score (0–10)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/fuzzy_pertinencia.png", dpi=150)
    plt.close()
    print("[Camada II] Gráfico salvo em results/fuzzy_pertinencia.png")


def executar(df_sentimento: pd.DataFrame) -> pd.DataFrame:
    """Ponto de entrada principal da Camada II."""
    products   = pd.read_csv("data/olist_products_dataset.csv")
    items      = pd.read_csv("data/olist_order_items_dataset.csv")
    translation = pd.read_csv("data/product_category_name_translation.csv")

    price_by_product = items.groupby("product_id")["price"].mean().reset_index()
    price_by_product.columns = ["product_id", "avg_price"]

    df_produtos = products.merge(price_by_product, on="product_id", how="left")
    df_produtos = df_produtos.merge(translation, on="product_category_name", how="left")
    df_produtos = df_produtos.dropna(subset=["avg_price"])

    df_catalogo = calcular_atratividade(df_sentimento, df_produtos)
    plotar_funcoes_pertinencia()
    df_catalogo.to_csv("results/catalogo_fuzzy.csv", index=False)
    return df_catalogo


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.layer1_nlp import executar as executar_nlp
    df_sent = executar_nlp()
    executar(df_sent)
