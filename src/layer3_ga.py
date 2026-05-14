"""
Camada III — Otimização Estocástica (Algoritmo Genético com DEAP)

Problema: dado um catálogo de produtos com score de atratividade fuzzy,
selecionar a melhor combinação de N_RECOMENDACOES produtos para recomendar
a um usuário, respeitando um orçamento máximo e maximizando diversidade de
categorias.

Representação:
  - Indivíduo : lista de N_RECOMENDACOES índices (posições no catálogo)
  - Fitness   : score_fuzzy_total + bonus_diversidade − penalidade_orcamento
"""

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

# ── Constantes ─────────────────────────────────────────────────────────────────
N_RECOMENDACOES = 5
TAMANHO_POP     = 100
N_GERACOES      = 80
PROB_CROSSOVER  = 0.8
PROB_MUTACAO    = 0.2
ORCAMENTO_MAX   = 1_000.0   # R$
BONUS_DIVERSIDADE = 1.5     # ponto por categoria distinta além de 1


# ── Fitness ─────────────────────────────────────────────────────────────────────
def avaliar(individuo, catalogo: pd.DataFrame) -> tuple:
    rows = catalogo.iloc[individuo]
    score_total = rows["score_fuzzy"].sum()

    n_categorias = rows["product_category_name"].nunique()
    bonus = BONUS_DIVERSIDADE * (n_categorias - 1)

    custo_total = rows["avg_price"].sum()
    penalidade = max(0, custo_total - ORCAMENTO_MAX) * 0.05

    fitness = score_total + bonus - penalidade
    return (fitness,)


# ── Operadores genéticos ────────────────────────────────────────────────────────
def criar_individuo(n_produtos: int) -> list:
    return random.sample(range(n_produtos), N_RECOMENDACOES)


def mutar(individuo, n_produtos: int) -> tuple:
    idx = random.randint(0, N_RECOMENDACOES - 1)
    candidatos = list(set(range(n_produtos)) - set(individuo))
    if candidatos:
        individuo[idx] = random.choice(candidatos)
    return (individuo,)


def crossover_ordenado(ind1, ind2) -> tuple:
    """Crossover de dois pontos preservando unicidade."""
    size = len(ind1)
    pt1, pt2 = sorted(random.sample(range(size), 2))
    # segmento central de ind2 que ainda não está em ind1[pt1:pt2]
    segmento = [g for g in ind2 if g not in ind1[pt1:pt2]]
    novo1 = segmento[:pt1] + ind1[pt1:pt2] + segmento[pt1:]
    segmento = [g for g in ind1 if g not in ind2[pt1:pt2]]
    novo2 = segmento[:pt1] + ind2[pt1:pt2] + segmento[pt1:]
    ind1[:] = novo1[:size]
    ind2[:] = novo2[:size]
    return ind1, ind2


def configurar_deap(n_produtos: int, catalogo: pd.DataFrame):
    # Limpa definições anteriores (evita erro ao rodar múltiplas vezes)
    for nome in ["FitnessMax", "Individual"]:
        if nome in creator.__dict__:
            delattr(creator, nome)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individuo",  tools.initIterate, creator.Individual,
                     lambda: criar_individuo(n_produtos))
    toolbox.register("population", tools.initRepeat, list, toolbox.individuo)
    toolbox.register("evaluate",   avaliar, catalogo=catalogo)
    toolbox.register("mate",       crossover_ordenado)
    toolbox.register("mutate",     mutar, n_produtos=n_produtos)
    toolbox.register("select",     tools.selTournament, tournsize=4)
    return toolbox


def executar_ga(catalogo: pd.DataFrame) -> tuple:
    """Executa o GA e retorna (melhor_individuo, log_geracoes)."""
    random.seed(42)
    n_produtos = len(catalogo)

    toolbox = configurar_deap(n_produtos, catalogo)

    populacao = toolbox.population(n=TAMANHO_POP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max",  np.max)
    stats.register("mean", np.mean)
    stats.register("min",  np.min)

    print(f"\n[Camada III] Iniciando GA: {N_GERACOES} gerações, pop={TAMANHO_POP}")

    populacao, log = algorithms.eaSimple(
        populacao, toolbox,
        cxpb=PROB_CROSSOVER,
        mutpb=PROB_MUTACAO,
        ngen=N_GERACOES,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    melhor = hof[0]
    return melhor, log


def exibir_resultado(melhor_individuo, catalogo: pd.DataFrame):
    recomendados = catalogo.iloc[melhor_individuo].copy()
    print("\n" + "=" * 60)
    print("  RECOMENDAÇÃO FINAL — Algoritmo Genético")
    print("=" * 60)
    colunas = ["product_id", "product_category_name", "avg_price", "prob_positivo", "score_fuzzy"]
    colunas_disp = [c for c in colunas if c in recomendados.columns]
    print(recomendados[colunas_disp].to_string(index=False))
    print("-" * 60)
    print(f"  Score total   : {recomendados['score_fuzzy'].sum():.2f}")
    print(f"  Custo total   : R$ {recomendados['avg_price'].sum():.2f}")
    print(f"  Categorias    : {recomendados['product_category_name'].nunique()}")
    fitness = avaliar(melhor_individuo, catalogo)[0]
    print(f"  Fitness final : {fitness:.2f}")
    print("=" * 60)
    recomendados.to_csv("results/recomendacao_final.csv", index=False)


def plotar_evolucao(log):
    gen   = [r["gen"]  for r in log]
    maxi  = [r["max"]  for r in log]
    media = [r["mean"] for r in log]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gen, maxi,  label="Melhor fitness",  color="royalblue",   linewidth=2)
    ax.plot(gen, media, label="Fitness médio",   color="darkorange",  linewidth=1.5, linestyle="--")
    ax.set_title("Algoritmo Genético — Evolução do Fitness por Geração", fontweight="bold")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ga_evolucao.png", dpi=150)
    plt.close()
    print("[Camada III] Gráfico salvo em results/ga_evolucao.png")


def executar(df_catalogo: pd.DataFrame) -> pd.DataFrame:
    """Ponto de entrada principal da Camada III."""
    # Usa apenas produtos com reviews suficientes para ter score confiável
    catalogo = df_catalogo[df_catalogo["n_reviews"] >= 1].reset_index(drop=True)
    print(f"[Camada III] Catálogo com {len(catalogo)} produtos elegíveis.")

    melhor, log = executar_ga(catalogo)
    exibir_resultado(melhor, catalogo)
    plotar_evolucao(log)
    return catalogo.iloc[melhor].copy()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.layer1_nlp import executar as executar_nlp
    from src.layer2_fuzzy import executar as executar_fuzzy
    df_sent    = executar_nlp()
    df_catalogo = executar_fuzzy(df_sent)
    executar(df_catalogo)
