"""
N2 — Sistema Inteligente de Recomendação de Produtos (E-commerce PT-BR)
Disciplina: Inteligência Artificial — Prof. Claudinei Dias (Ney)

Pipeline:
  Camada I  → PLN + Naive Bayes   (análise de sentimento)
  Camada II → Sistema Fuzzy       (score de atratividade)
  Camada III→ Algoritmo Genético  (seleção ótima de produtos)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.layer1_nlp   import executar as executar_nlp
from src.layer2_fuzzy import executar as executar_fuzzy
from src.layer3_ga    import executar as executar_ga

os.makedirs("results", exist_ok=True)


def main():
    print("=" * 60)
    print("  SISTEMA INTELIGENTE DE RECOMENDAÇÃO — N2 / IA")
    print("=" * 60)

    t0 = time.time()

    # ── Camada I ───────────────────────────────────────────────────────
    print("\n>>> CAMADA I: PLN + Naive Bayes")
    df_sentimento = executar_nlp()

    # ── Camada II ──────────────────────────────────────────────────────
    print("\n>>> CAMADA II: Sistema de Inferência Fuzzy")
    df_catalogo = executar_fuzzy(df_sentimento)

    # ── Camada III ─────────────────────────────────────────────────────
    print("\n>>> CAMADA III: Algoritmo Genético")
    recomendacao = executar_ga(df_catalogo)

    elapsed = time.time() - t0
    print(f"\nPipeline concluído em {elapsed:.1f}s")
    print("Arquivos gerados em results/:")
    for f in sorted(os.listdir("results")):
        print(f"  • {f}")


if __name__ == "__main__":
    main()
