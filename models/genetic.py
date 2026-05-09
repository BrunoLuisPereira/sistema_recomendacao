import pygad
import numpy as np


nomes_produtos = [
    "Notebook",
    "Mouse",
    "Headset",
    "Teclado"
]


lucros = np.array([100, 40, 60, 80])


custos = np.array([70, 20, 30, 50])

def executar_ga(interesse):

    
    if interesse < 30:
        orcamento = 50

    elif interesse < 70:
        orcamento = 100

    else:
        orcamento = 170

    def fitness_func(ga, solution, solution_idx):

        lucro = np.sum(solution * lucros)

        custo_total = np.sum(solution * custos)

        
        if custo_total > orcamento:
            return 0

        return lucro

    ga_instance = pygad.GA(

        num_generations=50,

        num_parents_mating=4,

        fitness_func=fitness_func,

        sol_per_pop=10,

        num_genes=4,

        gene_space=[0, 1]
    )

    ga_instance.run()

    solucao, fitness, _ = ga_instance.best_solution()

    recomendados = []

    for i in range(len(solucao)):

        if solucao[i] == 1:
            recomendados.append(nomes_produtos[i])

    return recomendados, fitness