import numpy as np
import skfuzzy as fuzz

from skfuzzy import control as ctrl



sentimento = ctrl.Antecedent(
    np.arange(0, 101, 1),
    'sentimento'
)

compras = ctrl.Antecedent(
    np.arange(0, 101, 1),
    'compras'
)

interesse = ctrl.Consequent(
    np.arange(0, 101, 1),
    'interesse'
)


sentimento['baixo'] = fuzz.trapmf(
    sentimento.universe,
    [0, 0, 20, 40]
)

sentimento['medio'] = fuzz.trimf(
    sentimento.universe,
    [30, 50, 70]
)

sentimento['alto'] = fuzz.trapmf(
    sentimento.universe,
    [60, 80, 100, 100]
)


compras['baixa'] = fuzz.trapmf(
    compras.universe,
    [0, 0, 20, 40]
)

compras['media'] = fuzz.trimf(
    compras.universe,
    [30, 50, 70]
)

compras['alta'] = fuzz.trapmf(
    compras.universe,
    [60, 80, 100, 100]
)


interesse['baixo'] = fuzz.trapmf(
    interesse.universe,
    [0, 0, 20, 40]
)

interesse['medio'] = fuzz.trimf(
    interesse.universe,
    [30, 50, 70]
)

interesse['alto'] = fuzz.trapmf(
    interesse.universe,
    [60, 80, 100, 100]
)



regras = [

    ctrl.Rule(
        sentimento['baixo'] &
        compras['baixa'],
        interesse['baixo']
    ),

    ctrl.Rule(
        sentimento['baixo'] &
        compras['media'],
        interesse['baixo']
    ),

    ctrl.Rule(
        sentimento['baixo'] &
        compras['alta'],
        interesse['medio']
    ),

    ctrl.Rule(
        sentimento['medio'] &
        compras['baixa'],
        interesse['baixo']
    ),

    ctrl.Rule(
        sentimento['medio'] &
        compras['media'],
        interesse['medio']
    ),

    ctrl.Rule(
        sentimento['medio'] &
        compras['alta'],
        interesse['alto']
    ),

    ctrl.Rule(
        sentimento['alto'] &
        compras['baixa'],
        interesse['medio']
    ),

    ctrl.Rule(
        sentimento['alto'] &
        compras['media'],
        interesse['alto']
    ),

    ctrl.Rule(
        sentimento['alto'] &
        compras['alta'],
        interesse['alto']
    )
]



sistema_controle = ctrl.ControlSystem(
    regras
)



def calcular_interesse(
    valor_sentimento,
    valor_compras
):

    simulador = ctrl.ControlSystemSimulation(
        sistema_controle
    )

    simulador.input['sentimento'] = valor_sentimento

    simulador.input['compras'] = valor_compras

    simulador.compute()

    if 'interesse' not in simulador.output:

        return 50

    return simulador.output['interesse']