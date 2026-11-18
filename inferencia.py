#MÓDULO DE INFERÊNCIA - Breast Cancer
#Recebe uma nova paciente e determina o indicativo de diagnóstico (atributo Class):
#   no-recurrence-events  -> sem recorrência
#   recurrence-events     -> com recorrência
#
#Uso:
#   python inferencia.py                      (classifica a paciente de exemplo abaixo)
#   python inferencia.py --interativo         (digita os dados da paciente no terminal)

import os
import sys
import pandas as pd
from pickle import load

from preprocessamento import ATRIBUTOS, tratar_ausentes, codificar

sys.stdout.reconfigure(encoding='utf-8')

PASTA_MODELOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modelos')

#Valores aceitos em cada atributo (descrição da base na UCI)
VALORES_VALIDOS = {
    'age': ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    'menopause': ['lt40', 'ge40', 'premeno'],
    'tumor-size': ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
                   '40-44', '45-49', '50-54', '55-59'],
    'inv-nodes': ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23',
                  '24-26', '27-29', '30-32', '33-35', '36-39'],
    'node-caps': ['yes', 'no'],
    'deg-malig': ['1', '2', '3'],
    'breast': ['left', 'right'],
    'breast-quad': ['left_up', 'left_low', 'right_up', 'right_low', 'central'],
    'irradiat': ['yes', 'no'],
}

#Carregar o que foi salvo pelo treinamento
normalizador = load(open(os.path.join(PASTA_MODELOS, 'normalizador_breast_cancer.model'), 'rb'))
classificador = load(open(os.path.join(PASTA_MODELOS, 'classificador_breast_cancer.model'), 'rb'))
metadados = load(open(os.path.join(PASTA_MODELOS, 'metadados_breast_cancer.model'), 'rb'))


def diagnosticar(paciente):
    """paciente: dict com os 9 atributos no formato original da base.
    Valores desconhecidos podem ser informados como '?'. Retorna a classe prevista."""
    for atributo in ATRIBUTOS:
        valor = str(paciente.get(atributo, '?'))
        if valor != '?' and valor not in VALORES_VALIDOS[atributo]:
            raise ValueError(f"Valor inválido para '{atributo}': {valor}. "
                             f"Aceitos: {VALORES_VALIDOS[atributo]}")

    #mesmas transformações do treinamento
    nova = pd.DataFrame([{a: str(paciente.get(a, '?')) for a in ATRIBUTOS}])
    nova = tratar_ausentes(nova, metadados['modas'])
    nova = codificar(nova, colunas_finais=metadados['colunas'])
    nova = normalizador.transform(nova)

    return classificador.predict(nova)[0]


def ler_paciente_terminal():
    paciente = {}
    for atributo in ATRIBUTOS:
        opcoes = ', '.join(VALORES_VALIDOS[atributo])
        while True:
            valor = input(f'{atributo} ({opcoes}) ou ? se desconhecido: ').strip()
            if valor == '?' or valor in VALORES_VALIDOS[atributo]:
                paciente[atributo] = valor
                break
            print('  valor inválido, tente novamente.')
    return paciente


if __name__ == '__main__':
    if '--interativo' in sys.argv:
        nova_paciente = ler_paciente_terminal()
    else:
        #Nova paciente de exemplo
        nova_paciente = {
            'age': '40-49',
            'menopause': 'premeno',
            'tumor-size': '30-34',
            'inv-nodes': '6-8',
            'node-caps': 'yes',
            'deg-malig': '3',
            'breast': 'left',
            'breast-quad': 'left_low',
            'irradiat': 'yes',
        }

    print('Modelo utilizado:', metadados['nome_modelo'],
          f"(acurácia no teste: {metadados['acuracia_teste']:.2%})")
    print('Paciente:', nova_paciente)
    diagnostico = diagnosticar(nova_paciente)
    descricao = 'COM indicativo de recorrência' if diagnostico == 'recurrence-events' \
        else 'SEM indicativo de recorrência'
    print('Indicativo de diagnóstico (Class):', diagnostico, '->', descricao)
