#Funções de pré-processamento compartilhadas entre o treinamento e a inferência.
#Assim a nova paciente passa EXATAMENTE pelas mesmas transformações dos dados de treino.

import pandas as pd

ATRIBUTOS = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
             'deg-malig', 'breast', 'breast-quad', 'irradiat']
CLASSE = 'Class'

#Atributos que são faixas numéricas ("50-59", "0-2"...). Têm ordem, então viram número
#(o limite inferior da faixa) em vez de dummies.
ATRIBUTOS_FAIXA = ['age', 'tumor-size', 'inv-nodes']
#Atributo já numérico (grau de malignidade 1, 2 ou 3)
ATRIBUTOS_NUMERICOS = ['deg-malig']
#Atributos categóricos (textos sem ordem) -> get_dummies
ATRIBUTOS_CATEGORICOS = ['menopause', 'node-caps', 'breast', 'breast-quad', 'irradiat']


def faixa_para_numero(valor):
    """'50-59' -> 50 ; '0-4' -> 0"""
    return int(str(valor).split('-')[0])


def tratar_ausentes(dados, modas):
    """Substitui os '?' (dados ausentes da base UCI) pela moda da coluna, calculada no treino."""
    dados = dados.replace('?', pd.NA)
    for coluna, moda in modas.items():
        if coluna in dados.columns:
            dados[coluna] = dados[coluna].fillna(moda)
    return dados


def codificar(dados, colunas_finais=None):
    """Converte os atributos brutos em uma tabela 100% numérica.

    colunas_finais: lista de colunas gerada no treino. Na inferência ela é usada para
    garantir que a nova paciente tenha as mesmas colunas (dummies) e na mesma ordem.
    """
    dados = dados[ATRIBUTOS].copy()

    #faixas -> número
    for coluna in ATRIBUTOS_FAIXA:
        dados[coluna] = dados[coluna].apply(faixa_para_numero)
    for coluna in ATRIBUTOS_NUMERICOS:
        dados[coluna] = dados[coluna].astype(int)

    #categóricos -> dummies (mesma técnica da aula: pd.get_dummies)
    dummies = pd.get_dummies(dados[ATRIBUTOS_CATEGORICOS], prefix=ATRIBUTOS_CATEGORICOS,
                             prefix_sep='_', dtype=int)
    dados = pd.concat([dados[ATRIBUTOS_FAIXA + ATRIBUTOS_NUMERICOS], dummies], axis=1)

    if colunas_finais is not None:
        #dummies que não aparecerem na nova paciente viram 0
        dados = dados.reindex(columns=colunas_finais, fill_value=0)
    return dados
