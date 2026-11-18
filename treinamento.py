#MÓDULO DE TREINAMENTO - Breast Cancer (UCI)
#https://archive.ics.uci.edu/dataset/14/breast+cancer
#
#Etapas (mesmo roteiro das aulas):
# 1 Carregar os dados
# 2 Tratar dados ausentes ('?')
# 3 Separar atributos e classe / codificar categóricos (get_dummies)
# 4 Separar treino e teste
# 5 Balancear os dados de treino (SMOTE)
# 6 Normalizar os atributos (MinMaxScaler)
# 7 Hiperparametrização (RandomizedSearchCV) de vários classificadores
#   -> SMOTE + MinMax + classificador ficam num Pipeline do imblearn, para que o
#      balanceamento seja refeito DENTRO de cada dobra da validação cruzada.
#      Se o SMOTE fosse aplicado antes do CV, os registros sintéticos "vazariam" para
#      as dobras de validação e a acurácia ficaria artificialmente alta.
# 8 Avaliar a acurácia de cada um no teste e escolher o de MELHOR ACURÁCIA
# 9 Treinamento definitivo do melhor modelo com todos os dados e salvar (pickle)

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pickle import dump
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB

from preprocessamento import ATRIBUTOS, CLASSE, tratar_ausentes, codificar

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')  #avisos de convergência/combinações inválidas da busca

SEMENTE = 42
PASTA_MODELOS = 'modelos'
os.makedirs(PASTA_MODELOS, exist_ok=True)

#1 CARREGAR OS DADOS
dados = pd.read_csv('breast-cancer.csv', sep=',', dtype=str)
print('Registros:', dados.shape[0])
print('Distribuição das classes:', Counter(dados[CLASSE]))

#2 TRATAR DADOS AUSENTES
#A base usa '?' para valores ausentes (node-caps: 8, breast-quad: 1). Substituímos pela moda.
modas = {c: dados.loc[dados[c] != '?', c].mode()[0] for c in ATRIBUTOS}
dados = tratar_ausentes(dados, modas)

#3 SEPARAR ATRIBUTOS E CLASSE E CODIFICAR OS CATEGÓRICOS
dados_classes = dados[CLASSE]
dados_atributos = codificar(dados)
colunas_finais = list(dados_atributos.columns)

#4 SEPARAR TREINO (70%) E TESTE (30%)
#O teste não é balanceado nem usado no treino: serve para medir a acurácia real.
atr_treino, atr_teste, cls_treino, cls_teste = train_test_split(
    dados_atributos, dados_classes, test_size=0.3, stratify=dados_classes, random_state=SEMENTE)

#5 e 6 BALANCEAR (SMOTE) E NORMALIZAR (MinMax)
#Só para visualizar o efeito do balanceamento no treino:
print('Treino antes do SMOTE:', Counter(cls_treino))
print('Treino após SMOTE:   ', Counter(SMOTE(random_state=SEMENTE).fit_resample(atr_treino, cls_treino)[1]))


def montar_pipeline(classificador):
    """Balanceador -> normalizador -> classificador.
    O SMOTE só atua no fit (treino); no predict os dados passam apenas pelo MinMax."""
    return Pipeline([
        ('balanceador', SMOTE(random_state=SEMENTE)),
        ('normalizador', MinMaxScaler()),
        ('classificador', classificador),
    ])


#7 HIPERPARAMETRIZAÇÃO DE VÁRIOS CLASSIFICADORES
candidatos = {
    'DecisionTree': (DecisionTreeClassifier(random_state=SEMENTE), {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 3, 4, 5, 6, 8, 10],
        'min_samples_split': [int(x) for x in np.linspace(start=2, stop=20, num=10)],
        'min_samples_leaf': [1, 2, 4, 8],
    }),
    'RandomForest': (RandomForestClassifier(random_state=SEMENTE), {
        'n_estimators': [int(x) for x in np.linspace(start=50, stop=400, num=8)],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 8, 12],
        'min_samples_split': [2, 4, 8, 12],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }),
    'GradientBoosting': (GradientBoostingClassifier(random_state=SEMENTE), {
        'n_estimators': [50, 100, 150, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [2, 3, 4, 5],
        'subsample': [0.6, 0.8, 1.0],
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': list(range(1, 31, 2)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'chebyshev'],
    }),
    'SVM': (SVC(random_state=SEMENTE), {
        'C': [0.01, 0.1, 0.5, 1, 2, 5, 10, 50],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    }),
    'LogisticRegression': (LogisticRegression(max_iter=2000, random_state=SEMENTE), {
        'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
    }),
    'MLP': (MLPClassifier(max_iter=2000, random_state=SEMENTE), {
        'hidden_layer_sizes': [(10,), (20,), (50,), (20, 10), (50, 20)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        'learning_rate_init': [0.001, 0.01],
    }),
    'NaiveBayes': (BernoulliNB(), {
        'alpha': [0.01, 0.1, 0.5, 1, 2, 5],
        'binarize': [0.0, 0.25, 0.5],
    }),
}

validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
resultados = []
for nome, (classificador, grade) in candidatos.items():
    busca = RandomizedSearchCV(
        estimator=montar_pipeline(classificador),
        param_distributions={'classificador__' + p: v for p, v in grade.items()},
        n_iter=30,
        cv=validacao,
        scoring='accuracy',
        random_state=SEMENTE,
        n_jobs=-1,
    )
    busca.fit(atr_treino, cls_treino)

    #8 AVALIAR A ACURÁCIA NO TESTE (dados nunca vistos)
    classes_preditas = busca.best_estimator_.predict(atr_teste)
    acuracia = accuracy_score(cls_teste, classes_preditas)
    resultados.append({
        'modelo': nome,
        'acuracia_teste': acuracia,
        'acuracia_cv': busca.best_score_,
        'parametros': {p.replace('classificador__', ''): v for p, v in busca.best_params_.items()},
        'pipeline': busca.best_estimator_,
    })
    print(f'{nome:<20} acurácia teste = {acuracia:.4f} | acurácia CV = {busca.best_score_:.4f}')

#Ranking: melhor acurácia no teste (empate -> melhor acurácia na validação cruzada)
resultados.sort(key=lambda r: (r['acuracia_teste'], r['acuracia_cv']), reverse=True)
ranking = pd.DataFrame(resultados)[['modelo', 'acuracia_teste', 'acuracia_cv', 'parametros']]
ranking.to_csv(os.path.join(PASTA_MODELOS, 'ranking_modelos.csv'), index=False)

melhor = resultados[0]
print('\n=== MELHOR MODELO:', melhor['modelo'], '===')
print('Acurácia no teste:', round(melhor['acuracia_teste'], 4))
print('Hiperparâmetros:', melhor['parametros'])
preditas = melhor['pipeline'].predict(atr_teste)
print('Matriz de confusão (linhas = real):')
rotulos = sorted(dados_classes.unique())
print(pd.DataFrame(confusion_matrix(cls_teste, preditas, labels=rotulos), index=rotulos, columns=rotulos))
print(classification_report(cls_teste, preditas, zero_division=0))

#9 TREINAMENTO DEFINITIVO COM TODOS OS DADOS
#Com a arquitetura escolhida, repetimos balanceamento + normalização na base inteira
#e treinamos o modelo final que será usado pela inferência.
classificador_final = candidatos[melhor['modelo']][0].set_params(**melhor['parametros'])
pipeline_final = montar_pipeline(classificador_final).fit(dados_atributos, dados_classes)
normalizador_final = pipeline_final.named_steps['normalizador']
modelo_final = pipeline_final.named_steps['classificador']

#Salvar tudo o que a inferência precisa
dump(normalizador_final, open(os.path.join(PASTA_MODELOS, 'normalizador_breast_cancer.model'), 'wb'))
dump(modelo_final, open(os.path.join(PASTA_MODELOS, 'classificador_breast_cancer.model'), 'wb'))
dump({'colunas': colunas_finais, 'modas': modas, 'nome_modelo': melhor['modelo'],
      'acuracia_teste': melhor['acuracia_teste'], 'parametros': melhor['parametros']},
     open(os.path.join(PASTA_MODELOS, 'metadados_breast_cancer.model'), 'wb'))
print('Modelos salvos na pasta', PASTA_MODELOS)
