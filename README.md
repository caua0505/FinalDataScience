# FinalDataScience
Entrega final da última avaliação de Data Science: classificador para a base
[Breast Cancer (UCI)](https://archive.ics.uci.edu/dataset/14/breast+cancer), que indica se uma
paciente tem indicativo de recorrência do câncer (atributo `Class`).

## Arquivos

| Arquivo | Função |
|---|---|
| `breast-cancer.csv` | Base de dados (286 pacientes, 9 atributos + `Class`) |
| `preprocessamento.py` | Tratamento de ausentes e codificação, usados pelo treino **e** pela inferência |
| `treinamento.py` | **Módulo de treinamento**: compara 8 classificadores e salva o de melhor acurácia |
| `inferencia.py` | **Módulo de inferência**: recebe uma nova paciente e devolve o diagnóstico |
| `modelos/` | Gerada pelo treino: normalizador, classificador, metadados e `ranking_modelos.csv` |

## Como executar

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python treinamento.py
python inferencia.py              # paciente de exemplo
python inferencia.py --interativo # digitar os dados da paciente
```

Também é possível usar a função diretamente:

```python
from inferencia import diagnosticar
diagnosticar({'age': '50-59', 'menopause': 'ge40', 'tumor-size': '15-19', 'inv-nodes': '0-2',
              'node-caps': 'no', 'deg-malig': '1', 'breast': 'right',
              'breast-quad': 'central', 'irradiat': 'no'})
# -> 'no-recurrence-events'
```

## Etapas do treinamento

1. **Carregar os dados** (`breast-cancer.csv`).
2. **Tratar ausentes**: os `?` (8 em `node-caps`, 1 em `breast-quad`) são trocados pela moda da coluna.
3. **Codificar os atributos**
   - faixas com ordem (`age`, `tumor-size`, `inv-nodes`) viram número (limite inferior da faixa, ex.: `50-59` → 50);
   - `deg-malig` já é numérico;
   - categóricos (`menopause`, `node-caps`, `breast`, `breast-quad`, `irradiat`) → `pd.get_dummies`.
4. **Separar treino (70%) e teste (30%)**, estratificado pela classe.
5. **Balancear com SMOTE**: a base é desbalanceada (201 sem recorrência × 85 com recorrência).
6. **Normalizar com MinMaxScaler** (somente os atributos).
7. **Hiperparametrização com RandomizedSearchCV** (30 combinações, validação cruzada de 5 dobras)
   em 8 algoritmos: Decision Tree, Random Forest, Gradient Boosting, KNN, SVM, Regressão Logística,
   MLP e Naive Bayes. SMOTE + MinMax + classificador ficam num `Pipeline` do imblearn, então o
   balanceamento é refeito dentro de cada dobra e os registros sintéticos não vazam para a validação.
8. **Avaliação no teste** e escolha do modelo de **melhor acurácia**.
9. **Treinamento definitivo** do melhor modelo com toda a base e gravação com `pickle`.

## Resultados

| Modelo | Acurácia no teste | Acurácia CV |
|---|---|---|
| **Decision Tree** | **70,93%** | 74,50% |
| SVM | 69,77% | 74,50% |
| KNN | 69,77% | 73,00% |
| Regressão Logística | 67,44% | 74,50% |
| Gradient Boosting | 66,28% | 72,00% |
| Naive Bayes | 65,12% | 73,00% |
| Random Forest | 65,12% | 72,50% |
| MLP | 65,12% | 72,00% |

Melhor modelo: **Decision Tree** (`criterion='entropy'`, `splitter='random'`, `max_depth=3`,
`min_samples_split=18`, `min_samples_leaf=2`).

Observação: essa base é reconhecidamente difícil (a literatura reporta cerca de 70–75%). Prever sempre
"sem recorrência" já acertaria 69,8% do teste, e o modelo identifica poucos casos de recorrência
(recall de 12% nessa classe), então o resultado deve ser lido como indicativo, não como diagnóstico.
