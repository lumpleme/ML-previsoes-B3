import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import random
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

JANELA_DIAS = 21
BASE = 'dados/petr4_multivar.csv'

class ModeloLSTMClassific(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=3):
        super(ModeloLSTMClassific, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========================================================= #

# carregando base de dados
df = pd.read_csv(BASE)
df['date'] = pd.to_datetime(df['date'])
df = df.dropna().reset_index(drop=True)

# calculando a flutuação normal com desvio padrão
desvio_padrao = df['return'].std()
print(f"\nDesvio Padrão Histórico do Ativo: {desvio_padrao:.4f}")

# definindo os limites
limite_neutro = 0.5 * desvio_padrao

# criar 3 categorias com base nos limites
def categorizar_retorno(retorno):
    if retorno < -limite_neutro:
        return 0  # Desce
    elif -limite_neutro <= retorno <= limite_neutro:
        return 1  # Neutro
    else:
        return 2  # Sobe

df['classe_alvo'] = df['return'].apply(categorizar_retorno)

print("\nDistribuição das classes na base de dados:")
print(df['classe_alvo'].value_counts().sort_index())

# ========================================================= #

# montando os conjuntos para treinamento
dados_x = df[['open', 'high', 'low', 'closed', 'vol']].values
dados_y = df['classe_alvo'].values

tamanho_treino = int(len(dados_y) * 0.8)
treino_x_bruto, teste_x_bruto = dados_x[:tamanho_treino], dados_x[tamanho_treino:]
treino_y_bruto, teste_y_bruto = dados_y[:tamanho_treino], dados_y[tamanho_treino:]

# normalizando o x
scaler_x = MinMaxScaler(feature_range=(0, 1))
treino_x_norm = scaler_x.fit_transform(treino_x_bruto)
teste_x_norm = scaler_x.transform(teste_x_bruto)

# criando as sequências com a janela definida
def criar_sequencias_classificacao(dados_x, dados_y, janela):
    X, y = [], []
    for i in range(len(dados_x) - janela):
        X.append(dados_x[i : i + janela])
        y.append(dados_y[i + janela])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

x_treino, y_treino = criar_sequencias_classificacao(treino_x_norm, treino_y_bruto, JANELA_DIAS)
x_teste, y_teste = criar_sequencias_classificacao(teste_x_norm, teste_y_bruto, JANELA_DIAS)

# ========================================================= #

# calculando pesos para balancear as classes

# calculando um peso para cada classe para equilibrar o treinamento
pesos_classes = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(treino_y_bruto),
    y=treino_y_bruto
)

# convertendo para um tensor do PyTorch
pesos_tensor = torch.tensor(pesos_classes, dtype=torch.float32)


print(f"Pesos calculados: {pesos_classes}")

# ========================================================= #

# configurando o modelo LSTM classificador
print("\nIniciando GridSearch do LSTM Classificador...")

net = NeuralNetClassifier(
    module=ModeloLSTMClassific,
    criterion=nn.CrossEntropyLoss,
    criterion__weight=pesos_tensor,
    optimizer=optim.Adam,
    batch_size=16,
    verbose=0,
    device='cpu'
)

# definindo os hiperparâmetros para o gridsearch
grid_params = {
    'module__hidden_size': [20, 50],
    'module__num_layers': [2, 3],
    'optimizer__lr': [0.001],
    'max_epochs': [20, 50, 100]
}

tscv = TimeSeriesSplit(n_splits=3)

# realizando o gridsearch para testar as combinações de hiperparâmetros
gs_lstm = GridSearchCV(net, grid_params, refit=True, cv=tscv, scoring='accuracy', verbose=2)
gs_lstm.fit(x_treino, y_treino)

print("\nMelhores hiperparâmetros encontrados:", gs_lstm.best_params_)
print("Melhor Acurácia no Treino:", gs_lstm.best_score_)

# ========================================================= #

# calculando as previsões no conjunto de teste
y_teste_previsto = gs_lstm.predict(x_teste)

print("RELATÓRIO DE CLASSIFICAÇÃO LSTM (TESTE):")
print(classification_report(y_teste, y_teste_previsto, 
                            target_names=['0: Desce', '1: Neutro', '2: Sobe']))

# ========================================================= #

# calculando a matriz de confusão
matriz_confusao = confusion_matrix(y_teste, y_teste_previsto)

# rótulos para o gráfico
nomes_classes = ['Desce', 'Neutro', 'Sobe']

# desenha o heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nomes_classes, yticklabels=nomes_classes)

plt.title('Matriz de confusão (3 classes): Previsão LSTM vs Realidade')
plt.xlabel('Previsão do modelo')
plt.ylabel('Realidade')
plt.tight_layout()
plt.show()