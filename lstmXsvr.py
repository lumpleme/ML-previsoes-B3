import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import random
from sklearn.svm import SVR


class ModeloLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(ModeloLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # camada LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # camada linear (saída)
        # para pegar o que o LSTM processou e transformar em um único valor (a previsão do dia seguinte)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        # o LSTM retorna as previsões de todos os passos (out)
        # e o estado da memória oculta, que será ignorado (hn, cn)
        out, (hn, cn) = self.lstm(x)
        
        # passar a previsão do último dia para a camada linear
        # out[:, -1, :] pega todas as amostras, o último dia, e todas as features
        out = self.linear(out[:, -1, :])
        
        return out
    


# carregando os dados
df = pd.read_csv('dados/Historico_retornos.csv')
df['data'] = pd.to_datetime(df['data'])

# removendo a linha com NaN no retorno
df = df.dropna(subset=['retorno'])

df_2010 = df[df['data'] >= '2010-01-01'].copy()

dados_retorno = df_2010[['retorno']].values

# 80% para treino, 20% para teste
tamanho_treino = int(len(dados_retorno) * 0.8)
treino_bruto = dados_retorno[:tamanho_treino]
teste_bruto = dados_retorno[tamanho_treino:]

# normalização [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
treino_normalizado = scaler.fit_transform(treino_bruto)
teste_normalizado = scaler.transform(teste_bruto)

# criar janelas deslizantes
def criar_sequencias(dados, janela):
    X, y = [], []
    for i in range(len(dados) - janela):
        # Pega o bloco de dias (ex: 21 dias)
        sequencia_x = dados[i : i + janela]
        # Pega o dia seguinte para ser o alvo da previsão
        alvo_y = dados[i + janela]
        
        X.append(sequencia_x)
        y.append(alvo_y)
    
    return np.array(X), np.array(y)

JANELA_DIAS = 21

x_treino, y_treino = criar_sequencias(treino_normalizado, JANELA_DIAS)
x_teste, y_teste = criar_sequencias(teste_normalizado, JANELA_DIAS)

x_treino = x_treino.astype(np.float32)
y_treino = y_treino.astype(np.float32)
x_teste = x_teste.astype(np.float32)
y_teste = y_teste.astype(np.float32)

print("Formato do x_treino:", x_treino.shape)

# ========================================================= #

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    print("Usando GPU: ", torch.cuda.get_device_name(0))
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# ========================================================= #

net = NeuralNetRegressor(
    module = ModeloLSTM,
    criterion = nn.MSELoss,
    optimizer = optim.Adam,
    max_epochs = 50,
    batch_size = 16,
    verbose = 0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

grid_params = {
    #'module__hidden_size': [20, 50],
    #'module__num_layers': [1, 2],
    #'optimizer__lr': [0.001, 0.01],
    #'max_epochs': [10, 20, 50]    
    'module__hidden_size': [50],
    'module__num_layers': [1],
    'optimizer__lr': [0.001],
    'max_epochs': [50]
}

# tscv = TimeSeriesSplit(n_splits=3)
tscv = TimeSeriesSplit(n_splits=2)

gs = GridSearchCV(net, grid_params, refit=True, cv=tscv, scoring='neg_mean_squared_error', verbose=2)
gs.fit(x_treino, y_treino)

print("Melhores parâmetros encontrados:", gs.best_params_)
print("Melhor MSE:", gs.best_score_)

# ========================================================= #

y_teste_previsto = gs.predict(x_teste)

# métricas do LSTM
rmse = np.sqrt(mean_squared_error(y_teste, y_teste_previsto))
mape = mean_absolute_percentage_error(y_teste, y_teste_previsto)

print("RESULTADOS FINAIS LSTM (TESTE)")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}")

# ========================================================= #
# Teste de regressão com SVM

x_treino_2d = x_treino.reshape(x_treino.shape[0], -1)
x_teste_2d = x_teste.reshape(x_teste.shape[0], -1)

# GridSearch para SVR
svr_params = {
    'C': [0.1, 1.0, 10.0],
    'gamma': ['scale', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}

svr_model = SVR()
gs_svr = GridSearchCV(svr_model, svr_params, cv=tscv, scoring='neg_mean_squared_error', verbose=2)
gs_svr.fit(x_treino_2d, y_treino.ravel())

print("Melhores parâmetros SVR encontrados:", gs_svr.best_params_)
print("Melhor MSE SVR:", gs_svr.best_score_)

y_teste_previsto_svr = gs_svr.predict(x_teste_2d)

# métricas do SVR
rmse_svr = np.sqrt(mean_squared_error(y_teste, y_teste_previsto_svr))
mape_svr = mean_absolute_percentage_error(y_teste, y_teste_previsto_svr)

print("RESULTADOS FINAIS SVR (TESTE)")
print(f"RMSE SVR: {rmse_svr:.4f}")
print(f"MAPE SVR: {mape_svr:.4f}")

# ========================================================= #

plt.figure(figsize=(14, 5))

plt.plot(y_teste, label='Retorno Real', color='lightblue', alpha=0.7)
plt.plot(y_teste_previsto, label='Previsão LSTM', color='red', linewidth=1.5, alpha=0.9)
plt.plot(y_teste_previsto_svr, label='Previsão SVR', color='green', linewidth=1.5, linestyle='--', alpha=0.9)

plt.title('Comparação: Retorno real vs LSTM vs SVR')
plt.xlabel('Dias (Teste)')
plt.ylabel('Retorno diário')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()