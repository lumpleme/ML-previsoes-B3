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
import matplotlib.patches as mpatches

JANELA_DIAS = 21
BASE = 'dados/ibov_multivar.csv'

class ModeloLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=1, output_size=1):
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
    

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ========================================================= #

# carregando os dados
df = pd.read_csv(BASE)
df['date'] = pd.to_datetime(df['date'])

# removendo NaN
df = df.dropna().reset_index(drop=True)

# montando os conjuntos para treinamento
dados_x = df[['open', 'high', 'low', 'closed', 'vol']]
dados_y = df['return']

# separando 80% para treino e 20% para teste
tamanho_treino = int(len(dados_y) * 0.8)
treino_x_bruto = dados_x[:tamanho_treino]
treino_y_bruto = dados_y[:tamanho_treino]
teste_x_bruto = dados_x[tamanho_treino:]
teste_y_bruto = dados_y[tamanho_treino:]

# normalização [0, 1]
scaler_x = MinMaxScaler(feature_range=(0, 1))
treino_x_normalizado = scaler_x.fit_transform(treino_x_bruto)
teste_x_normalizado = scaler_x.transform(teste_x_bruto)

scaler_y = MinMaxScaler(feature_range=(0, 1))
treino_y_normalizado = scaler_y.fit_transform(treino_y_bruto.values.reshape(-1, 1))
teste_y_normalizado = scaler_y.transform(teste_y_bruto.values.reshape(-1, 1))

# criar janelas deslizantes multivariadas
def criar_sequencias_multi(dados_x, dados_y, janela):
    X, y = [], []
    for i in range(len(dados_x) - janela):
        # pega o bloco de dias
        sequencia_x = dados_x[i : i + janela]
        # pega o dia seguinte como alvo
        alvo_y = dados_y[i + janela]
        
        X.append(sequencia_x)
        y.append(alvo_y)
    
    return np.array(X, dtype = np.float32), np.array(y, dtype = np.float32)

x_treino, y_treino = criar_sequencias_multi(treino_x_normalizado, treino_y_normalizado, JANELA_DIAS)
x_teste, y_teste = criar_sequencias_multi(teste_x_normalizado, teste_y_normalizado, JANELA_DIAS)

print("Formato do x_treino:", x_treino.shape)
print("Formato do y_treino:", y_treino.shape)

# ========================================================= #

# teste de regressão com o LSTM

net = NeuralNetRegressor(
    module = ModeloLSTM,
    criterion = nn.MSELoss,
    optimizer = optim.Adam,
    batch_size = 16,
    verbose = 0,
    device='cpu'
)

# definindo os hiperparâmetros para o GridSearch
grid_params = {
    'module__hidden_size': [20, 50],
    'module__num_layers': [2, 3],
    'optimizer__lr': [0.001],
    'max_epochs': [20, 50, 100]
}

tscv = TimeSeriesSplit(n_splits=3)

# realizando o GridSearch para o LSTM
gs_lstm = GridSearchCV(net, grid_params, refit=True, cv=tscv, scoring='neg_mean_squared_error', verbose=2)
gs_lstm.fit(x_treino, y_treino)

print("Melhores hiperparâmetros LSTM encontrados:", gs_lstm.best_params_)
print("Melhor MSE:", gs_lstm.best_score_)

# calculando as previsões no conjunto de teste
y_teste_previsto_lstm = gs_lstm.predict(x_teste)

# ========================================================= #

# pegando todos os resultados para comparação
resultados_grid = pd.DataFrame(gs_lstm.cv_results_)

cols = [
    'param_max_epochs', 
    'param_module__hidden_size', 
    'param_module__num_layers', 
    'param_optimizer__lr', 
    'mean_test_score'
]
tabela_comp = resultados_grid[cols].copy()

# MSE sempre positivo
tabela_comp['MSE'] = np.abs(tabela_comp['mean_test_score'])
tabela_comp = tabela_comp.drop(columns=['mean_test_score'])

# ordenar do melhor para o pior
tabela_comp = tabela_comp.sort_values(by='MSE').reset_index(drop=True)

tabela_comp.columns = ['Épocas', 'Neurônios', 'Camadas', 'Taxa de aprendizado (LR)', 'MSE']

print("\nTABELA COMPARATIVA DE HIPERPARÂMETROS LSTM:")
print(tabela_comp.to_string())

# guardando em um arquivo
tabela_comp.to_csv('tabela_comp_lstm_multivar.csv', index=False)

# ========================================================= #

# teste de regressão com SVM

# transformando os dados em 2D para o SVR
x_treino_2d = x_treino.reshape(x_treino.shape[0], -1)
x_teste_2d = x_teste.reshape(x_teste.shape[0], -1)

# definindo os hiperparâmetros para o GridSearch
svr_params = {
    'C': [0.1, 1.0, 10.0, 100.0, 1000.0],
    'gamma': ['scale', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}

svr_model = SVR()

# realizando o GridSearch para o SVR
gs_svr = GridSearchCV(svr_model, svr_params, cv=tscv, scoring='neg_mean_squared_error', verbose=2)
gs_svr.fit(x_treino_2d, y_treino.ravel())

print("Melhores hiperparâmetros SVR encontrados:", gs_svr.best_params_)
print("Melhor MSE SVR:", gs_svr.best_score_)

# calculando as previsões no conjunto de teste
y_teste_previsto_svr = gs_svr.predict(x_teste_2d)

# ========================================================= #

# pegando todos os resultados para comparação
resultados_grid_svr = pd.DataFrame(gs_svr.cv_results_)

cols_svr = [
    'param_C', 
    'param_kernel', 
    'param_gamma', 
    'mean_test_score'
]
tabela_comp_svr = resultados_grid_svr[cols_svr].copy()

# MSE sempre positivo
tabela_comp_svr['MSE'] = np.abs(tabela_comp_svr['mean_test_score'])
tabela_comp_svr = tabela_comp_svr.drop(columns=['mean_test_score'])

# ordenar do melhor para o pior
tabela_comp_svr = tabela_comp_svr.sort_values(by='MSE').reset_index(drop=True)
tabela_comp_svr.columns = ['C (Regularização)', 'Kernel', 'Gamma', 'MSE']

print("\nTABELA COMPARATIVA DE HIPERPARÂMETROS SVR")
print(tabela_comp_svr.to_string())

# guardando em um arquivo
tabela_comp_svr.to_csv('tabela_comp_svr_multivar.csv', index=False)

# ========================================================= #

# desnormalizando as previsões para comparação visual
y_teste_previsto_lstm = scaler_y.inverse_transform(y_teste_previsto_lstm.reshape(-1, 1)).flatten()
y_teste_previsto_svr = scaler_y.inverse_transform(y_teste_previsto_svr.reshape(-1, 1)).flatten()
y_teste = scaler_y.inverse_transform(y_teste.reshape(-1, 1)).flatten()

# métricas do LSTM
rmse = np.sqrt(mean_squared_error(y_teste, y_teste_previsto_lstm))
mape = mean_absolute_percentage_error(y_teste, y_teste_previsto_lstm)

print("RESULTADOS FINAIS LSTM (TESTE)")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}")

# métricas do SVR
rmse_svr = np.sqrt(mean_squared_error(y_teste, y_teste_previsto_svr))
mape_svr = mean_absolute_percentage_error(y_teste, y_teste_previsto_svr)

print("RESULTADOS FINAIS SVR (TESTE)")
print(f"RMSE SVR: {rmse_svr:.4f}")
print(f"MAPE SVR: {mape_svr:.4f}")

# montando o gráfico comparativo
plt.figure(figsize=(14, 5))

plt.plot(y_teste, label='Retorno real', color='lightblue', alpha=0.7)
plt.plot(y_teste_previsto_lstm, label='Previsão LSTM', color='red', linewidth=1.5, alpha=0.9)
plt.plot(y_teste_previsto_svr, label='Previsão SVR', color='green', linewidth=1.5, linestyle='--', alpha=0.9)

plt.title('Comparação Multivar: Retorno real vs LSTM vs SVR')
plt.xlabel('Dias (teste)')
plt.ylabel('Retorno diário')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================================= #

# calculando o Erro Absoluto Diário e o Acerto Direcional para o LSTM

erros_absolutos = np.abs(y_teste - y_teste_previsto_lstm)

# verificando quais acertou a direção
acertou_direcao = np.sign(y_teste) == np.sign(y_teste_previsto_lstm)

cores_barras = ['green' if acertou else 'red' for acertou in acertou_direcao[:45]]

# montando o gráfico (apenas 45 dias)
plt.figure(figsize=(12, 5))
plt.bar(range(45), erros_absolutos[:45], color=cores_barras, alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title('LSTM: Erro Absoluto e Acerto Direcional (teste) - 45 dias')
plt.xlabel('Dias (teste)')
plt.ylabel('Erro Absoluto')

# legenda
patch_verde = mpatches.Patch(color='green', label='Acertou a direção')
patch_vermelho = mpatches.Patch(color='red', label='Errou a direção')
plt.legend(handles=[patch_verde, patch_vermelho], loc='upper left')

# linha horizontal no zero 
plt.axhline(0, color='black', linewidth=1)
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.xticks(range(0, 45, 5))
plt.tight_layout()
plt.show()

# calculando a taxa de acerto direcional total
taxa_acerto = (sum(acertou_direcao) / len(acertou_direcao)) * 100
print(f"\nTaxa de Acerto Direcional LSTM de todo o período de teste: {taxa_acerto:.2f}%")

# percentual de acerto direcional das primeiras 10 previsões
taxa_acerto_10 = (sum(acertou_direcao[:10]) / 10) * 100
print(f"Taxa de Acerto Direcional LSTM das primeiras 10 previsões: {taxa_acerto_10:.2f}%")

# ========================================================= #

# calculando o Erro Absoluto Diário e o Acerto Direcional para o SVR

erros_absolutos_svr = np.abs(y_teste - y_teste_previsto_svr)

# verificando quais acertou a direção
acertou_direcao_svr = np.sign(y_teste) == np.sign(y_teste_previsto_svr)

cores_barras_svr = ['green' if acertou else 'red' for acertou in acertou_direcao_svr[:45]]

# montando o gráfico (apenas 45 dias)
plt.figure(figsize=(12, 5))
plt.bar(range(45), erros_absolutos_svr[:45], color=cores_barras_svr, alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title('SVR: Erro Absoluto e Acerto Direcional (teste) - 45 dias')
plt.xlabel('Dias (teste)')
plt.ylabel('Erro Absoluto')

# legenda
patch_verde = mpatches.Patch(color='green', label='Acertou a direção')
patch_vermelho = mpatches.Patch(color='red', label='Errou a direção')
plt.legend(handles=[patch_verde, patch_vermelho], loc='upper left')

# linha horizontal no zero 
plt.axhline(0, color='black', linewidth=1)
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.xticks(range(0, 45, 5))
plt.tight_layout()
plt.show()

# calculando a taxa de acerto direcional total
taxa_acerto_svr = (sum(acertou_direcao_svr) / len(acertou_direcao_svr)) * 100
print(f"\nTaxa de Acerto Direcional SVR de todo o período de teste: {taxa_acerto_svr:.2f}%")

# percentual de acerto direcional das primeiras 10 previsões
taxa_acerto_10_svr = (sum(acertou_direcao_svr[:10]) / 10) * 100
print(f"Taxa de Acerto Direcional SVR das primeiras 10 previsões: {taxa_acerto_10_svr:.2f}%")