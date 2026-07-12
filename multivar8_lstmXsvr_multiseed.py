import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import random
from sklearn.svm import SVR
import matplotlib.patches as mpatches

# ========================================================= #

SEEDS = [10, 429, 128963, 7, 9998]
JANELA_DIAS = 21
BASE = 'dados/petr4_multivar_indicadores.csv'

# ========================================================= #

class ModeloLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2, output_size=1):
        super(ModeloLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out


def criar_sequencias_multi(dados_x, dados_y, janela):
    X, y = [], []
    for i in range(len(dados_x) - janela):
        X.append(dados_x[i : i + janela])
        y.append(dados_y[i + janela])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ========================================================= #
# carregando e preparando os dados (fora do loop de seeds)

df = pd.read_csv(BASE)
df['date'] = pd.to_datetime(df['date'])

df = df.dropna().reset_index(drop=True)


dados_x = df[['open', 'high', 'low', 'closed', 'vol', 'SMA_9', 'SMA_21', 'RSI']]
dados_y = df['return']

tamanho_treino = int(len(dados_y) * 0.8)
treino_x_bruto = dados_x[:tamanho_treino]
treino_y_bruto = dados_y[:tamanho_treino]
teste_x_bruto  = dados_x[tamanho_treino:]
teste_y_bruto  = dados_y[tamanho_treino:]

# normalização [0, 1], com fit no treino e aplicação no teste
scaler_x = MinMaxScaler(feature_range=(0, 1))
treino_x_norm = scaler_x.fit_transform(treino_x_bruto)
teste_x_norm  = scaler_x.transform(teste_x_bruto)

scaler_y = MinMaxScaler(feature_range=(0, 1))
treino_y_norm = scaler_y.fit_transform(treino_y_bruto.values.reshape(-1, 1))
teste_y_norm  = scaler_y.transform(teste_y_bruto.values.reshape(-1, 1))

x_treino, y_treino = criar_sequencias_multi(treino_x_norm, treino_y_norm, JANELA_DIAS)
x_teste,  y_teste  = criar_sequencias_multi(teste_x_norm,  teste_y_norm,  JANELA_DIAS)

x_treino_2d = x_treino.reshape(x_treino.shape[0], -1)
x_teste_2d  = x_teste.reshape(x_teste.shape[0], -1)

# y real desnormalizado (igual para todas as seeds)
y_teste_real = scaler_y.inverse_transform(y_teste.reshape(-1, 1)).flatten()

tscv = TimeSeriesSplit(n_splits=3)

# ========================================================= #

# definindo os hiperparâmetros para o GridSearch
grid_params_lstm = {
    'module__hidden_size': [20, 50],
    'module__num_layers': [2, 3],
    'optimizer__lr': [0.001],
    'max_epochs': [20, 50, 100]
}

svr_params = {
    'C': [0.1, 1.0, 10.0, 100.0, 1000.0],
    'gamma': ['scale', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}

# ========================================================= #
# loop pelas seeds

consolidado = []

for seed in SEEDS:
    print(f"\nSEED: {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- LSTM ---
    net = NeuralNetRegressor(
        module=ModeloLSTM,
        criterion=nn.L1Loss,
        optimizer=optim.Adam,
        batch_size=16,
        verbose=0,
        device='cpu'
    )

    gs_lstm = GridSearchCV(net, grid_params_lstm, refit=True, cv=tscv,
                           scoring='neg_mean_absolute_error', verbose=2)
    gs_lstm.fit(x_treino, y_treino)

    print("Melhores hiperparâmetros LSTM:", gs_lstm.best_params_)

    # salvando tabela parcial do LSTM
    res_lstm = pd.DataFrame(gs_lstm.cv_results_)
    cols_lstm = ['param_max_epochs', 'param_module__hidden_size',
                 'param_module__num_layers', 'param_optimizer__lr', 'mean_test_score']
    tab_lstm = res_lstm[cols_lstm].copy()
    tab_lstm['MAE'] = np.abs(tab_lstm['mean_test_score'])
    tab_lstm = tab_lstm.drop(columns=['mean_test_score']).sort_values('MAE').reset_index(drop=True)
    tab_lstm.columns = ['Épocas', 'Neurônios', 'Camadas', 'LR', 'MAE']
    tab_lstm.to_csv(f'tabela_lstm_seed{seed}.csv', index=False)

    # --- SVR ---
    gs_svr = GridSearchCV(SVR(), svr_params, cv=tscv,
                          scoring='neg_mean_absolute_error', verbose=2)
    gs_svr.fit(x_treino_2d, y_treino.ravel())

    print("Melhores hiperparâmetros SVR:", gs_svr.best_params_)

    # salvando tabela parcial do SVR
    res_svr = pd.DataFrame(gs_svr.cv_results_)
    cols_svr = ['param_C', 'param_kernel', 'param_gamma', 'mean_test_score']
    tab_svr = res_svr[cols_svr].copy()
    tab_svr['MAE'] = np.abs(tab_svr['mean_test_score'])
    tab_svr = tab_svr.drop(columns=['mean_test_score']).sort_values('MAE').reset_index(drop=True)
    tab_svr.columns = ['C', 'Kernel', 'Gamma', 'MAE']
    tab_svr.to_csv(f'tabela_svr_seed{seed}.csv', index=False)

    # --- previsões desnormalizadas ---
    prev_lstm_norm = gs_lstm.predict(x_teste)
    prev_svr_norm  = gs_svr.predict(x_teste_2d)

    prev_lstm = scaler_y.inverse_transform(prev_lstm_norm.reshape(-1, 1)).flatten()
    prev_svr  = scaler_y.inverse_transform(prev_svr_norm.reshape(-1, 1)).flatten()

    # --- métricas LSTM ---
    rmse_lstm = np.sqrt(mean_squared_error(y_teste_real, prev_lstm))
    mape_lstm = mean_absolute_percentage_error(y_teste_real, prev_lstm)
    acerto_dir_lstm    = np.mean(np.sign(y_teste_real) == np.sign(prev_lstm)) * 100
    acerto_dir_lstm_10 = np.mean(np.sign(y_teste_real[:10]) == np.sign(prev_lstm[:10])) * 100

    # --- métricas SVR ---
    rmse_svr = np.sqrt(mean_squared_error(y_teste_real, prev_svr))
    mape_svr = mean_absolute_percentage_error(y_teste_real, prev_svr)
    acerto_dir_svr    = np.mean(np.sign(y_teste_real) == np.sign(prev_svr)) * 100
    acerto_dir_svr_10 = np.mean(np.sign(y_teste_real[:10]) == np.sign(prev_svr[:10])) * 100

    print(f"LSTM  RMSE: {rmse_lstm:.4f} | MAPE: {mape_lstm:.4f} | Acerto direcional: {acerto_dir_lstm:.2f}% | Acerto 10 primeiros: {acerto_dir_lstm_10:.2f}%")
    print(f"SVR   RMSE: {rmse_svr:.4f} | MAPE: {mape_svr:.4f} | Acerto direcional: {acerto_dir_svr:.2f}% | Acerto 10 primeiros: {acerto_dir_svr_10:.2f}%")

    # --- gráfico comparativo ---
    plt.figure(figsize=(14, 5))
    plt.plot(y_teste_real, label='Retorno real', color='lightblue', alpha=0.7)
    plt.plot(prev_lstm, label='Previsão LSTM', color='red', linewidth=1.5, alpha=0.9)
    plt.plot(prev_svr, label='Previsão SVR', color='green', linewidth=1.5, linestyle='--', alpha=0.9)
    plt.title(f'Seed {seed}: Retorno real vs LSTM vs SVR')
    plt.xlabel('Dias (teste)')
    plt.ylabel('Retorno diário')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'grafico_comparativo_seed{seed}.png', dpi=110)
    plt.close()

    # --- gráfico erro absoluto + acerto direcional LSTM ---
    erros_lstm = np.abs(y_teste_real - prev_lstm)
    acertou_lstm = np.sign(y_teste_real) == np.sign(prev_lstm)
    cores_lstm = ['green' if a else 'red' for a in acertou_lstm[:45]]

    plt.figure(figsize=(12, 5))
    plt.bar(range(45), erros_lstm[:45], color=cores_lstm, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title(f'Seed {seed}: LSTM: Erro Absoluto e Acerto Direcional (45 dias)')
    plt.xlabel('Dias (teste)')
    plt.ylabel('Erro Absoluto')
    patch_v = mpatches.Patch(color='green', label='Acertou a direção')
    patch_r = mpatches.Patch(color='red', label='Errou a direção')
    plt.legend(handles=[patch_v, patch_r], loc='upper left')
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.xticks(range(0, 45, 5))
    plt.tight_layout()
    plt.savefig(f'grafico_erros_lstm_seed{seed}.png', dpi=110)
    plt.close()

    # --- gráfico erro absoluto + acerto direcional SVR ---
    erros_svr = np.abs(y_teste_real - prev_svr)
    acertou_svr = np.sign(y_teste_real) == np.sign(prev_svr)
    cores_svr = ['green' if a else 'red' for a in acertou_svr[:45]]

    plt.figure(figsize=(12, 5))
    plt.bar(range(45), erros_svr[:45], color=cores_svr, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.title(f'Seed {seed}: SVR: Erro Absoluto e Acerto Direcional (45 dias)')
    plt.xlabel('Dias (teste)')
    plt.ylabel('Erro Absoluto')
    plt.legend(handles=[patch_v, patch_r], loc='upper left')
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.xticks(range(0, 45, 5))
    plt.tight_layout()
    plt.savefig(f'grafico_erros_svr_seed{seed}.png', dpi=110)
    plt.close()

    # --- acumulando para a tabela consolidada ---
    bp = gs_lstm.best_params_
    bp_svr = gs_svr.best_params_

    consolidado.append({
        'Seed': seed,
        # LSTM
        'LSTM_Épocas':    bp.get('max_epochs'),
        'LSTM_Neurônios': bp.get('module__hidden_size'),
        'LSTM_Camadas':   bp.get('module__num_layers'),
        'LSTM_LR':        bp.get('optimizer__lr'),
        'LSTM_RMSE':      round(rmse_lstm, 6),
        'LSTM_MAPE':      round(mape_lstm, 6),
        'LSTM_Acerto_Dir_%':    round(acerto_dir_lstm, 2),
        'LSTM_Acerto_10pri_%':  round(acerto_dir_lstm_10, 2),
        # SVR
        'SVR_C':      bp_svr.get('C'),
        'SVR_Kernel': bp_svr.get('kernel'),
        'SVR_Gamma':  bp_svr.get('gamma'),
        'SVR_RMSE':   round(rmse_svr, 6),
        'SVR_MAPE':   round(mape_svr, 6),
        'SVR_Acerto_Dir_%':   round(acerto_dir_svr, 2),
        'SVR_Acerto_10pri_%': round(acerto_dir_svr_10, 2),
    })

# ========================================================= #
# tabela consolidada final

df_consolid = pd.DataFrame(consolidado)

print("\nTABELA CONSOLIDADA DE TODAS AS SEEDS")
print(df_consolid.to_string(index=False))

df_consolid.to_csv('tabela_consolidada_multiseed.csv', index=False)
print("\nSalvo em: tabela_consolidada_multiseed.csv")
