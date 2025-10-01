# config/settings.py
"""
Modulo de configuracao centralizado para o projeto.
"""
from datetime import datetime

# --- Configuracoes de Arquivos e Logs ---
DATA_FILE_PATH = 'dadosOriginais (2).xlsx'
MODEL_SAVE_PATH = f'rf_model_investimento_aluno_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
LOG_FILE_PATH = f'rf_investimento_aluno_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
PLOT_SAVE_PATH = f'rf_investimento_aluno_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'

# --- Nomes das Colunas ---
COL_ANO = 'Ano Lançamento'
COL_MOVIMENTO_LIQUIDO = 'Movim. Líquido - R$_destino'
COL_QTD_ALUNOS = 'Quantidade Alunos'
CATEGORICAL_COLS = [
    'UO - Orgao Maximo', 'Unidade Orcamentaria', 'Modalidade Aplicacao',
    'Funcao Governo', 'Subfuncao Governo', 'Programa Governo', 'Acao Governo',
    'Regiao PT', 'UF - desc', 'Municipio PT'
]

# --- Variaveis de Feature Engineering ---
TARGET_COL = 'Investimento_por_Aluno'

# --- Configuracoes do Modelo ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
OPTIMIZE_HYPERPARAMS = True
PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
DEFAULT_RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
