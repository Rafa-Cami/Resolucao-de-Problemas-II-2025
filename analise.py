# FILTRAGEM INICIAL: 
# FUNÇÃO: EDUCAÇÃO / SUBFUNÇÃO DO GOVERNO: ENSINO SUPERIOR
# DE 2017 A 2024

# FILTRAGEM ADICIONAL:
# coluna a mais marcando pré pós e pandemia - OKAY
# tirar tuplas de Unidade Orçamentária != UNIVERSIDADE - OKAY
# agrupar universidade para somar investimentos por mês > mais tarde com o grupo
# tirar coluna K (UF PT) - OKAY

import pandas as pd

print("Análise de Dados Iniciada")

df_dados = pd.read_csv(
    '2017-2024.csv',
    delimiter=";",
    decimal=",",        # decimal é vírgula
    thousands=".",      # separador de milhar é ponto
    encoding="latin1"   # corrige acentuação
)

print("CSV carregado com sucesso")

# Remover coluna indesejada
df_dados.drop('UF PT', axis=1, inplace=True)

# Criar coluna Período
df_dados.loc[(df_dados['Ano Lançamento'] >= 2017) & (df_dados['Ano Lançamento'] < 2020), 'Período'] = 'PRÉ-PANDEMIA'
df_dados.loc[(df_dados['Ano Lançamento'] >= 2020) & (df_dados['Ano Lançamento'] <= 2021), 'Período'] = 'PANDEMIA'
df_dados.loc[(df_dados['Ano Lançamento'] > 2021), 'Período'] = 'PÓS-PANDEMIA'

# Filtrar unidade orçamentária
df_dados = df_dados[
    (df_dados['Unidade Orçamentária'].str.contains('UNIVERSIDADE')) |
    (df_dados['Unidade Orçamentária'].str.contains('UNIV'))
]

# Salvar de volta
df_dados.to_csv("dados.csv", index=False, sep=";", decimal=",", encoding="latin1")

print("CSV atualizado e salvo")
