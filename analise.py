import pandas as pd

print("Análise de Dados Iniciada")

# FILTRAGEM INICIAL: 
# FUNÇÃO: EDUCAÇÃO / SUBFUNÇÃO DO GOVERNO: ENSINO SUPERIOR
# DE 2017 A 2024

# FILTRAGEM ADICIONAL:
# coluna a mais marcando pré pós e pandemia - OKAY
# tirar tuplas de Unidade Orçamentária != UNIVERSIDADE - OKAY
# agrupar universidade para somar investimentos por mês > mais tarde com o grupo
# tirar coluna K (UF PT) - OKAY

df_dados = pd.read_csv('dados.csv', delimiter=";")
print("CSV carregado com sucesso")

# Sem coluna K
df_dados.drop('UF PT', axis=1, inplace=True)

# Adicionando coluna Período
df_dados.loc[(df_dados['Ano Lançamento'] >= 2017) & (df_dados['Ano Lançamento'] < 2020), 'Período'] = 'PRÉ-PANDEMIA'
df_dados.loc[(df_dados['Ano Lançamento'] >= 2020) & (df_dados['Ano Lançamento'] <= 2021), 'Período'] = 'PANDEMIA'
df_dados.loc[(df_dados['Ano Lançamento'] > 2021), 'Período'] = 'PÓS-PANDEMIA'


# Filtrando coluna de unidade orçamentária
df_dados = df_dados[(df_dados['Unidade Orçamentária'].str.contains('UNIVERSIDADE'))
                    | (df_dados['Unidade Orçamentária'].str.contains('UNIV'))]

df_dados.to_csv("dados.csv", index=False, sep=";")
print("CSV atualizado e salvo")

