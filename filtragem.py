import pandas as pd

print("Análise de Dados Iniciada")

df_dados = pd.read_csv(
    '2017-2024.csv',
    delimiter=";",
    decimal=",",
    thousands=".",
    encoding="latin1"
)

print("CSV carregado com sucesso")
print(f"Total de registros originais: {len(df_dados)}")

# Remover coluna indesejada
df_dados.drop('UF PT', axis=1, inplace=True)

# CORREÇÃO PARA 2020: Inverter as colunas "UO - Órgão Máximo" e "Unidade Orçamentária"
# Primeiro, vamos verificar os nomes exatos das colunas
print("Colunas disponíveis:", df_dados.columns.tolist())

# CORREÇÃO: Para registros de 2020, inverter as duas colunas
mask_2020 = df_dados['Ano Lançamento'] == 2020

# Salvar os valores temporariamente
temp_uo = df_dados.loc[mask_2020, 'UO - Órgão Máximo'].copy()
temp_uorc = df_dados.loc[mask_2020, 'Unidade Orçamentária'].copy()

# Inverter os valores
df_dados.loc[mask_2020, 'UO - Órgão Máximo'] = temp_uorc
df_dados.loc[mask_2020, 'Unidade Orçamentária'] = temp_uo

print(f"Registros de 2020 corrigidos: {mask_2020.sum()}")

# Criar coluna Período
df_dados['Período'] = 'PÓS-PANDEMIA'
df_dados.loc[df_dados['Ano Lançamento'] <= 2019, 'Período'] = 'PRÉ-PANDEMIA'
df_dados.loc[(df_dados['Ano Lançamento'] >= 2020) & (df_dados['Ano Lançamento'] <= 2021), 'Período'] = 'PANDEMIA'

# Filtrar unidade orçamentária
df_dados = df_dados[
    (df_dados['Unidade Orçamentária'].str.contains('UNIVERSIDADE', na=False)) |
    (df_dados['Unidade Orçamentária'].str.contains('UNIV', na=False))
]

# Salvar de volta
df_dados.to_csv("dadosOriginais.csv", index=False, sep=";", decimal=",", encoding="latin1")

print("CSV atualizado e salvo")
print(f"Distribuição por período:")
print(df_dados['Período'].value_counts())
print(f"Distribuição por ano:")
print(df_dados['Ano Lançamento'].value_counts().sort_index())

# Verificação adicional
print("\nVerificação das colunas corrigidas para 2020:")
print(df_dados[df_dados['Ano Lançamento'] == 2020][['UO - Órgão Máximo', 'Unidade Orçamentária']].head())