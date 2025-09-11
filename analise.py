import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

plt.style.use('seaborn-v0_8')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Carregando dados tratados...")
df = pd.read_csv('dadosOriginais.csv', delimiter=";", decimal=",", encoding="latin1")

print(f"Shape do dataset: {df.shape}")
print(f"Colunas disponíveis: {list(df.columns)}")

df.columns = df.columns.str.strip()
print(f"Colunas após strip: {list(df.columns)}")

if 'Movim. Líquido - R$' in df.columns:
    df.rename(columns={'Movim. Líquido - R$': 'Valor'}, inplace=True)
    print("Coluna 'Movim. Líquido - R$' renomeada para 'Valor'")

df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')

print("\nValores únicos em 'Ano Lançamento':")
print(df['Ano Lançamento'].unique())
print("\nValores únicos em 'Mês Base Lançamento':")
print(df['Mês Base Lançamento'].unique())

meses_para_numero = {
    'JANEIRO': 1, 'FEVEREIRO': 2, 'MARCO': 3, 'ABRIL': 4,
    'MAIO': 5, 'JUNHO': 6, 'JULHO': 7, 'AGOSTO': 8,
    'SETEMBRO': 9, 'OUTUBRO': 10, 'NOVEMBRO': 11, 'DEZEMBRO': 12
}

df['Mês_Numero'] = df['Mês Base Lançamento'].map(meses_para_numero)

meses_nao_mapeados = df[df['Mês_Numero'].isnull()]['Mês Base Lançamento'].unique()
if len(meses_nao_mapeados) > 0:
    print(f"Aviso: Meses não mapeados encontrados: {meses_nao_mapeados}")
    # Preencher valores faltantes com 1 (Janeiro) como padrão
    df['Mês_Numero'] = df['Mês_Numero'].fillna(1)

df['Data'] = pd.to_datetime(
    df['Ano Lançamento'].astype(str) + '-' +
    df['Mês_Numero'].astype(str).str.zfill(2) + '-01',
    format='%Y-%m-%d',
    errors='coerce'
)

datas_validas = df['Data'].notnull().sum()
print(f"\nDatas criadas com sucesso: {datas_validas} de {len(df)}")

df = df.dropna(subset=['Data', 'Valor'])
print(f"Shape após remover dados inválidos: {df.shape}")

if len(df) == 0:
    print("Não há dados válidos após a limpeza.")
    # Tentar uma abordagem alternativa usando apenas o ano
    df = pd.read_csv('dadosOriginais.csv', delimiter=";", decimal=",", encoding="latin1")
    df.columns = df.columns.str.strip()
    if 'Movim. Líquido - R$' in df.columns:
        df.rename(columns={'Movim. Líquido - R$': 'Valor'}, inplace=True)
    df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
    df['Data'] = pd.to_datetime(df['Ano Lançamento'].astype(str) + '-01-01', errors='coerce')
    df = df.dropna(subset=['Data', 'Valor'])
    print(f"Shape usando apenas ano: {df.shape}")

investimentos_mensais = df.groupby(['Data', 'Período'])['Valor'].sum().reset_index()

investimentos_mensais.sort_values('Data', inplace=True)

print(f"Dados agrupados por mês: {len(investimentos_mensais)} registros")
print(f"Períodos disponíveis: {investimentos_mensais['Período'].unique()}")

if len(investimentos_mensais) < 2:
    # Se não temos dados mensais, agrupar por ano
    print("Poucos dados mensais. Agrupando por ano...")
    investimentos_mensais = df.groupby([df['Data'].dt.year, 'Período'])['Valor'].sum().reset_index()
    investimentos_mensais.rename(columns={'Data': 'Ano'}, inplace=True)
    investimentos_mensais['Data'] = pd.to_datetime(investimentos_mensais['Ano'].astype(str) + '-01-01')
    investimentos_mensais.sort_values('Data', inplace=True)

investimentos_mensais['Meses'] = np.arange(len(investimentos_mensais))

periodos_disponiveis = investimentos_mensais['Período'].unique()
print(f"Períodos disponíveis: {periodos_disponiveis}")

if 'PÓS-PANDEMIA' in periodos_disponiveis:
    investimentos_mensais['Pós_Pandemia'] = (investimentos_mensais['Período'] == 'PÓS-PANDEMIA').astype(int)
    investimentos_mensais['Interacao_Pos'] = investimentos_mensais['Meses'] * investimentos_mensais['Pós_Pandemia']

    X = investimentos_mensais[['Meses', 'Pós_Pandemia', 'Interacao_Pos']]
    X = sm.add_constant(X)  # Adicionar intercepto
    y = investimentos_mensais['Valor']

    print(f"Shape de X: {X.shape}, Shape de y: {y.shape}")

    if X.shape[0] >= X.shape[1]:
        modelo = sm.OLS(y, X).fit()

        print("\n" + "=" * 60)
        print("REGRESSÃO LINEAR - INVESTIMENTOS EM UNIVERSIDADES PÚBLICAS")
        print("=" * 60)
        print(modelo.summary())

        # Interpretação dos coeficientes
        print("\nINTERPRETAÇÃO DOS RESULTADOS:")
        print(f"Intercepto (β0): R$ {modelo.params['const']:,.2f}")
        print(f"Tendência mensal base (β1): R$ {modelo.params['Meses']:,.2f} por mês")
        print(f"Impacto imediato pós-pandemia (β2): R$ {modelo.params['Pós_Pandemia']:,.2f}")
        print(f"Mudança na tendência pós-pandemia (β3): R$ {modelo.params['Interacao_Pos']:,.2f} por mês")
    else:
        print("Dados insuficientes para regressão linear")
else:
    print("Período PÓS-PANDEMIA não encontrado nos dados")

# Estatísticas descritivas por período
print("\nESTATÍSTICAS DESCRITIVAS POR PERÍODO:")
for periodo in ['PRÉ-PANDEMIA', 'PANDEMIA', 'PÓS-PANDEMIA']:
    if periodo in investimentos_mensais['Período'].values:
        dados_periodo = investimentos_mensais[investimentos_mensais['Período'] == periodo]
        media = dados_periodo['Valor'].mean()
        desvio = dados_periodo['Valor'].std()
        count = len(dados_periodo)
        print(f"{periodo}: Média = R$ {media:,.2f}, Desvio = R$ {desvio:,.2f}, N = {count}")

fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

cores = {'PRÉ-PANDEMIA': 'blue', 'PANDEMIA': 'orange', 'PÓS-PANDEMIA': 'green'}
for periodo, cor in cores.items():
    if periodo in investimentos_mensais['Período'].values:
        dados_periodo = investimentos_mensais[investimentos_mensais['Período'] == periodo]
        ax1.scatter(dados_periodo['Data'], dados_periodo['Valor'],
                    color=cor, alpha=0.7, label=periodo, s=50)

if 'modelo' in locals():
    ax1.plot(investimentos_mensais['Data'], modelo.predict(X),
             color='red', linewidth=3, label='Tendência (Regressão)')

ax1.set_title('Evolução dos Investimentos em Universidades Públicas')
ax1.set_ylabel('Investimento (R$)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='y')  # Desativar notação científica

plt.tight_layout()
plt.savefig('analise_investimentos_universidades.png', dpi=300, bbox_inches='tight')
plt.show()

if 'Unidade Orçamentária' in df.columns:
    print("\n" + "=" * 60)
    print("ANÁLISE POR UNIVERSIDADE")
    print("=" * 60)

    investimentos_por_universidade = df.groupby(['Unidade Orçamentária', 'Período'])['Valor'].sum().unstack()

    colunas_periodo = investimentos_por_universidade.columns
    if 'PRÉ-PANDEMIA' in colunas_periodo and 'PÓS-PANDEMIA' in colunas_periodo:
        investimentos_por_universidade['Variação_Pós_Pré'] = (
                (investimentos_por_universidade['PÓS-PANDEMIA'] - investimentos_por_universidade['PRÉ-PANDEMIA']) /
                investimentos_por_universidade['PRÉ-PANDEMIA'].replace(0, np.nan) * 100
        )

        investimentos_por_universidade.sort_values('Variação_Pós_Pré', ascending=False, inplace=True)

        print("Top 10 universidades com maior variação pós-pandemia:")
        print(investimentos_por_universidade[['PRÉ-PANDEMIA', 'PANDEMIA', 'PÓS-PANDEMIA', 'Variação_Pós_Pré']].head(
            10).round(2))

        if len(investimentos_por_universidade) > 0:
            plt.figure(figsize=(12, 6))
            top_10 = investimentos_por_universidade.head(10)
            colunas_plot = [col for col in ['PRÉ-PANDEMIA', 'PANDEMIA', 'PÓS-PANDEMIA'] if col in top_10.columns]
            top_10[colunas_plot].plot(kind='bar')
            plt.title('Top 10 Universidades com Maior Variação de Investimentos')
            plt.ylabel('Investimento Total (R$)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('top_universidades.png', dpi=300, bbox_inches='tight')
            plt.show()

print("\nAnálise concluída! Resultados salvos em 'analise_investimentos_universidades.png'")
