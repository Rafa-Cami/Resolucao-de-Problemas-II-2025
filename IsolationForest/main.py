import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Criar diretório para salvar as figuras
os.makedirs('figuras', exist_ok=True)

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Redirecionar todos os prints para um arquivo TXT
class Tee:
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    
    def flush(self):
        for f in self.files:
            f.flush()

# Abrir arquivo para salvar os logs
log_file = open('resultados_analise.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

print("=" * 80)
print("ANÁLISE DE OUTLIERS EM DADOS FINANCEIROS DE UNIVERSIDADES BRASILEIRAS")
print("Período: 2017-2024 | Foco: Impacto da Pandemia de COVID-19")
print(f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 80)

# Carregar os dados
df = pd.read_excel('Cópia de Dados Finais (1).xlsx', sheet_name='RemoveDuplicatas')

print("\n1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
print("Primeiras linhas dos dados:")
print(df.head())
print(f"\nDimensões do dataset: {df.shape}")
print(f"\nColunas disponíveis: {df.columns.tolist()}")

# Verificar dados faltantes
print(f"\nDados faltantes por coluna:")
print(df.isnull().sum())

# Converter colunas numéricas
df['Ano Lançamento'] = df['Ano Lançamento'].astype(int)
df['Movim. Líquido - R$_destino'] = pd.to_numeric(df['Movim. Líquido - R$_destino'], errors='coerce')
df['Quantidade Alunos'] = pd.to_numeric(df['Quantidade Alunos'], errors='coerce')

# Remover linhas com valores inválidos
df_clean = df.dropna(subset=['Movim. Líquido - R$_destino', 'Quantidade Alunos'])

# EXCLUIR UNIVERSIDADE FEDERAL DE RONDONÓPOLIS
universidades_antes = df_clean['Unidade Orçamentária'].nunique()
df_clean = df_clean[~df_clean['Unidade Orçamentária'].str.contains('RONDONOPOLIS', case=False, na=False)]
universidades_depois = df_clean['Unidade Orçamentária'].nunique()

print(f"\n=== FILTRO APLICADO ===")
print(f"Universidades antes do filtro: {universidades_antes}")
print(f"Universidades depois do filtro: {universidades_depois}")
print(f"Universidade Federal de Rondonópolis excluída da análise")

# 2. CALCULAR INVESTIMENTO POR ALUNO POR UNIVERSIDADE E ANO

print("\n2. CÁLCULO DO INVESTIMENTO POR ALUNO")

# Agrupar por universidade e ano, somando o investimento total
investimento_agrupado = df_clean.groupby(['Unidade Orçamentária', 'Ano Lançamento']).agg({
    'Movim. Líquido - R$_destino': 'sum',
    'Quantidade Alunos': 'first',
    'Período': 'first',
    'UF - desc': 'first',
    'Região PT': 'first'
}).reset_index()

# Calcular investimento por aluno
investimento_agrupado['Investimento_por_Aluno'] = (
    investimento_agrupado['Movim. Líquido - R$_destino'] / investimento_agrupado['Quantidade Alunos']
)

print(f"Dataset após agrupamento: {investimento_agrupado.shape}")
print(f"Anos disponíveis: {sorted(investimento_agrupado['Ano Lançamento'].unique())}")

# 3. PREPARAR DADOS PARA ANÁLISE DE OUTLIERS

print("\n3. PREPARAÇÃO PARA DETECÇÃO DE OUTLIERS")

# Criar uma tabela pivot com anos como colunas
pivot_table = investimento_agrupado.pivot_table(
    index='Unidade Orçamentária',
    columns='Ano Lançamento',
    values='Investimento_por_Aluno',
    aggfunc='mean'
).fillna(0)

print(f"Formato da tabela pivot: {pivot_table.shape}")

# 4. APLICAR ALGORITMO ISOLATION FOREST

print("\n4. APLICAÇÃO DO ALGORITMO ISOLATION FOREST")

# Preparar os dados para o modelo
X = pivot_table.values

# Normalizar os dados
scaler = StandardScaler() # Padronização: média 0, desvio padrão 1
X_scaled = scaler.fit_transform(X)

# Configurar e treinar o modelo Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42,
    verbose=0
)

# Prever outliers
outlier_predictions = iso_forest.fit_predict(X_scaled)

# Adicionar previsões ao DataFrame
pivot_table['Is_Outlier'] = outlier_predictions
pivot_table['Is_Outlier'] = pivot_table['Is_Outlier'].map({1: 'Normal', -1: 'Outlier'})

# 5. ANÁLISE DOS RESULTADOS

print("\n5. RESULTADOS DA ANÁLISE DE OUTLIERS")

# Contar outliers
outlier_count = (outlier_predictions == -1).sum()
normal_count = (outlier_predictions == 1).sum()

print(f"\n=== RESULTADOS DO ISOLATION FOREST ===")
print(f"Total de universidades analisadas: {len(pivot_table)}")
print(f"Universidades normais: {normal_count}")
print(f"Universidades outliers: {outlier_count}")
print(f"Taxa de outliers: {outlier_count/len(pivot_table)*100:.2f}%")

# Listar universidades outliers
outliers_df = pivot_table[pivot_table['Is_Outlier'] == 'Outlier']
print(f"\n=== UNIVERSIDADES IDENTIFICADAS COMO OUTLIERS ===")
for i, university in enumerate(outliers_df.index, 1):
    print(f"{i:2d}. {university}")

# 6. VISUALIZAÇÕES - SALVANDO AS FIGURAS

print("\n6. GERAÇÃO DE VISUALIZAÇÕES")

# Figura 1: Distribuição de outliers
plt.figure(figsize=(10, 6))
sns.countplot(data=pivot_table.reset_index(), x='Is_Outlier')
plt.title('Distribuição de Universidades: Normais vs Outliers')
plt.xlabel('Classificação')
plt.ylabel('Número de Universidades')
plt.savefig('figuras/distribuicao_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 1 salva: distribuicao_outliers.png")

# Figura 2: Evolução temporal do investimento por aluno
plt.figure(figsize=(14, 8))

# Selecionar algumas universidades para visualização (normais e outliers)
sample_normal = pivot_table[pivot_table['Is_Outlier'] == 'Normal'].sample(5, random_state=42)
sample_outliers = outliers_df.sample(min(5, len(outliers_df)), random_state=42)

# Plotar universidades normais
for idx, row in sample_normal.iterrows():
    plt.plot(range(2017, 2025), row[list(range(2017, 2025))], 
             alpha=0.7, linewidth=2, label=f'{idx} (Normal)')

# Plotar universidades outliers
for idx, row in sample_outliers.iterrows():
    plt.plot(range(2017, 2025), row[list(range(2017, 2025))], 
             linewidth=3, linestyle='--', label=f'{idx} (OUTLIER)')

plt.xlabel('Ano')
plt.ylabel('Investimento por Aluno (R$)')
plt.title('Evolução do Investimento por Aluno (2017-2024)\nComparação entre Universidades Normais e Outliers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figuras/evolucao_investimento.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 2 salva: evolucao_investimento.png")

# Figura 3: Heatmap do investimento por aluno (apenas outliers)
plt.figure(figsize=(12, 8))
outliers_data = outliers_df[list(range(2017, 2025))]

# Normalizar por linha para melhor visualização
outliers_normalized = outliers_data.div(outliers_data.max(axis=1), axis=0)

sns.heatmap(outliers_normalized, 
            cmap='YlOrRd', 
            annot=False, 
            cbar_kws={'label': 'Investimento por Aluno (Normalizado)'})
plt.title('Padrões de Investimento por Aluno - Universidades Outliers (2017-2024)')
plt.xlabel('Ano')
plt.ylabel('Universidade')
plt.tight_layout()
plt.savefig('figuras/heatmap_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 3 salva: heatmap_outliers.png")

# Figura 4: Boxplot do investimento por aluno por período
plt.figure(figsize=(12, 6))
sns.boxplot(data=investimento_agrupado, x='Período', y='Investimento_por_Aluno')
plt.title('Distribuição do Investimento por Aluno por Período Pandêmico')
plt.xlabel('Período')
plt.ylabel('Investimento por Aluno (R$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('figuras/boxplot_periodos.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figura 4 salva: boxplot_periodos.png")

# 7. ANÁLISE ESTATÍSTICA DETALHADA

print("\n7. ANÁLISE ESTATÍSTICA DETALHADA")

# Estatísticas por período pandêmico
period_stats = investimento_agrupado.groupby('Período')['Investimento_por_Aluno'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(2)

print(f"\nEstatísticas do Investimento por Aluno por Período:")
print(period_stats)

# Top 5 universidades com maior investimento por aluno em cada período
print(f"\nTOP 5 UNIVERSIDADES COM MAIOR INVESTIMENTO POR ALUNO:")
for periodo in ['PRÉ-PANDEMIA', 'PANDEMIA', 'PÓS-PANDEMIA']:
    periodo_data = investimento_agrupado[investimento_agrupado['Período'] == periodo]
    top5 = periodo_data.nlargest(5, 'Investimento_por_Aluno')[['Unidade Orçamentária', 'Ano Lançamento', 'Investimento_por_Aluno']]
    print(f"\n{periodo}:")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {row['Unidade Orçamentária']} ({row['Ano Lançamento']}): R$ {row['Investimento_por_Aluno']:,.2f}")

# 8. ANÁLISE ESPECÍFICA DOS OUTLIERS

print(f"\n8. ANÁLISE DETALHADA DOS OUTLIERS")

# Para cada outlier, analisar seu comportamento
print(f"\nCOMPORTAMENTO DETALHADO DE CADA OUTLIER:")
for i, university in enumerate(outliers_df.index, 1):
    uni_data = investimento_agrupado[investimento_agrupado['Unidade Orçamentária'] == university]
    print(f"\n--- {i}. {university} ---")
    print(f"Classificação: OUTLIER")
    
    # Calcular variação durante a pandemia
    pre_pandemic = uni_data[uni_data['Período'] == 'PRÉ-PANDEMIA']['Investimento_por_Aluno'].mean()
    pandemic = uni_data[uni_data['Período'] == 'PANDEMIA']['Investimento_por_Aluno'].mean()
    post_pandemic = uni_data[uni_data['Período'] == 'PÓS-PANDEMIA']['Investimento_por_Aluno'].mean()
    
    if not np.isnan(pre_pandemic) and not np.isnan(pandemic) and pre_pandemic > 0:
        variation_pandemic = ((pandemic - pre_pandemic) / pre_pandemic) * 100
        print(f"Variação PRÉ-PANDEMIA → PANDEMIA: {variation_pandemic:+.2f}%")
    
    if not np.isnan(pandemic) and not np.isnan(post_pandemic) and pandemic > 0:
        variation_post = ((post_pandemic - pandemic) / pandemic) * 100
        print(f"Variação PANDEMIA → PÓS-PANDEMIA: {variation_post:+.2f}%")
    
    # Mostrar investimento por ano
    print(f"Investimento por ano:")
    for _, row in uni_data.iterrows():
        print(f"  {row['Ano Lançamento']}: R$ {row['Investimento_por_Aluno']:,.2f}")

# Criar DataFrame com resultados completos
resultados_completos = pivot_table.reset_index()
resultados_completos.to_csv('resultados_outliers_universidades.csv', index=False, encoding='utf-8-sig')

# Fechar o arquivo de log e restaurar o stdout
log_file.close()
sys.stdout = original_stdout
