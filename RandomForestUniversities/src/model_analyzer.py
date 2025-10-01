# src/model_analyzer.py
"""
Modulo para analise de resultados e geracao de visualizacoes.
"""
import logging
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import settings

logger = logging.getLogger(__name__)


class ModelAnalyzer:
    """
    Realiza analises pos-predicao para entender os resultados do modelo.
    """
    def __init__(self, original_df: pd.DataFrame, X: pd.DataFrame, y_pred: np.ndarray):
        """
        Construtor da classe ModelAnalyzer.
        """
        self.df_analysis = original_df.loc[X.index].copy()
        self.df_analysis['Predicted'] = y_pred
        self.df_analysis['Error'] = self.df_analysis['Predicted'] - self.df_analysis[settings.TARGET_COL]
        self.df_analysis['Percentage_Error'] = (self.df_analysis['Error'] / self.df_analysis[settings.TARGET_COL].replace(0, 1e-6)) * 100

    def get_analysis_by_unit(self) -> pd.DataFrame:
        """
        Agrega os resultados de predicao por Unidade Orcamentaria.
        """
        if 'Unidade Orcamentaria' not in self.df_analysis.columns:
            return None

        unit_analysis = self.df_analysis.groupby('Unidade Orcamentaria').agg({
            settings.TARGET_COL: 'mean', 'Predicted': 'mean',
            'Error': 'mean', 'Percentage_Error': 'mean',
        }).round(2).rename(columns={
            settings.TARGET_COL: 'Real_Avg_Investment',
            'Predicted': 'Predicted_Avg_Investment',
            'Error': 'Avg_Error', 'Percentage_Error': 'Avg_Percentage_Error'
        })
        unit_analysis['Precision'] = 100 - abs(unit_analysis['Avg_Percentage_Error'])
        return unit_analysis.sort_values(by='Real_Avg_Investment', ascending=False)

    def get_analysis_by_year(self) -> pd.DataFrame:
        """
        Agrega os resultados de predicao por Ano de Lancamento.
        """
        if 'Ano_Lancamento' not in self.df_analysis.columns:
            return None

        return self.df_analysis.groupby('Ano_Lancamento').agg({
            settings.TARGET_COL: 'mean', 'Predicted': 'mean'
        }).round(2)


class Visualizer:
    """Classe estatica com metodos para gerar visualizacoes do modelo."""

    @staticmethod
    def generate_summary_plot(df_analysis: pd.DataFrame, feature_importance: pd.DataFrame, metrics: Dict[str, float]):
        """
        Gera um dashboard com os principais graficos de analise do modelo.
        """
        logger.info("Gerando dashboard de visualizacao do modelo.")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle('Analise do Modelo Random Forest - Investimento por Aluno', fontsize=20, weight='bold')

            Visualizer._plot_feature_importance(axes[0, 0], feature_importance)
            Visualizer._plot_real_vs_predicted(axes[0, 1], df_analysis, metrics['r2'])
            Visualizer._plot_error_distribution(axes[1, 0], df_analysis['Error'])
            Visualizer._plot_temporal_evolution(axes[1, 1], df_analysis)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(settings.PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Graficos salvos em: {settings.PLOT_SAVE_PATH}")

        except Exception as e:
            logger.error(f"Falha ao gerar graficos: {e}")

    @staticmethod
    def _plot_feature_importance(ax, feature_importance: pd.DataFrame):
        """Plota o grafico de importancia das features."""
        top_features = feature_importance.head(15).sort_values('importance', ascending=True)
        ax.barh(top_features['feature'], top_features['importance'], color='skyblue')
        ax.set_title('Top 15 Features Mais Importantes', fontsize=14)
        ax.set_xlabel('Importancia')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    @staticmethod
    def _plot_real_vs_predicted(ax, df_analysis: pd.DataFrame, r2: float):
        """Plota o grafico de dispersao de valores reais vs. preditos."""
        sns.scatterplot(x=settings.TARGET_COL, y='Predicted', data=df_analysis, alpha=0.5, ax=ax, s=50)
        min_val = min(df_analysis[settings.TARGET_COL].min(), df_analysis['Predicted'].min())
        max_val = max(df_analysis[settings.TARGET_COL].max(), df_analysis['Predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax.set_title('Valores Reais vs. Preditos', fontsize=14)
        ax.set_xlabel('Investimento Real por Aluno (R$)')
        ax.set_ylabel('Investimento Predito por Aluno (R$)')
        ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$', transform=ax.transAxes, fontsize=12, va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.grid(linestyle='--', alpha=0.6)

    @staticmethod
    def _plot_error_distribution(ax, errors: pd.Series):
        """Plota o histograma da distribuicao dos erros de predicao."""
        sns.histplot(errors, kde=True, bins=50, ax=ax, color='coral')
        ax.axvline(0, color='black', linestyle='--', lw=2)
        ax.set_title('Distribuicao dos Erros de Predicao', fontsize=14)
        ax.set_xlabel('Erro (Predito - Real)')
        ax.set_ylabel('Frequencia')
        ax.grid(linestyle='--', alpha=0.6)

    @staticmethod
    def _plot_temporal_evolution(ax, df_analysis: pd.DataFrame):
        """Plota a evolucao temporal do investimento medio (real vs. predito)."""
        if 'Ano_Lancamento' in df_analysis.columns:
            year_analysis = df_analysis.groupby('Ano_Lancamento').agg({
                settings.TARGET_COL: 'mean', 'Predicted': 'mean'
            })
            ax.plot(year_analysis.index, year_analysis[settings.TARGET_COL], 'o-', label='Real', lw=2)
            ax.plot(year_analysis.index, year_analysis['Predicted'], 's--', label='Predito', lw=2)
            ax.set_title('Evolucao Temporal (Media Anual)', fontsize=14)
            ax.set_xlabel('Ano')
            ax.set_ylabel('Investimento Medio por Aluno (R$)')
            ax.legend()
            ax.grid(linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
