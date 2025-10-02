# src/feature_engineer.py
"""
Modulo para engenharia de features.
"""
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Encapsula toda a logica de criacao de features para o modelo.

    Esta classe e responsavel por criar a variavel alvo, novas features
    numericas (log, raiz quadrada), codificar variaveis categoricas
    e criar features de interacao.

    :ivar label_encoders: Dicionario que armazena os objetos LabelEncoder
                          ajustados para cada coluna categorica.
    :vartype label_encoders: Dict[str, LabelEncoder]
    """
    def __init__(self):
        """Construtor da classe FeatureEngineer."""
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orquestra a criacao de todas as features para o modelo.

        :param df: DataFrame pre-processado.
        :type df: pd.DataFrame
        :return: DataFrame com as novas features.
        :rtype: pd.DataFrame
        """
        logger.info("Iniciando engenharia de features.")
        df_fe = df.copy()

        df_fe = self._create_target_variable(df_fe)
        df_fe = self._create_numerical_features(df_fe)
        df_fe = self._encode_categorical_features(df_fe)
        df_fe = self._create_temporal_features(df_fe)
        df_fe = self._create_interaction_features(df_fe)

        logger.info(f"Engenharia de features concluida. Shape final: {df_fe.shape}")
        return df_fe

    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria a variavel alvo 'Investimento_por_Aluno' e trata valores invalidos."""
        df[settings.TARGET_COL] = df[settings.COL_MOVIMENTO_LIQUIDO] / df[settings.COL_QTD_ALUNOS]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        initial_rows = len(df)
        df.dropna(subset=[settings.TARGET_COL], inplace=True)
        rows_removed = initial_rows - len(df)

        if rows_removed > 0:
            logger.info(f"{rows_removed} registros removidos por gerar target invalido (inf, nan).")

        return df

    def _create_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features numericas baseadas na quantidade de alunos."""
        df['Log_Quantidade_Alunos'] = np.log1p(df[settings.COL_QTD_ALUNOS])
        df['Sqrt_Quantidade_Alunos'] = np.sqrt(df[settings.COL_QTD_ALUNOS])
        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Codifica colunas categoricas usando LabelEncoder e agrupa categorias raras."""
        categorical_cols = [col for col in settings.CATEGORICAL_COLS if col in df.columns]

        for col in categorical_cols:
            df[col] = df[col].fillna('Missing')

            counts = df[col].value_counts()
            rare_cats = counts[counts < 5].index
            if len(rare_cats) > 0:
                df[col] = df[col].replace(rare_cats, 'Outros')

            le = LabelEncoder()
            encoded_col_name = f'{col}_encoded'
            df[encoded_col_name] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features baseadas na coluna de ano."""
        if settings.COL_ANO in df.columns:
            df['Ano_Lancamento'] = pd.to_numeric(df[settings.COL_ANO], errors='coerce').astype('Int64')
            df.dropna(subset=['Ano_Lancamento'], inplace=True)

            df['Tendencia_Temporal'] = df['Ano_Lancamento'] - df['Ano_Lancamento'].min()
            df['Ano_Quadratico'] = (df['Ano_Lancamento'] - df['Ano_Lancamento'].mean()) ** 2
        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria features de interacao entre quantidade de alunos e ano."""
        if 'Log_Quantidade_Alunos' in df.columns and 'Ano_Lancamento' in df.columns:
            df['Alunos_Ano_Interaction'] = df[settings.COL_QTD_ALUNOS] * df['Ano_Lancamento']
            df['LogAlunos_Ano_Interaction'] = df['Log_Quantidade_Alunos'] * df['Ano_Lancamento']
        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Retorna a lista final de nomes de features para o modelo.
        """
        base_features = [
            'Ano_Lancamento', settings.COL_QTD_ALUNOS, 'Log_Quantidade_Alunos',
            'Sqrt_Quantidade_Alunos', 'Tendencia_Temporal', 'Ano_Quadratico',
            'Alunos_Ano_Interaction', 'LogAlunos_Ano_Interaction'
        ]

        encoded_features = [col for col in df.columns if col.endswith('_encoded')]

        return [col for col in base_features + encoded_features if col in df.columns]
