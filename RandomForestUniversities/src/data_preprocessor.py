# src/data_preprocessor.py
"""
Modulo para pre-processamento e limpeza de dados.
"""
import logging
import pandas as pd
from config import settings

logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza basica e o pre-processamento do DataFrame.
    """
    logger.info("Iniciando pre-processamento basico dos dados.")
    df_clean = df.copy()

    initial_rows = len(df_clean)
    subset_cols = [settings.COL_MOVIMENTO_LIQUIDO, settings.COL_QTD_ALUNOS]
    df_clean.dropna(subset=subset_cols, inplace=True)
    rows_removed = initial_rows - len(df_clean)
    if rows_removed > 0:
        logger.info(f"{rows_removed} linhas removidas por valores nulos em colunas chave.")

    invalid_student_count = (df_clean[settings.COL_QTD_ALUNOS] <= 0).sum()
    if invalid_student_count > 0:
        logger.warning(f"Encontrados {invalid_student_count} registros com quantidade de alunos <= 0. Corrigindo para 1.")
        df_clean[settings.COL_QTD_ALUNOS] = df_clean[settings.COL_QTD_ALUNOS].clip(lower=1)

    logger.info(f"Pre-processamento concluido. Shape final: {df_clean.shape}")
    return df_clean
