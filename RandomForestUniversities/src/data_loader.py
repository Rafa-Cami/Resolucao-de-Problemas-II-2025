# src/data_loader.py
"""
Modulo para carregamento e validacao inicial de dados.
"""
import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

def load_data(file_path: str, required_columns: List[str]) -> pd.DataFrame:
    """
    Carrega dados de um arquivo e valida a presenca de colunas essenciais.
    """
    logger.info(f"Carregando dados de '{file_path}'")
    try:
        df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas essenciais faltantes no arquivo: {missing_cols}")

        return df
    except FileNotFoundError:
        logger.error(f"Arquivo nao encontrado em: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar os dados: {e}")
        raise
