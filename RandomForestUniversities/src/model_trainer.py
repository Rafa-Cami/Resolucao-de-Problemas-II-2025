# src/model_trainer.py
"""
Modulo para treinamento, avaliacao e persistencia do modelo.
"""
import logging
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import settings

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Encapsula o pipeline de treinamento, avaliacao e salvamento do modelo.
    """
    def __init__(self, random_state: int = settings.RANDOM_STATE):
        self.random_state = random_state
        self.model: RandomForestRegressor = None
        self.feature_importance: pd.DataFrame = None

    def train(self, X: pd.DataFrame, y: pd.Series, optimize: bool) -> Dict[str, float]:
        """
        Treina, avalia e armazena um modelo RandomForestRegressor.
        """
        logger.info("Iniciando treinamento do modelo Random Forest.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.TEST_SIZE, random_state=self.random_state
        )

        if optimize and len(X_train) > 50:
            self._tune_and_fit(X_train, y_train)
        else:
            logger.info("Usando parametros padrao do Random Forest.")
            self.model = RandomForestRegressor(**settings.DEFAULT_RF_PARAMS)
            self.model.fit(X_train, y_train)

        metrics = self._evaluate(X_test, y_test)
        self._log_metrics(metrics)
        self._capture_feature_importance(X.columns)

        return metrics

    def _tune_and_fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Otimiza hiperparametros com GridSearchCV e treina o melhor modelo."""
        logger.info("Otimizando hiperparametros com GridSearchCV...")
        rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(rf, settings.PARAM_GRID, cv=5, scoring='r2', n_jobs=-1, verbose=2)  # <--- ALTERADO
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logger.info(f"Melhores hiperparametros encontrados: {grid_search.best_params_}")

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Avalia o modelo no conjunto de teste e retorna as metricas."""
        y_pred = self.model.predict(X_test)

        y_test_safe = y_test.copy().replace(0, 1e-6)

        return {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100
        }

    def _log_metrics(self, metrics: Dict[str, float]):
        """Exibe as metricas de performance no log."""
        logger.info("[PERFORMANCE DO MODELO]")
        logger.info(f"R Score: {metrics['r2']:.4f}")
        logger.info(f"RMSE (Erro medio por aluno): R$ {metrics['rmse']:.2f}")
        logger.info(f"MAE (Erro absoluto medio por aluno): R$ {metrics['mae']:.2f}")
        logger.info(f"MAPE (Erro percentual medio): {metrics['mape']:.2f}%")

    def _capture_feature_importance(self, feature_names: List[str]):
        """Captura e loga a importancia das features do modelo treinado."""
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        logger.info("[TOP 10 FEATURES MAIS IMPORTANTES]")
        for _, row in self.feature_importance.head(10).iterrows():
            logger.info(f"- {row['feature']}: {row['importance']:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicoes com o modelo treinado.
        """
        if self.model is None:
            raise RuntimeError("O modelo deve ser treinado antes de fazer predicoes.")
        return self.model.predict(X)

    def save_model(self, file_path: str, artifacts: Dict[str, Any]):
        """
        Salva o modelo treinado e artefatos adicionais em um arquivo.
        """
        if self.model is None:
            logger.error("Nenhum modelo treinado para salvar.")
            return

        model_data = {'model': self.model, **artifacts}
        joblib.dump(model_data, file_path)
        logger.info(f"Modelo e artefatos salvos em: {file_path}")
