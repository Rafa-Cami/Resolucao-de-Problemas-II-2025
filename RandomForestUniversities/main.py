# main.py
"""
Script principal para execucao do pipeline de machine learning.
"""
import warnings
from config import settings
from src.utils import setup_logging
from src.data_loader import load_data
from src.data_preprocessor import preprocess_data
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_analyzer import ModelAnalyzer, Visualizer
import pandas as pd

warnings.filterwarnings('ignore')
logger = setup_logging()

def run_pipeline():
    """
    Orquestra e executa o pipeline completo de machine learning.
    """
    try:
        required_cols = [settings.COL_ANO, settings.COL_MOVIMENTO_LIQUIDO, settings.COL_QTD_ALUNOS]
        df_raw = load_data(settings.DATA_FILE_PATH, required_cols)

        df_clean = preprocess_data(df_raw)

        feature_engineer = FeatureEngineer()
        df_featured = feature_engineer.create_features(df_clean)

        feature_names = feature_engineer.get_feature_names(df_featured)
        X = df_featured[feature_names].dropna()
        y = df_featured.loc[X.index, settings.TARGET_COL]

        if len(X) < 50:
            logger.error("Dados insuficientes para treinamento apos pre-processamento.")
            return

        trainer = ModelTrainer()
        metrics = trainer.train(X, y, optimize=settings.OPTIMIZE_HYPERPARAMS)

        y_pred = trainer.predict(X)
        analyzer = ModelAnalyzer(df_featured, X, y_pred)
        unit_analysis = analyzer.get_analysis_by_unit()

        Visualizer.generate_summary_plot(analyzer.df_analysis, trainer.feature_importance, metrics)

        artifacts = {
            'label_encoders': feature_engineer.label_encoders,
            'feature_names': feature_names,
            'feature_importance': trainer.feature_importance
        }
        trainer.save_model(settings.MODEL_SAVE_PATH, artifacts)

        generate_final_report(unit_analysis)

    except Exception as e:
        logger.critical(f"Ocorreu um erro critico no pipeline: {e}", exc_info=True)

def generate_final_report(unit_analysis: pd.DataFrame):
    """
    Gera e loga o relatorio final com os principais insights.
    """
    logger.info("\n" + "="*80 + "\nRELATORIO FINAL\n" + "="*80)
    if unit_analysis is not None:
        logger.info("[ANALISE POR UNIDADE ORCAMENTARIA - TOP 10 MAIOR INVESTIMENTO REAL]")
        top_units = unit_analysis.head(10)
        for unit, data in top_units.iterrows():
            logger.info(
                f"- {str(unit)[:40]:<40s}: "
                f"Real=R${data['Real_Avg_Investment']:>10,.2f}, "
                f"Predito=R${data['Predicted_Avg_Investment']:>10,.2f}, "
                f"Precisao={data['Precision']:.1f}%"
            )

    logger.info("\nPipeline executado com sucesso!")


if __name__ == "__main__":
    run_pipeline()
