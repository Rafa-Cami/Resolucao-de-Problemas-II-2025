# src/utils.py
"""
Modulo contendo funcoes utilitarias para o projeto.
"""
import logging
import logging.config
from config import settings

def setup_logging():
    """
    Configura o sistema de logging para console e arquivo.
    """
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': { 'structured': { 'format': '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s'}},
        'handlers': {
            'file': { 'level': 'INFO', 'class': 'logging.FileHandler', 'filename': settings.LOG_FILE_PATH, 'formatter': 'structured'},
            'console': { 'level': 'INFO', 'class': 'logging.StreamHandler', 'formatter': 'structured'}
        },
        'root': { 'level': 'INFO', 'handlers': ['file', 'console']}
    })
    return logging.getLogger(__name__)
