"""
Utility bersama: memuat konfigurasi (config/settings.yaml) dan setup logging
untuk seluruh modul Mining Notísia Online no Deteksaun Topiku.
"""
import os
import logging
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "settings.yaml")


def load_config(config_path: str = None) -> dict:
    """Muat settings.yaml. Path pada 'paths' diresolusi relatif ke root project."""
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, rel_path in config["paths"].items():
        config["paths"][key] = os.path.join(PROJECT_ROOT, rel_path)

    return config


def setup_logging(logs_dir: str, name: str = "news_topic_mining") -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{name}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
