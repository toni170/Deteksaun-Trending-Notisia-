"""
Fatin (folder) ba fluxu servisu foun: lote koleta, ficheiru karga, dadus bersih, modelu.
Se `config["paths"]` la iha xave `batches`/`uploads`, uza folder ba "data" iha tutun `cleaned`.
"""
import os


def workflow_dirs(config: dict) -> dict:
    paths = config.get("paths", {})
    cleaned = paths.get("cleaned") or os.path.join(os.getcwd(), "data", "cleaned")
    data_root = os.path.dirname(cleaned.rstrip(os.sep))
    dirs = {
        "batches": paths.get("batches") or os.path.join(data_root, "batches"),
        "uploads": paths.get("uploads") or os.path.join(data_root, "uploads"),
        "cleaned": cleaned,
        "models": paths.get("models") or os.path.join(data_root, "models"),
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs
