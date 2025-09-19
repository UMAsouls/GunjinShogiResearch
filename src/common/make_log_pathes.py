from src.const import LOG_DIR, LOG_PIECES1, LOG_PIECES2, LOG_STEPS

def make_log_pathes(name:str) -> tuple[str, str, str]:
    """ログのパス作成用関数

    Args:
        name (str): ログの名前

    Returns:
        tuple[str, str, str]: PIECES1_PATH, PIECES2_PATH, STEPS_PATH
    """
    
    return f"{LOG_DIR}/{name}/{LOG_PIECES1}", f"{LOG_DIR}/{name}/{LOG_PIECES2}", f"{LOG_DIR}/{name}/{LOG_STEPS}"