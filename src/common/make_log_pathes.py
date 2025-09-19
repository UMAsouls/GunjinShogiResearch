from src.const import LOG_DIR, LOG_FIRST_BOARD, LOG_STEPS

def make_log_pathes(name:str) -> tuple[str, str]:
    """ログのパス作成用関数

    Args:
        name (str): ログの名前

    Returns:
        tuple[str, str]: FIRST_BOARD_PATH, STEPS_PATH
    """
    
    return f"{LOG_DIR}/{name}/{LOG_FIRST_BOARD}", f"{LOG_DIR}/{name}/{LOG_STEPS}"