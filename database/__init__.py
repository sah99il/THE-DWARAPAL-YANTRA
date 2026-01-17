import os

class DatabaseManager:
    def __init__(self, db_path="database/dwarapal.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._create_table()
