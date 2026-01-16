import sqlite3
import numpy as np
import io

class DatabaseManager:
    def __init__(self, db_path="database/dwarapal.db"):
        # Path to SQLite database file
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        # Create users table if it does not exist
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            ''')

    def add_user(self, name, embedding):
        # Store user name and face embedding
        out = io.BytesIO()
        np.save(out, embedding)
        out.seek(0)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO users (name, embedding) VALUES (?, ?)',
                (name, out.read())
            )

    def fetch_all_users(self):
        # Load all users and their embeddings from DB
        users = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, name, embedding FROM users')
            for row in cursor:
                out = io.BytesIO(row[2])
                out.seek(0)
                emb = np.load(out)
                users.append({
                    "id": row[0],
                    "name": row[1],
                    "embedding": emb
                })
        return users

    def delete_all(self):
        # Remove all records from users table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM users')
