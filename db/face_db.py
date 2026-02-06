# db/face_db.py
import sqlite3
import numpy as np
from collections import defaultdict
from pathlib import Path
import shutil


class FaceDatabase:
    def __init__(self, db_path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # If the DB path moved (e.g. to enrolled_users/faces.db) and an older
        # faces.db exists in the project root, migrate it automatically.
        if not db_path.exists():
            legacy_path = db_path.parent.parent / db_path.name
            if legacy_path.exists() and legacy_path != db_path:
                try:
                    legacy_path.replace(db_path)
                except OSError:
                    # Fall back to copy if a rename/move isn't possible.
                    shutil.copy2(legacy_path, db_path)
        self.conn = sqlite3.connect(str(db_path))
        self._create_table()

    def _create_table(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_username ON embeddings(username)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_templates (
                username TEXT PRIMARY KEY,
                embedding_sum BLOB NOT NULL,
                count INTEGER NOT NULL
            )
        """)
        # Backfill users table from any existing embeddings (safe if empty).
        cur.execute("""
            INSERT OR IGNORE INTO users (username)
            SELECT DISTINCT username FROM embeddings
        """)
        self.conn.commit()

        # Backfill templates only once (when table is empty).
        cur.execute("SELECT COUNT(*) FROM user_templates")
        row = cur.fetchone()
        if row and int(row[0]) == 0:
            cur.execute("SELECT username, embedding FROM embeddings")
            rows = cur.fetchall()
            sums = defaultdict(lambda: (None, 0))
            for username, blob in rows:
                emb = np.frombuffer(blob, dtype=np.float32)
                cur_sum, cur_count = sums[username]
                if cur_sum is None:
                    sums[username] = (emb.copy(), 1)
                else:
                    sums[username] = (cur_sum + emb, cur_count + 1)

            for username, (emb_sum, count) in sums.items():
                if emb_sum is None or count <= 0:
                    continue
                cur.execute(
                    "INSERT OR REPLACE INTO user_templates (username, embedding_sum, count) VALUES (?, ?, ?)",
                    (username, emb_sum.astype(np.float32).tobytes(), int(count)),
                )
            self.conn.commit()

    def add_embedding(self, username, embedding: np.ndarray):
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO users (username) VALUES (?)",
            (username,)
        )
        cur.execute(
            "INSERT INTO embeddings (username, embedding) VALUES (?, ?)",
            (username, emb.tobytes())
        )

        cur.execute("SELECT embedding_sum, count FROM user_templates WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            sum_blob, count = row
            emb_sum = np.frombuffer(sum_blob, dtype=np.float32)
            count = int(count)
            new_sum = emb_sum + emb
            new_count = count + 1
            cur.execute(
                "UPDATE user_templates SET embedding_sum = ?, count = ? WHERE username = ?",
                (new_sum.astype(np.float32).tobytes(), new_count, username),
            )
        else:
            cur.execute(
                "INSERT INTO user_templates (username, embedding_sum, count) VALUES (?, ?, ?)",
                (username, emb.tobytes(), 1),
            )
        self.conn.commit()

    def load_all_users(self):
        cur = self.conn.cursor()
        cur.execute("SELECT username, embedding FROM embeddings")

        users = defaultdict(list)
        for name, blob in cur.fetchall():
            emb = np.frombuffer(blob, dtype=np.float32)
            users[name].append(emb)

        return dict(users)

    def load_templates(self):
        """
        Returns a dict: {username: normalized_mean_embedding}
        """
        cur = self.conn.cursor()
        cur.execute("SELECT username, embedding_sum, count FROM user_templates")

        templates = {}
        for username, sum_blob, count in cur.fetchall():
            emb_sum = np.frombuffer(sum_blob, dtype=np.float32)
            count = int(count)
            if count <= 0:
                continue
            mean = emb_sum / float(count)
            n = np.linalg.norm(mean)
            if n == 0:
                continue
            templates[username] = (mean / n).astype(np.float32)

        return templates

    def list_users(self):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT username, COUNT(*) AS n
            FROM embeddings
            GROUP BY username
            ORDER BY username
        """)
        return list(cur.fetchall())

    def count_embeddings(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def count_user_embeddings(self, username):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM embeddings WHERE username = ?", (username,))
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def delete_user(self, username):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM embeddings WHERE username = ?", (username,))
        emb_deleted = int(cur.rowcount)
        cur.execute("DELETE FROM user_templates WHERE username = ?", (username,))
        cur.execute("DELETE FROM users WHERE username = ?", (username,))
        self.conn.commit()
        return emb_deleted

    def delete_users_like(self, pattern):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM embeddings WHERE username LIKE ?", (pattern,))
        emb_deleted = int(cur.rowcount)
        cur.execute("DELETE FROM user_templates WHERE username LIKE ?", (pattern,))
        cur.execute("DELETE FROM users WHERE username LIKE ?", (pattern,))
        self.conn.commit()
        return emb_deleted

    def close(self):
        self.conn.close()
