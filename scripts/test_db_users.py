from database.db_manager import DatabaseManager

db = DatabaseManager()
users = db.fetch_all_users()

print("Users in DB:")
for u in users:
    print(u["name"])
