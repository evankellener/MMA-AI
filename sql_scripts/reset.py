import sqlite3
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root (parent of sql_scripts)
project_root = os.path.dirname(script_dir)

# Connect to database in project root
db_path = os.path.join(project_root, 'sqlite_scrapper.db')
conn = sqlite3.connect(db_path)

# Read SQL file from same directory as this script
sql_path = os.path.join(script_dir, 'reset.sql')
with open(sql_path, 'r') as f:
    conn.executescript(f.read())
conn.commit()
conn.close()