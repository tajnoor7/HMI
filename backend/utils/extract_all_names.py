from entity_extraction import extract_actors
import sqlite3

DB_PATH = "../db/emails.db"
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT body FROM emails LIMIT 10")  # adjust as needed

all_actors = set()

for row in cur.fetchall():
    text = row['body']
    names = extract_actors(text)
    for name in names:
        all_actors.add(name)

conn.close()

# Save to file or print
with open("actor_name_dump.txt", "w") as f:
    for name in sorted(all_actors):
        f.write(name + "\n")

print(f"Extracted {len(all_actors)} unique actor names.")
