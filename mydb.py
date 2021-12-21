import sqlite3

conn = sqlite3.connect('weight.db')
cur = conn.cursor()

cur.execute("CREATE TABLE IF NOT EXISTS weight (date DATE, time TIME, weight INT)")

cur.execute("INSERT INTO weight("12.12.2021", "0:24", 59)")

conn.commit()
conn.close()

