import sqlite3


conn = sqlite3.connect('mydatabase.db')
cur = conn.cursor()

query = "CREATE TABLE IF NOT EXISTS book (author TEXT, year INT)"
cur.execute(query)

cur.execute("INSERT INTO book VALUES('Pushkin', 1799)")

book1 = ("Esenin", 1890)
cur.execute("INSERT INTO book VALUES(?, ?)", book1)

cur.execute('UPDATE book SET year = 1895 where author = "Esenin"')

cur.execute("SELECT * FROM book")
# cur.execute("SELECT author FROM book")
# cur.execute("SELECT * FROM book WHERE year=1799")
rows = cur.fetchall()
# print(rows)
for row in rows:
    print(row)

conn.commit()
conn.close()