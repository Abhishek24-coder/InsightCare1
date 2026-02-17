import psycopg

conn = psycopg.connect("postgresql://postgres:motog85@localhost:5432/insightcare")
cur = conn.cursor()
cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname='public'")
tables = cur.fetchall()
print("📊 Database Tables:")
for t in tables:
    print(f"  ✅ {t[0]}")
conn.close()
print("\n✅ Database is ready!")
