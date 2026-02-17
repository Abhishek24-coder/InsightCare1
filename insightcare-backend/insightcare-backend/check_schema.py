import psycopg

conn = psycopg.connect("postgresql://postgres:motog85@localhost:5432/insightcare")
cur = conn.cursor()
cur.execute(
    """
    SELECT column_name, data_type, udt_name 
    FROM information_schema.columns 
    WHERE table_name='diagnoses' 
    ORDER BY ordinal_position
"""
)

print("Diagnoses table schema:")
for row in cur.fetchall():
    print(f"  {row[0]:20} {row[1]:20} ({row[2]})")

conn.close()
