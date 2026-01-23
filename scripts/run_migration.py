#!/usr/bin/env python3
"""Execute SQL migration directly on Supabase PostgreSQL database."""

import os
import sys
from pathlib import Path

import psycopg2  # noqa: E402 - imported here to allow .env loading first

# Load environment from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Supabase database connection details
# Format: postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
SUPABASE_PROJECT_REF = "your-project-ref"
# Get the database password from Supabase dashboard (Settings > Database > Connection string)
# For now, we'll use the service role key (which works for some operations)
# But typically you need the actual DB password

# The actual password for direct DB connection
# This is different from the service role key
DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")

if not DB_PASSWORD:
    # Try to prompt for it
    print("SUPABASE_DB_PASSWORD not set in environment.")
    print("You can find this in Supabase Dashboard > Settings > Database > Connection string")
    print("Or set SUPABASE_DB_PASSWORD environment variable")
    # For now, let's try the pooler connection which might work with service key
    pass

def execute_migration(sql_file: str) -> bool:
    """Execute a SQL migration file."""
    sql_path = Path(sql_file)
    if not sql_path.exists():
        print(f"Error: SQL file not found: {sql_file}")
        return False

    sql_content = sql_path.read_text()
    print(f"Migration file: {sql_path.name}")
    print(f"SQL size: {len(sql_content)} bytes")

    # Connection string - using transaction pooler for better compatibility
    # The pooler connection uses port 6543 instead of 5432
    conn_string = f"postgresql://postgres.{SUPABASE_PROJECT_REF}:{DB_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"

    try:
        print("\nConnecting to Supabase PostgreSQL...")
        conn = psycopg2.connect(conn_string)
        conn.autocommit = True  # Each statement runs in its own transaction
        cursor = conn.cursor()

        print("Executing migration...")
        cursor.execute(sql_content)

        print("Migration executed successfully!")

        # Verify tables were created
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'dub_%'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        print(f"\nCreated tables: {[t[0] for t in tables]}")

        cursor.close()
        conn.close()
        return True

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sql_file = Path(__file__).parent.parent / "supabase_migrations/migrations/002_dubbing_studio_schema.sql"
    else:
        sql_file = sys.argv[1]

    success = execute_migration(str(sql_file))
    sys.exit(0 if success else 1)
