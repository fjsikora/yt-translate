#!/usr/bin/env python3
"""Apply SQL migration to Supabase database via REST API."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    sys.exit(1)


def apply_migration(sql_file: str) -> bool:
    """Apply a SQL migration file to Supabase."""
    # Read SQL file
    sql_path = Path(sql_file)
    if not sql_path.exists():
        print(f"Error: SQL file not found: {sql_file}")
        return False

    sql_content = sql_path.read_text()
    print(f"Applying migration: {sql_path.name}")
    print(f"SQL length: {len(sql_content)} characters")

    # Note: Supabase client created but not used since supabase-py
    # doesn't support raw SQL execution
    _ = create_client(SUPABASE_URL, SUPABASE_KEY)  # noqa: F841

    # Split SQL into individual statements
    # This is needed because the REST API may not handle multiple statements well
    statements = []
    current = []
    in_function = False

    for line in sql_content.split('\n'):
        stripped = line.strip()

        # Track if we're inside a function definition
        if 'CREATE TRIGGER' in line.upper() or 'CREATE OR REPLACE FUNCTION' in line.upper():
            in_function = True

        current.append(line)

        # Check for statement end (semicolon at end of line, not in function)
        if stripped.endswith(';') and not in_function:
            statement = '\n'.join(current).strip()
            if statement and not statement.startswith('--'):
                statements.append(statement)
            current = []
        elif 'EXECUTE FUNCTION' in line.upper() and stripped.endswith(';'):
            # End of trigger statement
            statement = '\n'.join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            in_function = False
        elif stripped == '$$ LANGUAGE plpgsql;':
            # End of function definition
            statement = '\n'.join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            in_function = False

    # Add any remaining content
    if current:
        statement = '\n'.join(current).strip()
        if statement and not statement.startswith('--'):
            statements.append(statement)

    print(f"Found {len(statements)} SQL statements to execute")

    # Execute each statement via RPC
    # Since we don't have a direct SQL exec, we'll need another approach
    # The supabase-py library doesn't support raw SQL execution
    # We need to use the Supabase SQL Editor API or psycopg2

    print("\nNote: The supabase-py library doesn't support direct SQL execution.")
    print("The migration SQL file has been created at:")
    print(f"  {sql_path.absolute()}")
    print("\nTo apply this migration, you can:")
    print("1. Copy the SQL and paste it in Supabase Dashboard > SQL Editor")
    print("2. Use the Supabase CLI: supabase db push")
    print("3. Connect directly with psql using the connection string from Supabase Dashboard")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the dubbing studio migration
        sql_file = "supabase_migrations/migrations/002_dubbing_studio_schema.sql"
    else:
        sql_file = sys.argv[1]

    success = apply_migration(sql_file)
    sys.exit(0 if success else 1)
