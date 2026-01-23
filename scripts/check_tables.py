#!/usr/bin/env python3
"""Check if dubbing studio tables exist in Supabase."""

import sys
from pathlib import Path

# Add parent directory to path before importing local modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from yt_translate.storage.db import get_supabase_client  # noqa: E402

def check_tables():
    """Check if dub_projects, dub_tracks, dub_segments tables exist."""
    client = get_supabase_client()

    tables = ["dub_projects", "dub_tracks", "dub_segments"]
    results = {}

    for table in tables:
        try:
            # Try to query the table - if it doesn't exist, we'll get an error
            result = client.table(table).select("id").limit(1).execute()
            results[table] = {"exists": True, "count": len(result.data)}
            print(f"✓ {table} exists")
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "relation" in error_msg:
                results[table] = {"exists": False, "error": "Table does not exist"}
                print(f"✗ {table} does not exist")
            else:
                results[table] = {"exists": False, "error": str(e)}
                print(f"? {table} error: {e}")

    return results

if __name__ == "__main__":
    print("Checking dubbing studio tables in Supabase...\n")
    results = check_tables()

    all_exist = all(r.get("exists", False) for r in results.values())
    print(f"\nAll tables exist: {all_exist}")
    sys.exit(0 if all_exist else 1)
