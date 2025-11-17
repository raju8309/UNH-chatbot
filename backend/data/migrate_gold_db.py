#!/usr/bin/env python3
import sys
from pathlib import Path

# add backend to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.gold_db import GoldSetDB

def main():
    print("=" * 60)
    print("Gold Set Database Migration")
    print("=" * 60)
    print()
    
    # paths
    jsonl_path = ROOT.parent / "automation_testing" / "gold.jsonl"
    db_path = ROOT / "data" / "gold_set.db"
    
    # check if JSONL exists
    if not jsonl_path.exists():
        print(f"ERROR: Gold JSONL file not found: {jsonl_path}")
        sys.exit(1)
    
    print(f"Source: {jsonl_path}")
    print(f"Target: {db_path}")
    print()
    
    # check if database already exists
    if db_path.exists():
        response = input("Database already exists. Overwrite? (yes/no): ").strip().lower()
        if response != 'yes':
            print("Migration cancelled.")
            sys.exit(0)
        overwrite = True
    else:
        overwrite = False
    
    # create database and import
    print("\nCreating database and importing data...")
    with GoldSetDB(str(db_path)) as db:
        count = db.import_from_jsonl(str(jsonl_path), overwrite=overwrite)
        
        print("\n" + "=" * 60)
        print("Migration Statistics")
        print("=" * 60)
        
        stats = db.get_statistics()
        print(f"Total Questions:      {stats['total_questions']}")
        print(f"Total Nuggets:        {stats['total_nuggets']}")
        print(f"Total Gold Passages:  {stats['total_gold_passages']}")
        print(f"Has Embeddings:       {stats['has_embeddings']}")
        
        print(f"\nCategories ({len(stats['categories'])}):")
        for category, count in sorted(stats['categories'].items()):
            print(f"  {category:30s} {count:4d}")
    
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. The database is ready to use")
    print("2. Embeddings will be computed on first use")
    print("3. You can query the database with:")
    print(f"   python {db_path.parent}/gold_db.py --db {db_path} --stats")
    print()

if __name__ == "__main__":
    main()