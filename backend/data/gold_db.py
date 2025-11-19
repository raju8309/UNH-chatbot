#!/usr/bin/env python3
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np
from datetime import datetime

class GoldSetDB:
    
    def __init__(self, db_path: str = "backend/data/gold_set.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        # enable foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        # main questions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                reference_answer TEXT NOT NULL,
                url TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # nuggets table (one-to-many with questions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nuggets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                nugget TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            )
        """)
        
        # gold passages table (one-to-many with questions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gold_passages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                passage_id TEXT NOT NULL,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            )
        """)
        
        # embeddings table (for semantic search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                question_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id) ON DELETE CASCADE
            )
        """)
        
        # create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON questions(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_lower ON questions(LOWER(query))")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nuggets_qid ON nuggets(question_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_passages_qid ON gold_passages(question_id)")
        
        self.conn.commit()
    
    def import_from_jsonl(self, jsonl_path: str, overwrite: bool = False):
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
        
        cursor = self.conn.cursor()
        
        # clear existing data if overwrite
        if overwrite:
            cursor.execute("DELETE FROM questions")
            cursor.execute("DELETE FROM nuggets")
            cursor.execute("DELETE FROM gold_passages")
            cursor.execute("DELETE FROM embeddings")
            self.conn.commit()
            print("Cleared existing data")
        
        count = 0
        skipped = 0
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    # extract fields
                    qid = entry.get('id', '')
                    query = entry.get('query', '')
                    reference_answer = entry.get('reference_answer', '')
                    url = entry.get('url', '')
                    nuggets = entry.get('nuggets', [])
                    gold_passages = entry.get('gold_passages', [])
                    
                    # extract category from id (e.g., "academic-standards:q0001")
                    category = qid.split(':')[0] if ':' in qid else 'uncategorized'
                    
                    # check if question already exists
                    cursor.execute("SELECT id FROM questions WHERE id = ?", (qid,))
                    if cursor.fetchone():
                        if not overwrite:
                            skipped += 1
                            continue
                    
                    # insert question
                    cursor.execute("""
                        INSERT OR REPLACE INTO questions 
                        (id, query, reference_answer, url, category, updated_at)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (qid, query, reference_answer, url, category))
                    
                    # insert nuggets
                    if nuggets:
                        for nugget in nuggets:
                            if isinstance(nugget, str) and nugget.strip():
                                cursor.execute("""
                                    INSERT INTO nuggets (question_id, nugget)
                                    VALUES (?, ?)
                                """, (qid, nugget.strip()))
                    
                    # insert gold passages
                    if gold_passages:
                        for passage in gold_passages:
                            if isinstance(passage, str) and passage.strip():
                                cursor.execute("""
                                    INSERT INTO gold_passages (question_id, passage_id)
                                    VALUES (?, ?)
                                """, (qid, passage.strip()))
                    
                    count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
        
        self.conn.commit()
        print(f"Imported {count} questions from {jsonl_path}")
        if skipped > 0:
            print(f"Skipped {skipped} existing questions")
        
        return count
    
    def store_embeddings(self, embeddings: np.ndarray, question_ids: List[str], 
                        model_name: str = "all-MiniLM-L6-v2"):
        cursor = self.conn.cursor()
        
        for qid, embedding in zip(question_ids, embeddings):
            # convert numpy array to bytes
            embedding_bytes = embedding.tobytes()
            
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (question_id, embedding, model_name, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (qid, embedding_bytes, model_name))
        
        self.conn.commit()
        print(f"Stored embeddings for {len(question_ids)} questions")
    
    def get_all_questions(self) -> List[Dict]:
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, query, reference_answer, url, category
            FROM questions
            ORDER BY category, id
        """)
        
        questions = []
        for row in cursor.fetchall():
            questions.append({
                'id': row['id'],
                'query': row['query'],
                'reference_answer': row['reference_answer'],
                'url': row['url'],
                'category': row['category']
            })
        
        return questions
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        cursor = self.conn.cursor()
        
        # get question
        cursor.execute("""
            SELECT id, query, reference_answer, url, category
            FROM questions
            WHERE id = ?
        """, (question_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        question = {
            'id': row['id'],
            'query': row['query'],
            'reference_answer': row['reference_answer'],
            'url': row['url'],
            'category': row['category']
        }
        
        # get nuggets
        cursor.execute("""
            SELECT nugget FROM nuggets WHERE question_id = ?
        """, (question_id,))
        question['nuggets'] = [r['nugget'] for r in cursor.fetchall()]
        
        # get gold passages
        cursor.execute("""
            SELECT passage_id FROM gold_passages WHERE question_id = ?
        """, (question_id,))
        question['gold_passages'] = [r['passage_id'] for r in cursor.fetchall()]
        
        return question
    
    def get_embeddings(self) -> tuple:
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT question_id, embedding 
            FROM embeddings
            ORDER BY question_id
        """)
        
        question_ids = []
        embeddings = []
        
        for row in cursor.fetchall():
            question_ids.append(row['question_id'])
            # convert bytes back to numpy array
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            embeddings.append(embedding)
        
        if embeddings:
            embeddings = np.vstack(embeddings)
            return embeddings, question_ids
        
        return None, []
    
    def get_questions_by_category(self, category: str) -> List[Dict]:
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, query, reference_answer, url, category
            FROM questions
            WHERE category = ?
            ORDER BY id
        """, (category,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_categories(self) -> Dict[str, int]:
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT category, COUNT(*) as count
            FROM questions
            GROUP BY category
            ORDER BY category
        """)
        
        return {row['category']: row['count'] for row in cursor.fetchall()}
    
    def search_questions(self, search_term: str) -> List[Dict]:
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, query, reference_answer, url, category
            FROM questions
            WHERE LOWER(query) LIKE LOWER(?)
               OR LOWER(reference_answer) LIKE LOWER(?)
            ORDER BY id
        """, (f'%{search_term}%', f'%{search_term}%'))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        cursor = self.conn.cursor()
        
        # total questions
        cursor.execute("SELECT COUNT(*) as count FROM questions")
        total_questions = cursor.fetchone()['count']
        
        # categories
        categories = self.get_categories()
        
        # has embeddings
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        has_embeddings = cursor.fetchone()['count'] > 0
        
        # total nuggets
        cursor.execute("SELECT COUNT(*) as count FROM nuggets")
        total_nuggets = cursor.fetchone()['count']
        
        # total gold passages
        cursor.execute("SELECT COUNT(*) as count FROM gold_passages")
        total_passages = cursor.fetchone()['count']
        
        return {
            'total_questions': total_questions,
            'categories': categories,
            'has_embeddings': has_embeddings,
            'total_nuggets': total_nuggets,
            'total_gold_passages': total_passages
        }
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gold Set Database Manager")
    parser.add_argument("--import", dest="import_file", 
                       help="Import from JSONL file")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing data")
    parser.add_argument("--stats", action="store_true",
                       help="Show database statistics")
    parser.add_argument("--search", help="Search questions")
    parser.add_argument("--category", help="List questions in category")
    parser.add_argument("--db", default="backend/data/gold_set.db",
                       help="Database path")
    
    args = parser.parse_args()
    
    with GoldSetDB(args.db) as db:
        if args.import_file:
            print(f"Importing from {args.import_file}...")
            db.import_from_jsonl(args.import_file, overwrite=args.overwrite)
        
        if args.stats:
            print("\n=== Database Statistics ===")
            stats = db.get_statistics()
            print(f"Total Questions: {stats['total_questions']}")
            print(f"Total Nuggets: {stats['total_nuggets']}")
            print(f"Total Gold Passages: {stats['total_gold_passages']}")
            print(f"Has Embeddings: {stats['has_embeddings']}")
            print(f"\nCategories:")
            for cat, count in stats['categories'].items():
                print(f"  {cat}: {count}")
        
        if args.search:
            print(f"\n=== Search Results for '{args.search}' ===")
            results = db.search_questions(args.search)
            for q in results:
                print(f"\n[{q['id']}] {q['query']}")
                print(f"Answer: {q['reference_answer'][:100]}...")
        
        if args.category:
            print(f"\n=== Questions in Category '{args.category}' ===")
            results = db.get_questions_by_category(args.category)
            for q in results:
                print(f"[{q['id']}] {q['query']}")