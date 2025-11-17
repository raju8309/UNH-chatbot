from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from config.settings import get_config

# import the database manager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.gold_db import GoldSetDB

class GoldSetManager:
    def __init__(self, db_path: str = None):
        # load configuration
        self.config = get_config()
        self.enabled = self.config.get("gold_set", {}).get("enabled", True)

        # database setup - resolve path relative to this file
        if db_path is None:
            # Default: backend/data/gold_set.db relative to this file
            db_path = Path(__file__).parent.parent / "data" / "gold_set.db"
        self.db_path = Path(db_path)
        self.db = None

        # in-memory caches
        self.gold_questions: Set[str] = set()
        self.gold_answers: Dict[str, str] = {}
        self.gold_urls: Dict[str, str] = {}
        self.gold_embeddings: Optional[np.ndarray] = None
        self.gold_question_ids: List[str] = []
        self.embed_model: Optional[SentenceTransformer] = None

        if self.enabled:
            self._initialize_database()
            self._load_cache()
        else:
            print("Gold set disabled — skipping database initialization.")

    def _initialize_database(self):
        try:
            # Ensure the database file exists
            if not self.db_path.exists():
                print(f"WARNING: Gold set database not found at: {self.db_path}")
                print(f"         Absolute path: {self.db_path.absolute()}")
                # Check if migration is needed
                parent_dir = self.db_path.parent
                if parent_dir.exists():
                    print(f"         Database directory exists but file is missing")
                    print(f"         Run: python backend/data/migrate_gold_db.py")
                else:
                    print(f"         Database directory does not exist: {parent_dir}")
                self.db = None
                return
            
            print(f"Loading gold set database from: {self.db_path.absolute()}")
            self.db = GoldSetDB(str(self.db_path))
            stats = self.db.get_statistics()
            print(f"Gold set database initialized: {stats['total_questions']} questions")
        except Exception as e:
            print(f"Error initializing gold set database: {e}")
            import traceback
            traceback.print_exc()
            self.db = None

    def _load_cache(self):
        if not self.db:
            return

        try:
            all_questions = []
            question_ids = []

            for q in self.db.get_all_questions():
                query_lower = q['query'].lower()
                self.gold_questions.add(query_lower)
                self.gold_answers[query_lower] = q['reference_answer']
                self.gold_urls[query_lower] = q['url']

                full_q = self.db.get_question_by_id(q['id'])
                if full_q:
                    all_questions.append(full_q)
                    question_ids.append(q['id'])

            self._cached_questions = all_questions
            self._cached_question_ids = question_ids

            print(f"Cached {len(self.gold_questions)} gold questions with full data")

        except Exception as e:
            print(f"Error loading gold set cache: {e}")

    def get_gold_documents(self) -> List[Document]:
        if not self.enabled or not self.db:
            return []

        documents = []
        try:
            questions = self.db.get_all_questions()
            for q in questions:
                full_q = self.db.get_question_by_id(q['id'])
                if not full_q:
                    continue

                content = f"Question: {full_q['query']}\n\nAnswer: {full_q['reference_answer']}"
                nuggets = full_q.get('nuggets', [])
                if nuggets:
                    content += f"\n\nKey Points: {'; '.join(nuggets)}"

                # Create a clean, user-friendly title from category
                gold_id = full_q['id']
                category = full_q.get('category', '')
                if category:
                    # Convert "academic-standards" to "Academic Standards"
                    clean_category = category.replace('-', ' ').replace('_', ' ').title()
                    friendly_title = f"{clean_category} Information"
                else:
                    friendly_title = "Graduate Catalog Information"

                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'gold_set',
                        'gold_id': gold_id,
                        'url': full_q['url'],
                        'tier': 0,
                        'is_gold': True,
                        'original_query': full_q['query'],
                        'gold_passages': full_q.get('gold_passages', []),
                        'category': category,
                        'title': friendly_title  # Use friendly title instead of "Gold Q&A: id"
                    }
                )
                documents.append(doc)

            print(f"Generated {len(documents)} gold documents")

        except Exception as e:
            print(f"Error generating gold documents: {e}")

        return documents

    def compute_gold_embeddings(self, embed_model: SentenceTransformer):
        if not self.enabled or not self.db:
            return None

        self.embed_model = embed_model

        try:
            existing_embeddings, existing_ids = self.db.get_embeddings()

            if existing_embeddings is not None and len(existing_ids) > 0:
                print(f"Loaded {len(existing_ids)} existing gold embeddings from database")
                self.gold_embeddings = existing_embeddings
                self.gold_question_ids = existing_ids
                return

            questions = self.db.get_all_questions()
            if not questions:
                print("No questions found in database for embedding computation.")
                return

            queries = [q['query'] for q in questions]
            question_ids = [q['id'] for q in questions]

            print(f"Computing embeddings for {len(queries)} gold questions...")
            embeddings = embed_model.encode(queries, convert_to_numpy=True)

            self.db.store_embeddings(
                embeddings,
                question_ids,
                model_name=embed_model._model_card_vars.get('model_name', 'unknown')
            )

            self.gold_embeddings = embeddings
            self.gold_question_ids = question_ids

            print(f"Computed and stored embeddings for {len(queries)} gold questions")

        except Exception as e:
            print(f"Error computing gold embeddings: {e}")

    def find_matching_gold_entry(self, query: str, threshold: float = 0.85) -> Optional[Dict]:
        if not self.enabled or self.gold_embeddings is None or self.embed_model is None:
            return None
        if not self.db:
            return None

        try:
            query_embedding = self.embed_model.encode([query], convert_to_numpy=True)[0]
            similarities = np.dot(self.gold_embeddings, query_embedding) / (
                np.linalg.norm(self.gold_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= threshold:
                question_id = self.gold_question_ids[best_idx]
                entry = self.db.get_question_by_id(question_id)
                if entry:
                    entry['match_score'] = best_score
                    return entry

        except Exception as e:
            print(f"Error finding matching gold entry: {e}")

        return None

    def get_direct_answer_with_similarity(
        self, 
        query: str, 
        threshold: float = None
    ) -> Optional[Tuple[str, float, Dict]]:
        """
        Check if query has high semantic similarity to a gold set question.
        If similarity exceeds threshold, return the gold answer directly.
        
        Returns:
            Tuple of (answer, similarity_score, metadata) if match found, else None
        """
        if not self.enabled or self.gold_embeddings is None or self.embed_model is None:
            return None
        if not self.db:
            return None
        
        # Use threshold from config if not provided
        if threshold is None:
            gold_cfg = self.config.get("gold_set", {})
            threshold = float(gold_cfg.get("direct_answer_threshold", 0.85))
        
        try:
            # Encode the query
            query_embedding = self.embed_model.encode([query], convert_to_numpy=True)[0]
            
            # Compute similarities with all gold questions
            similarities = np.dot(self.gold_embeddings, query_embedding) / (
                np.linalg.norm(self.gold_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )

            # Find best match
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            # If similarity exceeds threshold, return direct answer
            if best_score >= threshold:
                question_id = self.gold_question_ids[best_idx]
                entry = self.db.get_question_by_id(question_id)
                
                if entry:
                    answer = entry.get('reference_answer', '')
                    metadata = {
                        'gold_id': question_id,
                        'gold_query': entry.get('query', ''),
                        'url': entry.get('url', ''),
                        'similarity_score': best_score,
                        'match_type': 'direct_gold_match',
                        'nuggets': entry.get('nuggets', []),
                        'category': entry.get('category', '')
                    }
                    
                    print(f"[GOLD SET] Direct answer match found: {question_id} (similarity: {best_score:.3f})")
                    return (answer, best_score, metadata)
        
        except Exception as e:
            print(f"Error in direct answer matching: {e}")
        
        return None

    def should_use_direct_answer(self, query: str) -> bool:
        """
        Check if we should bypass retrieval and use direct gold answer.
        
        Returns:
            True if high similarity match exists and direct answers are enabled
        """
        if not self.enabled:
            return False
        
        gold_cfg = self.config.get("gold_set", {})
        if not gold_cfg.get("enable_direct_answer", True):
            return False
        
        threshold = float(gold_cfg.get("direct_answer_threshold", 0.85))
        result = self.get_direct_answer_with_similarity(query, threshold)
        
        return result is not None

    def get_direct_answer(self, query: str, threshold: float = 0.85) -> Optional[str]:
        """
        Legacy method - returns just the answer string.
        For new code, use get_direct_answer_with_similarity() instead.
        """
        if not self.enabled:
            return None

        result = self.get_direct_answer_with_similarity(query, threshold)
        if result:
            answer, _, _ = result
            return answer
        return None

    def is_gold_url(self, url: str) -> bool:
        if not self.enabled:
            return False
        return any(url in gold_url for gold_url in self.gold_urls.values())

    def get_gold_boost_for_chunk(self, chunk_text: str, chunk_metadata: Dict, query: str) -> float:
        if not self.enabled:
            return 1.0

        boost = 1.0

        if chunk_metadata.get('is_gold', False):
            boost *= 2.5

        chunk_url = chunk_metadata.get('url', '')
        if self.is_gold_url(chunk_url):
            boost *= 1.5

        query_lower = query.lower()
        if query_lower in self.gold_questions:
            gold_answer = self.gold_answers.get(query_lower, '')
            if gold_answer and gold_answer.lower() in chunk_text.lower():
                boost *= 1.8

        return boost

    def get_statistics(self) -> Dict:
        if not self.enabled or not self.db:
            return {
                'enabled': False,
                'total_entries': 0,
                'total_questions': 0,
                'categories': {},
                'has_embeddings': False,
                'direct_answer_enabled': False,
                'direct_answer_threshold': 0.0
            }

        try:
            stats = self.db.get_statistics()
            gold_cfg = self.config.get("gold_set", {})
            return {
                'enabled': True,
                'total_entries': stats['total_questions'],
                'total_questions': len(self.gold_questions),
                'categories': stats['categories'],
                'has_embeddings': stats['has_embeddings'],
                'direct_answer_enabled': gold_cfg.get('enable_direct_answer', True),
                'direct_answer_threshold': gold_cfg.get('direct_answer_threshold', 0.85)
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'enabled': True,
                'total_entries': 0,
                'total_questions': len(self.gold_questions),
                'categories': {},
                'has_embeddings': False,
                'direct_answer_enabled': False,
                'direct_answer_threshold': 0.0
            }

    def search_questions(self, search_term: str) -> List[Dict]:
        if not self.enabled or not self.db:
            return []

        try:
            return self.db.search_questions(search_term)
        except Exception as e:
            print(f"Error searching questions: {e}")
            return []

    def get_questions_by_category(self, category: str) -> List[Dict]:
        if not self.enabled or not self.db:
            return []

        try:
            return self.db.get_questions_by_category(category)
        except Exception as e:
            print(f"Error getting questions by category: {e}")
            return []

    def close(self):
        if self.db:
            self.db.close()

    def __del__(self):
        self.close()

# global instance
_gold_manager: Optional[GoldSetManager] = None

def get_gold_manager() -> GoldSetManager:
    global _gold_manager
    if _gold_manager is None:
        _gold_manager = GoldSetManager()
    return _gold_manager

def initialize_gold_set(embed_model: SentenceTransformer):
    manager = get_gold_manager()
    manager.compute_gold_embeddings(embed_model)
    return manager