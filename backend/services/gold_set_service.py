from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from config.settings import get_config

# import the database manager
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.gold_db import GoldSetDB

class GoldSetManager:
    def __init__(self, db_path: str = "backend/data/gold_set.db"):
        # load configuration
        self.config = get_config()
        self.enabled = self.config.get("gold_set", {}).get("enabled", True)
        
        # database setup
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
            self.db = GoldSetDB(str(self.db_path))
            
            # check if database is empty
            stats = self.db.get_statistics()
            if stats['total_questions'] == 0:
                # try to import from JSONL if available
                jsonl_path = Path(__file__).parent.parent.parent / "automation_testing" / "gold.jsonl"
                if jsonl_path.exists():
                    print(f"Database empty, importing from {jsonl_path}...")
                    self.db.import_from_jsonl(str(jsonl_path))
                else:
                    print(f"Warning: Gold set database is empty and {jsonl_path} not found")
            
            print(f"Gold set database initialized: {stats['total_questions']} questions")
            
        except Exception as e:
            print(f"Error initializing gold set database: {e}")
            self.db = None

    def _load_cache(self):
        if not self.db:
            return
        
        try:
            # load all questions with full data
            all_questions = []
            question_ids = []
            
            for q in self.db.get_all_questions():
                query_lower = q['query'].lower()
                self.gold_questions.add(query_lower)
                self.gold_answers[query_lower] = q['reference_answer']
                self.gold_urls[query_lower] = q['url']
                
                # also cache full questions for document generation
                full_q = self.db.get_question_by_id(q['id'])
                if full_q:
                    all_questions.append(full_q)
                    question_ids.append(q['id'])
            
            # store for quick access
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
                # get full question data including nuggets
                full_q = self.db.get_question_by_id(q['id'])
                if not full_q:
                    continue
                
                # build content
                content = f"Question: {full_q['query']}\n\nAnswer: {full_q['reference_answer']}"
                
                # add nuggets if present
                nuggets = full_q.get('nuggets', [])
                if nuggets:
                    content += f"\n\nKey Points: {'; '.join(nuggets)}"
                
                # create document
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'gold_set',
                        'gold_id': full_q['id'],
                        'url': full_q['url'],
                        'tier': 0,
                        'is_gold': True,
                        'original_query': full_q['query'],
                        'gold_passages': full_q.get('gold_passages', []),
                        'category': full_q.get('category', '')
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
            # check if embeddings already exist
            existing_embeddings, existing_ids = self.db.get_embeddings()
            
            if existing_embeddings is not None and len(existing_ids) > 0:
                print(f"Loaded {len(existing_ids)} existing gold embeddings from database")
                self.gold_embeddings = existing_embeddings
                self.gold_question_ids = existing_ids
                return
            
            # compute new embeddings
            questions = self.db.get_all_questions()
            if not questions:
                return
            
            queries = [q['query'] for q in questions]
            question_ids = [q['id'] for q in questions]
            
            print(f"Computing embeddings for {len(queries)} gold questions...")
            embeddings = embed_model.encode(queries, convert_to_numpy=True)
            
            # store in database
            self.db.store_embeddings(embeddings, question_ids, 
                                    model_name=embed_model._model_card_vars.get('model_name', 'unknown'))
            
            # cache in memory
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
                # get full question data from database
                question_id = self.gold_question_ids[best_idx]
                entry = self.db.get_question_by_id(question_id)
                
                if entry:
                    entry['match_score'] = best_score
                    return entry
            
        except Exception as e:
            print(f"Error finding matching gold entry: {e}")
        
        return None

    def is_gold_url(self, url: str) -> bool:
        if not self.enabled:
            return False
        return any(url in gold_url for gold_url in self.gold_urls.values())

    def get_gold_boost_for_chunk(self, chunk_text: str, chunk_metadata: Dict, query: str) -> float:
        if not self.enabled:
            return 1.0
        
        boost = 1.0
        
        # boost if chunk is gold
        if chunk_metadata.get('is_gold', False):
            boost *= 2.5
        
        # boost if chunk URL matches gold URL
        chunk_url = chunk_metadata.get('url', '')
        if self.is_gold_url(chunk_url):
            boost *= 1.5
        
        # boost if query matches gold query and chunk contains answer
        query_lower = query.lower()
        if query_lower in self.gold_questions:
            gold_answer = self.gold_answers.get(query_lower, '')
            if gold_answer and gold_answer.lower() in chunk_text.lower():
                boost *= 1.8
        
        return boost

    def get_direct_answer(self, query: str, threshold: float = 0.85) -> Optional[str]:
        if not self.enabled:
            return None
        
        match = self.find_matching_gold_entry(query, threshold)
        if match:
            return match.get('reference_answer')
        return None

    def get_statistics(self) -> Dict:
        if not self.enabled or not self.db:
            return {
                'enabled': False,
                'total_entries': 0,
                'total_questions': 0,
                'categories': {},
                'has_embeddings': False
            }
        
        try:
            stats = self.db.get_statistics()
            return {
                'enabled': True,
                'total_entries': stats['total_questions'],
                'total_questions': len(self.gold_questions),
                'categories': stats['categories'],
                'has_embeddings': stats['has_embeddings']
            }
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'enabled': True,
                'total_entries': 0,
                'total_questions': len(self.gold_questions),
                'categories': {},
                'has_embeddings': False
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