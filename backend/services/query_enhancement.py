from typing import List, Dict
import re
from sentence_transformers import CrossEncoder

class OpenSourceQueryEnhancer:
    def __init__(self):
        # use free cross-encoder for query expansion
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def expand_acronyms(self, query: str) -> str:
        expansions = {
            r'\bms\b': 'master of science masters',
            r'\bm\.s\.\b': 'master of science',
            r'\bphd\b': 'doctorate doctoral doctor of philosophy',
            r'\bph\.d\.\b': 'doctorate doctoral',
            r'\bgpa\b': 'grade point average gpa',
            r'\bgre\b': 'graduate record examination gre',
            r'\bgmat\b': 'graduate management admission test gmat',
            r'\btoefl\b': 'test of english foreign language toefl',
            r'\bcs\b': 'computer science',
            r'\bee\b': 'electrical engineering',
            r'\bme\b': 'mechanical engineering',
        }
        
        expanded = query.lower()
        for pattern, expansion in expansions.items():
            expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
        
        return expanded
    
    def extract_key_terms(self, query: str) -> List[str]:
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'do', 'does', 
                     'can', 'i', 'my', 'in', 'for', 'to', 'of', 'about'}
        
        words = query.lower().split()
        key_terms = [w for w in words if w not in stopwords and len(w) > 2]
        
        # add bi-grams for common academic phrases
        bigrams = []
        for i in range(len(words) - 1):
            if words[i] not in stopwords and words[i+1] not in stopwords:
                bigrams.append(f"{words[i]} {words[i+1]}")
        
        return key_terms + bigrams
    
    def boost_program_terms(self, query: str) -> str:
        program_indicators = [
            'requirements', 'credits', 'courses', 'admission',
            'curriculum', 'degree', 'program', 'graduate'
        ]
        
        boosted = query
        for term in program_indicators:
            if term in query.lower():
                # repeat important terms for emphasis in semantic search
                boosted += f" {term}"
        
        return boosted
    
    def rewrite_for_policy_queries(self, query: str) -> str:
        policy_patterns = {
            r'(can|may) i': 'student policy for',
            r'how (many|much)': 'requirement for',
            r'what (is|are) the': 'policy regarding',
        }
        
        rewritten = query
        for pattern, replacement in policy_patterns.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        return rewritten
    
    def enhance_query(self, query: str, query_type: str = 'general') -> Dict:
        # start with original
        enhanced = {
            "original": query,
            "expanded": query,
            "key_terms": [],
            "rewritten": query
        }
        
        expanded = self.expand_acronyms(query)
        enhanced["expanded"] = expanded
        
        enhanced["key_terms"] = self.extract_key_terms(query)
        
        boosted = self.boost_program_terms(expanded)
        
        if query_type == 'policy':
            rewritten = self.rewrite_for_policy_queries(boosted)
        else:
            rewritten = boosted
        
        enhanced["rewritten"] = rewritten
        
        return enhanced
    
    def generate_sub_queries(self, query: str) -> List[str]:
        sub_queries = [query]
        
        if ' and ' in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            if len(parts) == 2:
                sub_queries.extend(parts)
        
        question_words = ['what', 'how', 'when', 'where', 'why']
        sentences = re.split(r'[.?!]', query)
        for sent in sentences:
            if any(qw in sent.lower() for qw in question_words):
                sub_queries.append(sent.strip())
        
        sub_queries = list(set([q.strip() for q in sub_queries if q.strip()]))
        
        return sub_queries[:3]