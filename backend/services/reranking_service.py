from sentence_transformers import CrossEncoder
from typing import List, Tuple, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class OpenSourceReranker:
    def __init__(self):
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # TF-IDF for keyword matching
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def cross_encoder_score(self, query: str, chunks: List[Tuple[str, Dict]]) -> List[float]:
        pairs = [[query, chunk[0][:512]] for chunk in chunks]  # Limit length
        # get scores
        scores = self.cross_encoder.predict(pairs)
        return scores.tolist()
    
    def tfidf_score(self, query: str, chunks: List[Tuple[str, Dict]]) -> List[float]:
        docs = [query] + [chunk[0] for chunk in chunks]
        try:
            tfidf_matrix = self.tfidf.fit_transform(docs)
            
            # compute similarity with query
            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vec, doc_vecs)[0]
            return similarities.tolist()
        except:
            # return uniform scores on error
            return [0.5] * len(chunks)
    
    def metadata_score(self, chunks: List[Tuple[str, Dict]], query: str) -> List[float]:
        scores = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        for text, meta in chunks:
            score = 0.5
            
            tier = meta.get("tier", 3)
            score += (4 - tier) * 0.15
            
            title = (meta.get("title") or "").lower()
            title_words = set(title.split())
            overlap = len(query_terms & title_words)
            if overlap > 0:
                score += min(overlap * 0.1, 0.3)
            
            url = (meta.get("url") or "").lower()
            if any(term in url for term in query_terms):
                score += 0.15
            
            scores.append(min(max(score, 0.0), 1.0))
        
        return scores
    
    def position_bias_adjustment(self, scores: List[float], positions: List[int]) -> List[float]:
        adjusted = []
        for score, pos in zip(scores, positions):
            # Logarithmic decay for position bias
            position_boost = 1.0 / (1.0 + np.log1p(pos))
            adjusted.append(score * (0.85 + 0.15 * position_boost))
        return adjusted
    
    def rerank(
        self, 
        query: str,
        chunks: List[Tuple[str, Dict]],
        semantic_scores: List[float],
        use_cross_encoder: bool = True,
        use_tfidf: bool = True,
        top_k: int = 5
    ) -> List[int]:
        
        if not chunks:
            return []
        
        scores_dict = {
            'semantic': np.array(semantic_scores[:len(chunks)]),
            'metadata': np.array(self.metadata_score(chunks, query))
        }
        
        if use_cross_encoder:
            top_candidates = min(len(chunks), 20)
            ce_scores = self.cross_encoder_score(query, chunks[:top_candidates])
            ce_scores.extend([0.3] * (len(chunks) - top_candidates))
            scores_dict['cross_encoder'] = np.array(ce_scores)
        
        if use_tfidf:
            scores_dict['tfidf'] = np.array(self.tfidf_score(query, chunks))
        
        positions = list(range(len(chunks)))
        scores_dict['semantic'] = self.position_bias_adjustment(
            scores_dict['semantic'].tolist(), 
            positions
        )
        
        weights = {
            'semantic': 0.25,
            'metadata': 0.15,
            'cross_encoder': 0.45 if use_cross_encoder else 0.0,
            'tfidf': 0.15 if use_tfidf else 0.0
        }
        
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        final_scores = np.zeros(len(chunks))
        for feature, weight in weights.items():
            if feature in scores_dict:
                final_scores += weight * np.array(scores_dict[feature])
        
        top_indices = np.argsort(final_scores)[::-1][:top_k].tolist()
        
        return top_indices
    
    def diversity_filter(
        self,
        chunks: List[Tuple[str, Dict]],
        ranked_indices: List[int],
        diversity_threshold: float = 0.7
    ) -> List[int]:
        if len(ranked_indices) <= 3:
            return ranked_indices
        
        selected = [ranked_indices[0]]  # Always keep top
        
        for idx in ranked_indices[1:]:
            chunk_text = chunks[idx][0]
            chunk_words = set(chunk_text.lower().split())
            
            is_diverse = True
            for selected_idx in selected:
                selected_text = chunks[selected_idx][0]
                selected_words = set(selected_text.lower().split())
                
                if len(chunk_words) > 0 and len(selected_words) > 0:
                    overlap = len(chunk_words & selected_words)
                    union = len(chunk_words | selected_words)
                    similarity = overlap / union
                    
                    if similarity > diversity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                selected.append(idx)
        
        return selected