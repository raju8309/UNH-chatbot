from typing import List, Tuple, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class OpenSourceCompressor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def extract_query_relevant_sentences(
        self, 
        query: str, 
        text: str,
        top_n: int = 3
    ) -> str:
        # split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text
        
        # use TF-IDF to find most relevant sentences
        try:
            docs = [query] + sentences
            tfidf_matrix = self.tfidf.fit_transform(docs)
            
            # compare each sentence to query
            query_vec = tfidf_matrix[0:1]
            sentence_vecs = tfidf_matrix[1:]
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vec, sentence_vecs)[0]
            
            # get top N sentences
            top_indices = np.argsort(similarities)[::-1][:top_n]
            top_sentences = [sentences[i] for i in sorted(top_indices)]
            
            return '. '.join(top_sentences) + '.'
        except:
            # fallback to first few sentences
            return '. '.join(sentences[:top_n]) + '.'
    
    def extract_key_information(self, text: str, query: str) -> str:
        key_patterns = [
            r'\d+\s*credits?',
            r'\d+\.\d+\s*GPA',
            r'[A-Z]{2,5}\s+\d{3}',
            r'deadline[s]?:?\s*[^.]*',
            r'require[sd]?:?\s*[^.]*',
            r'prerequisite[s]?:?\s*[^.]*',
            r'\d+\s*(?:semester|year|month)s?',
        ]
        
        extracted = []
        for pattern in key_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context (50 chars each side)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                if context not in extracted:
                    extracted.append(context)
        
        if extracted:
            return ' ... '.join(extracted)
        
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        # patterns to remove
        boilerplate_patterns = [
            r'for more information.*?visit.*?website',
            r'please contact.*?for.*?questions',
            r'copyright.*?all rights reserved',
            r'last updated.*?\d{4}',
        ]
        
        cleaned = text
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned.strip()
    
    def compress_chunk(
        self, 
        query: str, 
        chunk_text: str,
        max_ratio: float = 0.6
    ) -> str:
        cleaned = self.remove_boilerplate(chunk_text)
        
        key_info = self.extract_key_information(cleaned, query)
        
        if len(key_info) > len(cleaned) * 0.3:
            compressed = key_info
        else:
            compressed = self.extract_query_relevant_sentences(query, cleaned, top_n=3)
        
        if len(compressed) < len(chunk_text) * max_ratio:
            return compressed
        
        return chunk_text
    
    def compress_chunks(
        self, 
        query: str, 
        chunks: List[Tuple[str, Dict]], 
        max_chunks: int = 5,
        aggressive: bool = False
    ) -> List[Tuple[str, Dict]]:
        compressed = []
        
        for text, source in chunks[:max_chunks]:
            if len(text) < 200:
                compressed.append((text, source))
                continue
            
            tier = source.get("tier", 3)
            
            if tier == 1:
                max_ratio = 0.8 if not aggressive else 0.6
            else:
                max_ratio = 0.6 if not aggressive else 0.4
            
            try:
                compressed_text = self.compress_chunk(query, text, max_ratio)
                compressed.append((compressed_text, source))
            except Exception as e:
                # Fallback to original on error
                compressed.append((text, source))
        
        return compressed
    
    def deduplicate_content(self, chunks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        if len(chunks) <= 1:
            return chunks
        
        unique_chunks = [chunks[0]]
        
        for text, source in chunks[1:]:
            words = set(text.lower().split())
            
            is_duplicate = False
            for existing_text, _ in unique_chunks:
                existing_words = set(existing_text.lower().split())
                
                if len(words) > 0 and len(existing_words) > 0:
                    overlap = len(words & existing_words)
                    union = len(words | existing_words)
                    similarity = overlap / union
                    
                    if similarity > 0.8:  # 80% similar = duplicate
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_chunks.append((text, source))
        
        return unique_chunks