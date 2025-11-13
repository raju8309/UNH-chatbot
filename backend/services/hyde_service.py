"""
HyDE (Hypothetical Document Embeddings) Service

Instead of embedding the user's query directly, we generate a hypothetical answer
that the document might contain, then embed that. This improves retrieval because:
1. Hypothetical answers use similar language/structure as real documents
2. Answer-to-answer matching works better than query-to-answer matching
3. LLM can inject domain knowledge to expand the query

This runs at QUERY TIME (not index time like synthetic Q&A).
"""
from typing import Optional
from transformers import pipeline
from config.settings import get_config, load_retrieval_config
import torch

class HyDEGenerator:
    """Generates hypothetical document answers for better semantic search."""
    
    def __init__(self):
        load_retrieval_config()
        self.hyde_pipeline = None
        
    def _get_pipeline(self):
        """Load model for hypothetical document generation."""
        if self.hyde_pipeline is None:
            cfg = get_config()
            hyde_cfg = cfg.get("hyde", {})
            
            # Use SMALL model by default - this runs at query time!
            model_name = hyde_cfg.get("model", "google/flan-t5-small")
            force_cpu = hyde_cfg.get("force_cpu", False)  # Default to GPU if available for speed
            
            if force_cpu:
                device = -1
                print(f"HyDE using CPU (force_cpu=true)")
            elif torch.cuda.is_available():
                device = 0
                device_name = torch.cuda.get_device_name(0)
                print(f"HyDE using GPU: {device_name}")
            else:
                device = -1
                print(f"HyDE using CPU (no GPU detected)")
            
            print(f"Loading HyDE model: {model_name} (query-time generation - needs to be fast!)")
            
            # Determine pipeline type
            if any(name in model_name.lower() for name in ['t5', 'bart', 'pegasus']):
                pipeline_type = "text2text-generation"
            else:
                pipeline_type = "text-generation"
            
            self.hyde_pipeline = pipeline(
                pipeline_type,
                model=model_name,
                device=device,
                max_length=512,
                trust_remote_code=True
            )
            self.pipeline_type = pipeline_type
            print(f"  ✓ HyDE model loaded")
            
        return self.hyde_pipeline
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical answer to the query as it might appear in the catalog.
        
        Args:
            query: User's question
            
        Returns:
            Hypothetical document text that answers the query
        """
        cfg = get_config()
        hyde_cfg = cfg.get("hyde", {})
        
        if not hyde_cfg.get("enabled", False):
            # If HyDE disabled, just return the original query
            return query
        
        try:
            pipeline = self._get_pipeline()
            
            # Craft prompt based on pipeline type
            if self.pipeline_type == "text2text-generation":
                # T5/FLAN-T5: Clear instruction format
                prompt = (
                    f"Write a detailed answer to this question as it would appear in a university graduate catalog. "
                    f"Include specific policies, requirements, and procedures.\n\n"
                    f"Question: {query}\n\n"
                    f"Answer:"
                )
                
                max_tokens = hyde_cfg.get("max_tokens", 150)
                temperature = hyde_cfg.get("temperature", 0.5)
                
                result = pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                    repetition_penalty=1.2
                )
                
                hypothetical_doc = result[0]["generated_text"].strip()
                
            else:
                # Causal LM (GPT-style): Instruction format
                tokenizer = pipeline.tokenizer
                
                prompt = (
                    f"Instruct: Write a detailed answer to this question as it would appear "
                    f"in a university graduate catalog:\n\n{query}\n\nOutput:"
                )
                
                max_tokens = hyde_cfg.get("max_tokens", 150)
                temperature = hyde_cfg.get("temperature", 0.5)
                
                result = pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                    repetition_penalty=1.2,
                    return_full_text=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                hypothetical_doc = result[0]["generated_text"].strip()
            
            # Clean up the output
            hypothetical_doc = self._clean_output(hypothetical_doc)
            
            # If generation failed or is too short, fall back to original query
            if len(hypothetical_doc) < 20:
                print(f"  Warning: HyDE generation too short, using original query")
                return query
            
            if hyde_cfg.get("verbose", False):
                print(f"\n[HyDE] Original query: {query}")
                print(f"[HyDE] Hypothetical doc: {hypothetical_doc[:200]}...")
            
            return hypothetical_doc
            
        except Exception as e:
            print(f"Warning: HyDE generation failed: {e}")
            return query  # Fall back to original query
    
    def _clean_output(self, text: str) -> str:
        """Clean up generated hypothetical document."""
        # Remove common artifacts and prompt leakage
        text = text.replace("Answer:", "").strip()
        text = text.replace("Output:", "").strip()
        text = text.replace("Question:", "").strip()
        
        # Remove prompt instructions that leaked through
        prompt_fragments = [
            "Write a detailed answer",
            "as it would appear in a university graduate catalog",
            "Include specific policies, requirements, and procedures",
            "provide specific policies, requirements, and procedures",
            "Write an answer",
            "based on the following",
        ]
        
        for fragment in prompt_fragments:
            # Case-insensitive removal
            import re
            text = re.sub(re.escape(fragment), '', text, flags=re.IGNORECASE)
        
        # Remove leading/trailing punctuation artifacts
        text = text.strip('.:;,- \n\t')
        
        # Take only first paragraph/section if multiple
        # (we want focused hypothetical docs)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not any(frag.lower() in line.lower() for frag in prompt_fragments):
                cleaned_lines.append(line)
                # Stop after first substantial paragraph
                if len(' '.join(cleaned_lines)) > 100:
                    break
        
        result = ' '.join(cleaned_lines).strip()
        
        # Final cleanup - ensure it doesn't start with lowercase article/preposition
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        return result
    
    def should_use_hyde(self, query: str) -> bool:
        """
        Determine if HyDE should be used for this query.
        
        Can skip HyDE for:
        - Very short queries (< 3 words)
        - Simple lookups (just a course code)
        - Navigation queries
        
        Returns True if HyDE should be used.
        """
        cfg = get_config()
        hyde_cfg = cfg.get("hyde", {})
        
        # Check if globally enabled
        if not hyde_cfg.get("enabled", False):
            return False
        
        # Check minimum query length
        min_words = hyde_cfg.get("min_query_words", 3)
        if len(query.split()) < min_words:
            return False
        
        # Skip for simple course code lookups (e.g., "CS 725")
        import re
        course_code_regex = r"^\s*[A-Z]{2,5}\s*\d{3}[A-Z]?\s*$"
        if re.match(course_code_regex, query):
            return False
        
        return True


# Global instance
_hyde_generator = None

def get_hyde_generator() -> HyDEGenerator:
    """Get or create the global HyDE generator instance."""
    global _hyde_generator
    if _hyde_generator is None:
        _hyde_generator = HyDEGenerator()
    return _hyde_generator
