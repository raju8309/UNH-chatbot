"""
Synthetic Q&A Generation Service
Converts catalog chunks into Q&A format to improve semantic matching with user queries.
Uses a larger, more capable LLM for question generation (only runs during indexing, not live).
"""
import re
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from transformers import pipeline
from config.settings import get_config, load_retrieval_config

class SyntheticQAGenerator:
    """Generates question-answer pairs from catalog text chunks using a larger LLM."""
    
    def __init__(self):
        # Ensure config is loaded
        load_retrieval_config()
        self.qa_pipeline = None
        self.output_file = None  # File handle for incremental saving
        
    def _get_pipeline(self):
        """
        Load a larger, more capable model for question generation.
        Since this only runs during index building (not live queries), we can afford it.
        """
        if self.qa_pipeline is None:
            import torch
            cfg = get_config()
            qa_cfg = cfg.get("synthetic_qa", {})
            
            # Get model from config
            model_name = qa_cfg.get("question_model", "google/flan-t5-large")
            # Check force_cpu setting
            force_cpu = qa_cfg.get("force_cpu", True)
            
            if force_cpu:
                device = -1
                print(f"Using CPU for question generation (force_cpu=true in config)")
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    print(f"   GPU available but disabled: {device_name}")
                    print(f"   Set force_cpu=false to try GPU")
            elif torch.cuda.is_available():
                device = 0
                device_name = torch.cuda.get_device_name(0)
                print(f"🚀 Using GPU: {device_name}")
            else:
                device = -1
                print(f"No GPU detected, using CPU")
            
            print(f"Loading question generation model: {model_name}")
            
            # Determine pipeline type based on model
            # T5/FLAN-T5/BART use text2text-generation
            # Phi-2, Qwen, Llama, Mistral use text-generation (causal LM)
            if any(name in model_name.lower() for name in ['t5', 'bart', 'pegasus']):
                pipeline_type = "text2text-generation"
            else:
                pipeline_type = "text-generation"
            
            print(f"  Using pipeline: {pipeline_type}")
            
            self.qa_pipeline = pipeline(
                pipeline_type,
                model=model_name,
                device=device,
                max_length=512,
                trust_remote_code=True  # Required for some models like Phi-2
            )
            self.pipeline_type = pipeline_type
            print(f"  ✓ Model loaded on {'GPU' if device == 0 else 'CPU'}")
        return self.qa_pipeline
    
    def generate_question_for_chunk(self, chunk_text: str, context_title: str = "", num_questions: int = 1) -> List[str]:
        """
        Use a larger LLM to generate multiple specific, natural questions that this chunk answers.
        Returns a list of questions (may be empty if chunk doesn't contain question-answerable content).
        
        Args:
            chunk_text: The text chunk to generate questions for
            context_title: Optional context/title for the chunk
            num_questions: Number of diverse questions to generate
            
        Returns:
            List of generated questions (may be fewer than num_questions if quality filtering removes some)
        """
        # Skip very short chunks or list items
        if len(chunk_text.strip()) < 50:
            return []
        
        try:
            pipeline = self._get_pipeline()
            cfg = get_config()
            qa_cfg = cfg.get("synthetic_qa", {})
            temperature = qa_cfg.get("temperature", 0.7)
            
            # Generate multiple questions with higher diversity
            all_questions = []
            seen_normalized = set()
            
            # Handle different output formats for different pipeline types
            if self.pipeline_type == "text2text-generation":
                # T5/FLAN-T5/BART: Generate questions ONE AT A TIME with varied prompts for diversity
                raw_questions = []
                
                # Use different prompt variations for diversity
                prompt_templates = [
                    f"Write a short question (under 12 words) that a graduate student would ask about this policy:\n\n{chunk_text[:500]}\n\nQuestion:",
                    f"What would a student want to know about this policy? Write one brief question:\n\n{chunk_text[:500]}\n\nQuestion:",
                    f"Generate a specific question about this graduate school policy:\n\n{chunk_text[:500]}\n\nQuestion:",
                ]
                
                for i in range(num_questions):
                    prompt = prompt_templates[i % len(prompt_templates)]
                    
                    result = pipeline(
                        prompt,
                        max_new_tokens=25,
                        temperature=temperature + (i * 0.15),  # Increase temp for later questions
                        do_sample=True,
                        top_p=0.9,
                        num_return_sequences=1,
                        repetition_penalty=2.0
                    )
                    raw_questions.append(result[0]["generated_text"].strip())
                
            else:
                # Causal LM: Generate questions one at a time with varied prompts
                tokenizer = pipeline.tokenizer
                raw_questions = []
                
                # Use different prompt variations for diversity
                prompt_templates = [
                    f"Instruct: Write one short question (under 12 words) that a graduate student would ask about this policy:\n\n{chunk_text[:500]}\n\nOutput:",
                    f"Instruct: What would a student want to know about this policy? Write one brief question:\n\n{chunk_text[:500]}\n\nOutput:",
                    f"Instruct: Generate a specific question about this graduate school policy:\n\n{chunk_text[:500]}\n\nOutput:",
                ]
                
                for i in range(num_questions):
                    prompt = prompt_templates[i % len(prompt_templates)]
                    
                    result = pipeline(
                        prompt,
                        max_new_tokens=20,
                        temperature=temperature + (i * 0.1),  # Increase temp for later questions
                        do_sample=True,
                        top_p=0.7,
                        num_return_sequences=1,
                        repetition_penalty=1.1,
                        return_full_text=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    raw_questions.append(result[0]["generated_text"].strip())
            
            # Clean and validate each question
            for raw_q in raw_questions:
                if not raw_q:
                    continue
                    
                question = raw_q
                
                # Stop at first question mark
                if '?' in question:
                    question = question.split('?')[0] + '?'
                
                # Remove common prefixes
                question = re.sub(r'^Question:\s*', '', question, flags=re.IGNORECASE)
                question = re.sub(r'^Student question:\s*', '', question, flags=re.IGNORECASE)
                question = re.sub(r'^Answer:\s*', '', question, flags=re.IGNORECASE)
                question = re.sub(r'^Q:\s*', '', question, flags=re.IGNORECASE)
                question = re.sub(r'^A:\s*', '', question, flags=re.IGNORECASE)
                
                # Take only first line
                question = question.split('\n')[0].strip()
                
                # Clean up common artifacts
                question = re.sub(r'^\d+[\.\)]\s*', '', question)  # Remove numbering
                question = re.sub(r'^[-•*]\s*', '', question)  # Remove bullets
                question = re.sub(r'^["\']|["\']$', '', question)  # Remove quotes
                question = question.strip()
                
                # Ensure it ends with question mark
                if question and not question.endswith('?'):
                    question += '?'
                
                # Validate minimum requirements
                if not question or len(question) < 10 or len(question) > 150:
                    continue
                
                # Filter overly generic/formal questions
                lower_q = question.lower()
                
                # Reject meta-references to the text itself
                meta_terms = ['the text', 'this passage', 'above text', 'following text', 
                             'the document', 'this document', 'described above', 'mentioned above']
                if any(bad in lower_q for bad in meta_terms):
                    continue
                
                # Reject overly generic starts (too formal) unless specific
                generic_starts = [
                    'what is the process',
                    'what are the requirements',
                    'what is the policy',
                    'what are the procedures',
                    'what is the definition'
                ]
                if any(lower_q.startswith(bad) for bad in generic_starts):
                    if not any(marker in lower_q for marker in ['master', 'ph.d', 'doctoral', 'certificate', 
                                                                 'graduate', 'credit', 'gpa', 'grade']):
                        continue
                
                # Check for duplicates (normalized)
                normalized = lower_q.strip()
                if normalized in seen_normalized:
                    continue
                
                seen_normalized.add(normalized)
                all_questions.append(question)
                
                # Stop if we have enough questions
                if len(all_questions) >= num_questions:
                    break
            
            return all_questions
            
        except Exception as e:
            print(f"Warning: Question generation failed for chunk: {e}")
            return []
    
    def create_qa_chunk(self, original_chunk: str, questions: List[str], source_meta: Dict) -> Tuple[str, Dict]:
        """
        Create a Q&A formatted chunk with MULTIPLE questions that will semantically match user queries better.
        
        Format: "Questions:\n1. {q1}\n2. {q2}\n3. {q3}\n\nAnswer: {original_chunk}"
        """
        # Clean up the chunk - remove excessive whitespace
        clean_chunk = re.sub(r'\s+', ' ', original_chunk).strip()
        
        # Create multi-question Q&A format
        if len(questions) == 1:
            # Single question - use simple format
            qa_text = f"Question: {questions[0]}\n\nAnswer: {clean_chunk}"
        else:
            # Multiple questions - use numbered list format
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            qa_text = f"Questions:\n{questions_text}\n\nAnswer: {clean_chunk}"
        
        # Copy metadata and mark as synthetic QA
        qa_meta = source_meta.copy()
        qa_meta['is_synthetic_qa'] = True
        qa_meta['questions'] = questions  # Store all questions
        qa_meta['num_questions'] = len(questions)
        
        return qa_text, qa_meta
    
    def _load_combined_qa(
        self, 
        chunks: List[Tuple[str, Dict, Dict]], 
        combined_path: Path, 
        tier_filter,
        program_filter: List[str] = [],
        keep_original_chunks: bool = True
    ) -> List[Tuple[str, Dict, Dict]]:
        """Load pre-generated combined Q&A chunks from file."""
        print(f"\nLoading pre-generated Q&A from: {combined_path}")
        
        augmented = []
        chunks_with_qa = set()  # Track which original chunks have Q&A versions
        
        # Build a mapping from original chunk text to (idx, metadata, source)
        chunk_lookup = {}
        for idx, (chunk_text, chunk_meta, chunk_source) in enumerate(chunks, 1):
            # Use normalized text as key
            normalized = re.sub(r'\s+', ' ', chunk_text).strip()
            chunk_lookup[normalized] = (idx, chunk_meta, chunk_source)
        
        loaded_count = 0
        skipped_tier = 0
        skipped_program = 0
        skipped_notfound = 0
        
        with open(combined_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                tier = record.get('tier', 2)
                
                # Find matching original chunk to get source URL
                original_text = record['original_text']
                normalized = re.sub(r'\s+', ' ', original_text).strip()
                
                if normalized not in chunk_lookup:
                    skipped_notfound += 1
                    continue
                
                chunk_idx, source_meta, source_source = chunk_lookup[normalized]
                
                # Apply tier OR program filter
                chunk_url = source_source.get('url', '')
                matches_tier = tier_filter == "all" or tier in tier_filter
                matches_program = False
                if program_filter:
                    for program_slug in program_filter:
                        if program_slug in chunk_url:
                            matches_program = True
                            break
                
                if not matches_tier and not matches_program:
                    skipped_tier += 1
                    skipped_program += 1
                    continue
                
                chunk_idx, source_meta, source_source = chunk_lookup[normalized]
                chunks_with_qa.add(chunk_idx)  # Mark this chunk as having Q&A
                
                # Use pre-generated Q&A chunk
                qa_text = record['full_qa_chunk']
                questions = record['questions']
                
                # Create metadata
                qa_meta = source_meta.copy()
                qa_meta['is_synthetic_qa'] = True
                qa_meta['questions'] = questions
                qa_meta['num_questions'] = len(questions)
                
                # Copy source (this is the key fix!)
                qa_source = source_source.copy()
                
                augmented.append((qa_text, qa_meta, qa_source))
                loaded_count += 1
        
        # Add original chunks based on configuration
        originals_added = 0
        if keep_original_chunks:
            # Add ALL original chunks
            for chunk in chunks:
                augmented.append(chunk)
            originals_added = len(chunks)
            print(f"Keeping BOTH original chunks AND Q&A chunks")
        else:
            # Add only originals that DON'T have Q&A versions
            for idx, chunk in enumerate(chunks, 1):
                if idx not in chunks_with_qa:
                    augmented.append(chunk)
                    originals_added += 1
            print(f"Q&A chunks REPLACE originals (originals kept only if no Q&A)")
        
        total_questions = sum(
            meta.get('num_questions', 0) 
            for _, meta, _ in augmented 
            if meta.get('is_synthetic_qa')
        )
        
        print(f"✓ Loaded {loaded_count} pre-generated Q&A chunks")
        print(f"  Total questions: {total_questions}")
        if loaded_count > 0:
            print(f"  Average: {total_questions / loaded_count:.1f} questions per chunk")
        if tier_filter != "all" or program_filter:
            filter_desc = f"tiers {tier_filter}" if tier_filter != "all" else "all tiers"
            if program_filter:
                filter_desc += f" + {len(program_filter)} specific programs"
            print(f"  Filtered {skipped_tier} chunks (not in {filter_desc})")
        if skipped_notfound > 0:
            print(f"  Skipped {skipped_notfound} chunks (no matching original)")
        if not keep_original_chunks:
            print(f"  Added {originals_added} original chunks (no Q&A version)")
            print(f"  Replaced {len(chunks_with_qa)} chunks with Q&A versions")
        else:
            print(f"  Kept all {originals_added} original chunks")
        print()
        
        return augmented
    
    
    def augment_chunks_with_qa(
        self, 
        chunks: List[Tuple[str, Dict, Dict]]
    ) -> List[Tuple[str, Dict, Dict]]:
        """
        For each chunk, generate MULTIPLE synthetic Q&A pairs and add them to the index.
        
        If keep_original_chunks=False:
          - Chunks WITH Q&A: only Q&A version is kept (original replaced)
          - Chunks WITHOUT Q&A: original is kept (e.g., tier 3/4 when only generating for tier 1/2)
        
        Each chunk is a 3-tuple: (text, meta, source)
        
        First checks for pre-generated combined Q&A file to avoid regeneration.
        Falls back to generation if file doesn't exist.
        """
        # Get config
        cfg = get_config()
        qa_cfg = cfg.get("synthetic_qa", {})
        tier_filter = qa_cfg.get("generate_for_tiers", "all")
        program_filter = qa_cfg.get("generate_for_programs", [])
        questions_per_chunk = qa_cfg.get("questions_per_chunk", 3)
        
        # Set up paths
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "synthetic_qa_generated.jsonl"
        
        # Try to load pre-generated Q&A first
        if output_path.exists():
            return self._load_combined_qa(chunks, output_path, tier_filter, program_filter, keep_original_chunks=False)
        
        # GENERATION PATH: Create new Q&A chunks
        augmented = []
        chunks_with_qa = set()  # Track which chunks got Q&A versions
        
        # Open file in write mode (overwrites previous run)
        print(f"\nSaving generated Q&A pairs to: {output_path}")
        
        # Show tier filtering info
        if tier_filter == "all":
            print(f"Generating {questions_per_chunk} Q&A pairs per chunk for ALL chunks (~11,000 chunks)")
            print(f"Estimated time: {6 * questions_per_chunk}-{15 * questions_per_chunk} hours on CPU")
        else:
            print(f"Generating {questions_per_chunk} Q&A pairs per chunk for tiers: {tier_filter}")
            estimated_chunks = 2000  # Rough estimate for tiers [1, 2]
            if program_filter:
                print(f"  + Specific programs: {program_filter}")
                estimated_chunks += len(program_filter) * 100  # ~100 chunks per program
            print(f"Estimated time: {questions_per_chunk}-{3 * questions_per_chunk} hours on CPU (for ~{estimated_chunks} chunks)")
        
        print("Q&A chunks will REPLACE originals (originals kept only if no Q&A generated)")
        print("(You can view this file in real-time as questions are generated)\n")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add synthetic Q&A versions (MULTIPLE questions per chunk)
            generated_count = 0
            skipped_count = 0
            filtered_count = 0
            duplicate_count = 0
            duplicate_chunks_skipped = 0
            
            # Track generated questions to avoid duplicates
            seen_questions = set()
            # Track generated Q&A chunks to avoid duplicates
            seen_qa_chunks = set()
            
            for idx, (chunk_text, chunk_meta, chunk_source) in enumerate(chunks, 1):
                # Skip if already synthetic Q&A (avoid duplicates if run multiple times)
                if chunk_meta.get('is_synthetic_qa'):
                    continue
                
                # Tier filtering
                chunk_tier = chunk_meta.get('tier', 2)
                chunk_url = chunk_source.get('url', '')
                
                # Check if chunk matches tier filter OR program filter
                matches_tier = tier_filter == "all" or chunk_tier in tier_filter
                matches_program = False
                
                if program_filter:
                    # Check if URL contains any of the program slugs
                    for program_slug in program_filter:
                        if program_slug in chunk_url:
                            matches_program = True
                            break
                
                # Skip if doesn't match either filter
                if not matches_tier and not matches_program:
                    filtered_count += 1
                    continue
                
                context_title = chunk_meta.get('title', '')
                questions = self.generate_question_for_chunk(chunk_text, context_title, num_questions=questions_per_chunk)
                
                if questions:
                    # Filter duplicate questions within this chunk
                    unique_questions = []
                    for question in questions:
                        normalized_q = question.lower().strip()
                        if normalized_q not in seen_questions:
                            seen_questions.add(normalized_q)
                            unique_questions.append(question)
                        else:
                            duplicate_count += 1
                    
                    if unique_questions:
                        # Create ONE Q&A chunk with ALL unique questions
                        qa_chunk, qa_meta = self.create_qa_chunk(chunk_text, unique_questions, chunk_meta)
                        # Copy the source from original chunk
                        qa_source = chunk_source.copy()
                        
                        # Deduplicate Q&A chunks (check if we've seen this Q&A text before)
                        qa_chunk_key = qa_chunk.lower().strip()
                        if qa_chunk_key in seen_qa_chunks:
                            duplicate_chunks_skipped += 1
                            continue
                        
                        seen_qa_chunks.add(qa_chunk_key)
                        augmented.append((qa_chunk, qa_meta, qa_source))
                        chunks_with_qa.add(idx)  # Mark this chunk as having Q&A
                        generated_count += len(unique_questions)
                        
                        # Save to file immediately (incremental)
                        record = {
                            "index": len(augmented),  # Index of this Q&A chunk
                            "original_chunk_index": idx,
                            "tier": chunk_tier,
                            "questions": unique_questions,  # All questions for this chunk
                            "num_questions": len(unique_questions),
                            "original_text": chunk_text,
                            "title": context_title,
                            "full_qa_chunk": qa_chunk
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        f.flush()  # Force write to disk immediately
                        
                        # Progress indicator
                        if generated_count % 10 == 0:
                            latest_questions = ", ".join([q[:30] + "..." for q in unique_questions[:2]])
                            print(f"  Generated {generated_count} questions so far... (latest: {latest_questions})")
                else:
                    skipped_count += 1
        
        # Add original chunks that don't have Q&A versions
        originals_added = 0
        for idx, chunk in enumerate(chunks, 1):
            if idx not in chunks_with_qa:
                augmented.append(chunk)
                originals_added += 1
        
        # Count total Q&A chunks created (not individual questions)
        qa_chunks_created = len([meta for _, meta, _ in augmented if meta.get('is_synthetic_qa')])
        
        print(f"\n✓ Generated {qa_chunks_created} combined Q&A chunks ({generated_count} total questions)")
        print(f"  From {len(chunks)} original chunks")
        if qa_chunks_created > 0:
            print(f"  Average {generated_count / qa_chunks_created:.1f} questions per combined chunk")
        if tier_filter != "all" or program_filter:
            filter_desc = f"tiers {tier_filter}" if tier_filter != "all" else "all tiers"
            if program_filter:
                filter_desc += f" + {len(program_filter)} specific programs"
            print(f"  (Filtered {filtered_count} chunks - not in {filter_desc})")
        print(f"  (Skipped {skipped_count} chunks - too short or unsuitable for Q&A)")
        print(f"  (Skipped {duplicate_count} duplicate questions)")
        if duplicate_chunks_skipped > 0:
            print(f"  (Skipped {duplicate_chunks_skipped} duplicate Q&A chunks)")
        print(f"  Added {originals_added} original chunks (no Q&A version generated)")
        print(f"  Replaced {len(chunks_with_qa)} original chunks with Q&A versions")
        print(f"  View results in: {output_path}\n")
        return augmented


# Global instance
_qa_generator = None

def get_qa_generator() -> SyntheticQAGenerator:
    global _qa_generator
    if _qa_generator is None:
        _qa_generator = SyntheticQAGenerator()
    return _qa_generator
