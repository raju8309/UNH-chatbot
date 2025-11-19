"""
Fine-tune embedding model on MS MARCO passage ranking dataset.
This will improve retrieval quality by training the model to better distinguish
relevant from irrelevant passages.

MS MARCO dataset: https://huggingface.co/datasets/ms_marco
"""

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import os

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(4)  # Limit CPU threads to avoid overload

# Configuration
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Your current model
OUTPUT_MODEL_PATH = "../models/finetuned-minilm-msmarco"
BATCH_SIZE = 16
EPOCHS = 3
MAX_TRAIN_SAMPLES = 100000  # MS MARCO is huge, use subset for speed
MAX_VAL_SAMPLES = 1000
WARMUP_STEPS = 1000


def load_ms_marco_data(split="train", max_samples=None):
    """
    Load MS MARCO passage ranking dataset.
    
    Returns:
        List of InputExample with (query, positive_passage, negative_passage)
    """
    print(f"Loading MS MARCO {split} split...")
    
    # Load from HuggingFace
    # Using 'v1.1' which has query, positive passages, and negative passages
    dataset = load_dataset("ms_marco", "v1.1", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    examples = []
    skipped = 0
    
    print(f"Processing {len(dataset)} examples...")
    for item in tqdm(dataset):
        try:
            query = item['query']
            passages = item['passages']
            
            # Check if we have passages and is_selected data
            if not passages or 'is_selected' not in passages or 'passage_text' not in passages:
                skipped += 1
                continue
            
            is_selected = passages['is_selected']
            passage_texts = passages['passage_text']
            
            # Find positive passage (is_selected == 1 or True)
            positive_indices = [i for i, selected in enumerate(is_selected) if selected == 1 or selected == True]
            negative_indices = [i for i, selected in enumerate(is_selected) if selected == 0 or selected == False]
            
            # Need both positive and negative passages
            if not positive_indices or not negative_indices:
                skipped += 1
                continue
            
            positive_passage = passage_texts[positive_indices[0]]
            negative_passage = passage_texts[negative_indices[0]]
            
            # Skip if passages are empty
            if not positive_passage or not negative_passage or not query:
                skipped += 1
                continue
            
            examples.append(InputExample(
                texts=[query, positive_passage, negative_passage]
            ))
        except (KeyError, IndexError, TypeError) as e:
            skipped += 1
            continue
    
    print(f"Created {len(examples)} training examples ({skipped} skipped)")
    return examples


def load_catalog_data_as_validation():
    """
    Use your gold set as validation data to ensure model learns
    catalog-specific patterns.
    
    Returns:
        Dict for InformationRetrievalEvaluator
    """
    gold_path = Path(__file__).parent.parent.parent / "automation_testing" / "gold.jsonl"
    
    if not gold_path.exists():
        print(f"Warning: Gold set not found at {gold_path}")
        return None
    
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    print("Loading catalog validation data from gold set...")
    with open(gold_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item['id']
            query = item['query']
            answer = item.get('answer', '')
            
            # Use query ID as key
            queries[qid] = query
            
            # Use answer as relevant document
            corpus[qid] = answer
            
            # Mark this document as relevant for this query
            relevant_docs[qid] = {qid}
    
    print(f"Loaded {len(queries)} validation queries")
    return {
        'queries': queries,
        'corpus': corpus,
        'relevant_docs': relevant_docs
    }


def create_hybrid_validation():
    """
    Combine MS MARCO validation with catalog data for balanced evaluation.
    """
    # Load a small MS MARCO validation set
    ms_marco_examples = load_ms_marco_data(split="validation", max_samples=MAX_VAL_SAMPLES)
    
    # Convert to evaluator format
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for i, example in enumerate(ms_marco_examples):
        qid = f"msmarco_{i}"
        query = example.texts[0]
        positive_doc = example.texts[1]
        
        queries[qid] = query
        corpus[f"{qid}_pos"] = positive_doc
        relevant_docs[qid] = {f"{qid}_pos"}
    
    # Add catalog data
    catalog_data = load_catalog_data_as_validation()
    if catalog_data:
        queries.update(catalog_data['queries'])
        corpus.update(catalog_data['corpus'])
        relevant_docs.update(catalog_data['relevant_docs'])
    
    return {
        'queries': queries,
        'corpus': corpus,
        'relevant_docs': relevant_docs
    }


def main():
    """
    Fine-tune the embedding model on MS MARCO data.
    """
    print("=" * 80)
    print("MS MARCO Fine-tuning for Retrieval")
    print("=" * 80)
    
    # Load base model
    print(f"\nLoading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL, device='cpu')  # Force CPU
    print("Model loaded on CPU")
    
    # Load training data
    train_examples = load_ms_marco_data(split="train", max_samples=MAX_TRAIN_SAMPLES)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # Define loss function
    # MultipleNegativesRankingLoss: given (query, positive, negative), 
    # trains model to score positive higher than negative
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create evaluator
    print("\nPreparing validation data...")
    val_data = create_hybrid_validation()
    
    evaluator = InformationRetrievalEvaluator(
        queries=val_data['queries'],
        corpus=val_data['corpus'],
        relevant_docs=val_data['relevant_docs'],
        name='msmarco-catalog-eval'
    )
    
    # Calculate training steps
    num_train_steps = len(train_dataloader) * EPOCHS
    
    print("\n" + "=" * 80)
    print("Training Configuration:")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  Training Examples: {len(train_examples)}")
    print(f"  Validation Queries: {len(val_data['queries'])}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Total Steps: {num_train_steps}")
    print(f"  Warmup Steps: {WARMUP_STEPS}")
    print(f"  Output Path: {OUTPUT_MODEL_PATH}")
    print("=" * 80)
    
    # Train
    print("\nStarting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=OUTPUT_MODEL_PATH,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader) // 4,  # Evaluate 4 times per epoch
        save_best_model=True,
        show_progress_bar=True
    )
    
    print("\n" + "=" * 80)
    print(f"Training complete! Model saved to: {OUTPUT_MODEL_PATH}")
    print("=" * 80)
    
    # Save training metadata
    metadata = {
        'base_model': BASE_MODEL,
        'dataset': 'ms_marco',
        'train_samples': len(train_examples),
        'val_samples': len(val_data['queries']),
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    
    metadata_path = Path(OUTPUT_MODEL_PATH) / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTo use this model, update backend/config/settings.py:")
    print(f'  EMBED_MODEL_NAME = "{OUTPUT_MODEL_PATH}"')


if __name__ == "__main__":
    main()
