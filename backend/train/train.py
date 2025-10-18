import json
import sys
import torch
import textwrap
from pathlib import Path
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import load_retrieval_config
from models.ml_models import initialize_models
from services.chunk_service import load_initial_data
from services.qa_service import get_context
from services.qa_service import get_prompt

def create_training_data():
    """Generate training data using existing retrieval pipeline"""
    training_examples = []
    verification_data = []
    
    # Load the training data
    print("Creating training data from train.json...")
    train_set = ROOT / "train" / "train.json"
    if not train_set.exists():
        raise FileNotFoundError(f"Train file not found: {train_set}")

    with open(train_set, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for i, record in enumerate(data):
        try:
            question = record["query"]
            _, _, context = get_context(question)
            training_examples.append({
                "input": get_prompt(question, context),
                "output": record["answer"],
                "question": question
            })
            verification_data.append({
                "output": record["answer"],
                "question": question,
                "context": context
            })
        except json.JSONDecodeError as e:
            print(f"Error parsing record {i}: {e}")
            continue
        except KeyError as e:
            print(f"Missing field in record {i}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error in record {i}: {e}")
            continue

    verification_file = ROOT / "train" / "verify.txt"
    with open(verification_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(verification_data):
            # Wrap question
            f.write("QUESTION:\n")
            wrapped_question = textwrap.fill(item['question'], width=80, initial_indent="", subsequent_indent="")
            f.write(f"{wrapped_question}\n\n")
            # Wrap answer
            f.write("ANSWER:\n")
            wrapped_answer = textwrap.fill(item['output'], width=80, initial_indent="", subsequent_indent="")
            f.write(f"{wrapped_answer}\n\n")
            # Wrap context
            f.write("CONTEXT:\n")
            if item['context']:
                # Split context by double newlines
                context_entries = item['context'].split('\n\n')
                for j, entry in enumerate(context_entries):
                    if entry.strip():
                        wrapped_entry = textwrap.fill(entry.strip(), width=80, initial_indent="", subsequent_indent="")
                        f.write(f"[{j+1}] {wrapped_entry}\n\n")
            else:
                f.write("No context found\n\n")

            f.write("="*80 + "\n\n")
    
    print(f"Created {len(training_examples)} training examples")
    return training_examples

def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Tokenize the training examples"""
    inputs = examples["input"]
    targets = examples["output"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding=True
    )
    # Tokenize targets
    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length, 
        truncation=True, 
        padding=True
    )
    
    # Replace padding token ids with -100 (ignored by loss function)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in labels["input_ids"]
    ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    # Check GPU availability for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected, training on CPU")
    
    # Load data
    print("Loading data...")
    load_retrieval_config()
    initialize_models()
    load_initial_data()
    
    # Create training data
    training_examples = create_training_data()
    # Split into train/validation
    train_examples, val_examples = train_test_split(
        training_examples, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    # Load model and tokenizer
    model_name = "google/flan-t5-small"
    print(f"Loading model: {model_name}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU for training (if available)
    model = model.to(device)
    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    # Training arguments - optimized for GPU training
    output_dir = ROOT / "train" / "models" / "flan-t5-small-finetuned"
    # Use GPU-optimized settings if available
    if torch.cuda.is_available():
        train_batch_size = 16     # Larger batch for GPU
        eval_batch_size = 16
        gradient_accumulation_steps = 1
        fp16 = True              # Mixed precision for faster training
        dataloader_num_workers = 4
        pin_memory = True
    else:
        train_batch_size = 4     # Smaller batch for CPU
        eval_batch_size = 4
        gradient_accumulation_steps = 4
        fp16 = False
        dataloader_num_workers = 0
        pin_memory = False
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=25,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        num_train_epochs=5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=1e-4,
        fp16=fp16,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        report_to=None,
        seed=42,
    )
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # Train the model
    print("Starting training...")
    trainer.train()
    # Move model back to CPU before saving for deployment compatibility
    print("Moving model to CPU for deployment...")
    model = model.to("cpu")
    # Save the final model (on CPU)
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    # Evaluate on validation set
    print("Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
    # Test a few examples (on CPU to match deployment)
    print("\n=== Testing trained model (CPU mode for deployment) ===")
    model.eval()
    for i, example in enumerate(val_examples[:3]):
        inputs = tokenizer(
            example["input"], 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )  # Keep on CPU
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=128,
                pad_token_id=tokenizer.eos_token_id
            )
        
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\nExample {i+1}:")
        print(f"Question: {example['question']}")
        print(f"Expected: {example['output']}")
        print(f"Predicted: {predicted}")
        print("-" * 50)
    print(f"\nTraining complete! Model saved to: {output_dir}")
