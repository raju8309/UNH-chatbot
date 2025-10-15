import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# Add backend path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import main
from main import load_retrieval_cfg, load_initial_data, get_context, get_prompt


def create_training_data():
    """Generate training data using existing retrieval pipeline"""
    training_examples = []

    gold_path = ROOT / "automation_testing" / "gold.jsonl"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    print("Creating training data from gold.jsonl...")

    with open(gold_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question = record["query"]
                expected_answer = record["reference_answer"]

                _, _, context = get_context(question)
                training_examples.append({
                    "input": get_prompt(question, context),
                    "output": expected_answer,
                    "question": question  # For debugging
                })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue

    print(f"Created {len(training_examples)} training examples")
    return training_examples


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Tokenize the training examples"""
    inputs = examples["input"]
    targets = examples["output"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding=True
    )

    # Replace padding token ids with -100 for loss computation
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("Loading data...")
    load_retrieval_cfg()
    load_initial_data()

    training_examples = create_training_data()
    train_examples, val_examples = train_test_split(training_examples, test_size=0.2, random_state=42)

    print(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")

    model_name = "google/flan-t5-small"
    print(f"Loading model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Create and tokenize datasets
    train_dataset = Dataset.from_list(train_examples).map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_examples[0].keys()
    )
    val_dataset = Dataset.from_list(val_examples).map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=val_examples[0].keys()
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    output_dir = ROOT / "backend" / "models" / "flan-t5-small-finetuned"

    # GPU/CPU optimized settings
    if torch.cuda.is_available():
        train_batch_size, eval_batch_size = 16, 16
        gradient_accumulation_steps = 1
        fp16 = True
        dataloader_num_workers = 4
        pin_memory = True
    else:
        train_batch_size, eval_batch_size = 4, 4
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Moving model to CPU for deployment...")
    model = model.to("cpu")
    print(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final validation loss: {eval_results['eval_loss']:.4f}")

    print("\n=== Testing trained model (CPU mode) ===")
    model.eval()
    for i, example in enumerate(val_examples[:3]):
        inputs = tokenizer(example["input"], return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nExample {i+1}:\nQuestion: {example['question']}\nExpected: {example['output']}\nPredicted: {predicted}")
        print("-" * 50)

    print(f"\nTraining complete! Model saved to: {output_dir}")
