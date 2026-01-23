import os

# Set this BEFORE importing torch/transformers
# "0" refers to the specific GPU ID you want to use.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# --- CONFIGURATION ---
MODEL_NAME = "answerdotai/ModernBERT-base"
TRAIN_FILE = "bert_training_data.csv"    # The balanced Training CSV
VAL_FILE = "bert_val_data.csv"           # The dedicated Validation CSV
OUTPUT_DIR = "/data2/brain2text/lm/modernbert_domain_classifier"
MAX_LENGTH = 256
BATCH_SIZE = 64
EPOCHS = 10  # Increased slightly so you can see saturation; EarlyStopping can also be added
LEARNING_RATE = 2e-5

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 1. Load Data
    print(f"Loading Training data from {TRAIN_FILE}...")
    train_df = pd.read_csv(TRAIN_FILE)
    
    print(f"Loading Validation data from {VAL_FILE}...")
    eval_df = pd.read_csv(VAL_FILE)
    
    # Ensure labels are integers
    train_df['label'] = train_df['label'].astype(int)
    eval_df['label'] = eval_df['label'].astype(int)
    
    # Convert to Hugging Face Datasets
    # (No random splitting here; we use the files exactly as prepared)
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # 2. Tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding=False,    # We will pad dynamically in the collator
            truncation=True, 
            max_length=MAX_LENGTH
        )

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # 3. Model Setup
    print("Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, # 1 = In-Domain, 0 = Out-of-Domain
    )
    
    # 4. Trainer Setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",      # Evaluate at end of every epoch
        save_strategy="epoch",      # Save checkpoint at end of every epoch
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True, # Will load the checkpoint with highest accuracy
        metric_for_best_model="accuracy", 
        save_total_limit=1,          # Only keep the top 2 checkpoints to save disk space
        fp16=torch.cuda.is_available(), 
        report_to="none"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    print("Starting Training...")
    trainer.train()

    # 6. Save Final Model (The "best" model loaded at the end)
    print(f"Saving best model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Final Eval
    metrics = trainer.evaluate()
    print("\nFinal Metrics on Validation Set:")
    print(metrics)

if __name__ == "__main__":
    main()