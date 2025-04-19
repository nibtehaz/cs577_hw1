import torch
import numpy as np
import os
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate


# Load WNUT dataset
dataset = load_dataset("wnut_17")

# Load tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

# Model save path
model_save_path = "./wnut_trained_model"


# Tokenize dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        padding="max_length",
        truncation=True,
        max_length=128,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        new_labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                new_labels.append(-100)  # Ignore padding tokens
            elif word_id != previous_word_id:
                new_labels.append(label[word_id])  # Assign label to first subword
            else:
                new_labels.append(-100)  # Ignore subwords
            previous_word_id = word_id
        labels.append(new_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Load evaluation metric
metric = evaluate.load("seqeval")


# Compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [dataset["train"].features["ner_tags"].feature.names[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [dataset["train"].features["ner_tags"].feature.names[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"macro_f1": results["overall_f1"]}


# Load model (if saved) or train from scratch
if os.path.exists(model_save_path):
    print("Loading pre-trained model from disk...")
    model = BertForTokenClassification.from_pretrained(model_save_path)
else:
    print("Training new model...")
    model = BertForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(
        dataset["train"].features["ner_tags"].feature.names))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        metric_for_best_model="macro_f1",
        load_best_model_at_end=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

# Manually compute F1 on test set
print("Evaluating on test set...")
test_trainer = Trainer(model)
raw_test_predictions = test_trainer.predict(tokenized_datasets["test"])

import pickle 
pickle.dump(raw_test_predictions,open('raw_test_predictions.p','wb'))

# Compute F1 score
test_metrics = compute_metrics((raw_test_predictions.predictions, raw_test_predictions.label_ids))
print(f"Test Set Macro F1 Score: {test_metrics['macro_f1']:.4f}")

