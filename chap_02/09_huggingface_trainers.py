import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset


torch.cuda.empty_cache()

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


dataset = load_dataset("imdb")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",  # Output directory
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
    learning_rate=2e-5,  # Learning rate
    per_device_train_batch_size=8,  # Batch size for training
    num_train_epochs=3,  # Number of epochs
    weight_decay=0.01,  # Weight decay
    logging_dir="./logs",  # Log directory
)
trainer = Trainer(
    model=model,  # Your DistilBERT model
    args=training_args,  # Training arguments
    train_dataset=tokenized_datasets["train"],  # Training dataset
    eval_dataset=tokenized_datasets["test"],  # Evaluation dataset
)
trainer.train()
