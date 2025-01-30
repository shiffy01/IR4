import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Disable W&B logging
os.environ["WANDB_MODE"] = "disabled"

# Load the dataset
file_path = "/content/shuffled_file.xlsx"
df = pd.read_excel(file_path)

# Map sentiments to numerical labels
label_mapping = {
    "anti-i": 0,
    "pro-p": 1,
    "neutral": 2,
    "anti-p": 3,
    "pro-i": 4
}
df['label encoded'] = df['label'].map(label_mapping)

# Split into training, validation, and testing sets
train_df, test_df = train_test_split(df, test_size=0.18, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

print(f"Training set size: {train_df.shape[0]}")
print(f"Validation set size: {val_df.shape[0]}")
print(f"Testing set size: {test_df.shape[0]}")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize datasets
def tokenize_data(df):
    return tokenizer(
        df['Sentence'].tolist(), truncation=True, padding=True, max_length=256
    )

train_encodings = tokenize_data(train_df)
val_encodings = tokenize_data(val_df)
test_encodings = tokenize_data(test_df)

# Define PyTorch dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare datasets
train_dataset = SentimentDataset(train_encodings, train_df['label encoded'].tolist())
val_dataset = SentimentDataset(val_encodings, val_df['label encoded'].tolist())
test_dataset = SentimentDataset(test_encodings, test_df['label encoded'].tolist())

# Load pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results', # Directory for storing outputs
    num_train_epochs=10,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training per device (per GPU/CPU)
    per_device_eval_batch_size=16,  # Batch size for evaluation per device
    warmup_steps=500,  # Number of warm-up steps for the learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay (L2 regularization) to prevent overfitting
    logging_dir='./logs',  # Directory for storing training logs
    logging_steps=10,  # Log training metrics every 10 steps
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    save_strategy="epoch",  # Save model checkpoints at the end of each epoch
    load_best_model_at_end=True,  # Load the best model at the end of training based on the metric
    metric_for_best_model="f1"  # Metric used to determine the best model (e.g., F1-score)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
eval_results = trainer.evaluate(test_dataset)
print("Evaluation Results on Test Set:", eval_results)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert")
tokenizer.save_pretrained("fine_tuned_bert")

# Prediction function
def predict(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits).item()


# Add predicted labels to the DataFrame
df['Predicted Label'] = df['Sentence'].apply(predict)

# Save the updated DataFrame to a new Excel file
output_file_path = "/content/with_prediction.xlsx"
df.to_excel(output_file_path, index=False)

print(f"Updated Excel file saved to {output_file_path}")
