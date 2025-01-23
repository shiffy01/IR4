# from random import random
#
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# file_path = "shuffled_file.xlsx"  # Replace with your file path
#
# # Read the Excel file (assumes there's only one sheet)
# df = pd.read_excel(file_path)
#
# # Shuffle the DataFrame
# df = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
#
# # Optionally save the shuffled DataFrame back to a file
# df.to_excel("shuffled_file.xlsx", index=False)
#
#
# print("Combined DataFrame shape:", df.shape)
# print(df.head(3))
#
# # Map sentiments to numerical labels
# label_mapping = {
#     "anti-i" : 0,
#     "pro-p" : 1,
#     "neutral" : 2,
#     "anti-p" : 3,
#     "pro-i" : 4
# }
# df['label encoded'] = df['label'].map(label_mapping)
# print(df.head(3))
# # Split into training and validation sets
# train_df, test_df = train_test_split(df, test_size=0.18, random_state=42)
#
# # Tokenize text data
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# def tokenize_function(examples):
#     return tokenizer(df['Sentence'], padding="max_length", truncation=True, max_length=256)
#
# train_encodings = tokenizer(train_df['Sentence'].tolist(), truncation=True, padding=True, max_length=256)
# val_encodings = tokenizer(test_df['Sentence'].tolist(), truncation=True, padding=True, max_length=256)
#
# print("######################################################")
#
# # Check the first tokenized example to make sure it worked
# print("First tokenized sentence (input_ids):", train_encodings['input_ids'][0])
# print("First tokenized sentence (attention_mask):", train_encodings['attention_mask'][0])
#
# # Prepare PyTorch dataset
# class SentimentDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#
#     def __len__(self):
#         return len(self.labels)
#
# train_dataset = SentimentDataset(train_encodings, train_df['label'].tolist())
# test_dataset = SentimentDataset(val_encodings, test_df['label'].tolist())
#
# # Load pre-trained BERT model with a classification head
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
#
# # Compute metrics for evaluation
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
#     labels = labels.numpy()
#     accuracy = accuracy_score(labels, predictions)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
#     return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
#
# # Set up training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=32,
#     per_device_eval_batch_size=32,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1"
# )
#
# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# # Train the model
# trainer.train()
#
# # Evaluate the model
# eval_results = trainer.evaluate()
# print("Evaluation Results:", eval_results)
#
# # Save the fine-tuned model
# model.save_pretrained("fine_tuned_bert")
# tokenizer.save_pretrained("fine_tuned_bert")
#
# # Function for predictions
# def predict(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     return torch.argmax(logits).item()
#
# # Example prediction
# example_text = "israel is the best country to live in."
# predicted_label = predict(example_text)
# print(f"Predicted Label: {predicted_label}")

from random import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
file_path = "shuffled_file.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Shuffle the combined DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Optionally save the shuffled DataFrame back to a file
df.to_excel("shuffled_file.xlsx", index=False)

print("Combined DataFrame shape:", df.shape)
print(df.head(3))

# Map sentiments to numerical labels
label_mapping = {
    "anti-i": 0,
    "pro-p": 1,
    "neutral": 2,
    "anti-p": 3,
    "pro-i": 4
}
df['label encoded'] = df['label'].map(label_mapping)
print(df.head(3))

# Split into training, validation, and testing sets
train_df, test_df = train_test_split(df, test_size=0.18, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

print(f"Training set size: {train_df.shape[0]}")
print(f"Validation set size: {val_df.shape[0]}")
print(f"Testing set size: {test_df.shape[0]}")

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize datasets
train_encodings = tokenizer(train_df['Sentence'].tolist(), truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_df['Sentence'].tolist(), truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_df['Sentence'].tolist(), truncation=True, padding=True, max_length=256)

# Prepare PyTorch dataset
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

train_dataset = SentimentDataset(train_encodings, train_df['label encoded'].tolist())
val_dataset = SentimentDataset(val_encodings, val_df['label encoded'].tolist())
test_dataset = SentimentDataset(test_encodings, test_df['label encoded'].tolist())

# Load pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1).numpy()
    labels = labels.numpy()
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
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

# Function for predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits).item()

# Example prediction
example_text = "Israel is the best country to live in."
predicted_label = predict(example_text)
print(f"Predicted Label: {predicted_label}")
