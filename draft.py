import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load data
data = pd.read_csv("your_data.csv")  # Ensure it has text, label (Israel/Palestine), and sentiment columns

# Combine labels
def combine_labels(row):
    if row['sentiment'] == 'neutral':
        return 'Neutral'
    elif row['label'] == 'Israel' and row['sentiment'] == 'positive':
        return 'Pro-Israel'
    elif row['label'] == 'Israel' and row['sentiment'] == 'negative':
        return 'Anti-Israel'
    elif row['label'] == 'Palestine' and row['sentiment'] == 'positive':
        return 'Pro-Palestine'
    elif row['label'] == 'Palestine' and row['sentiment'] == 'negative':
        return 'Anti-Palestine'

data['combined_label'] = data.apply(combine_labels, axis=1)

# Feature extraction (using embeddings)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

data['features'] = data['text'].apply(embed_text)

# Train-test split
X = list(data['features'])
y = data['combined_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Predict for new data
new_sentence = "Your new sentence here."
new_features = embed_text(new_sentence)
prediction = clf.predict([new_features])
print(f"Prediction: {prediction[0]}")


#TODO
# divide into 5 catagories
# make sure there is around the same number of each (100)
# find a bert model
# run tokenizer... etc and the model
# fine tuning

