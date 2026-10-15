import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW

def run_comparative_study(csv_path='spam.csv'):
    print("="*60)
    print("STEP 1: DATA LOADING & EXPLORATION")
    print("="*60)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found. Please place the dataset CSV in the working directory.")
        return

    # Standardize columns dynamically
    text_col, label_col = None, None
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'message' in col_lower:
            text_col = col
        elif 'label' in col_lower or 'spam' in col_lower or 'target' in col_lower:
            label_col = col

    if not text_col or not label_col:
        # Fallback to standard columns if renaming fails
        text_col = df.columns[0]
        label_col = df.columns[1]

    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)

    print(f"Dataset Shape: {df.shape}")
    print("Class Distribution:\n", df['label'].value_counts(normalize=True))

    # Stratified Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['label'].values, 
        test_size=0.2, random_state=42, stratify=df['label'].values
    )

    results = {}

    print("\n" + "="*60)
    print("STEP 2: PART A - TRADITIONAL MACHINE LEARNING PIPELINE")
    print("="*60)

    # TF-IDF Feature Extraction
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    trad_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    for name, model in trad_models.items():
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test_tfidf)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Training Time (s)": train_time,
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        }
        print(f"[{name}] Done in {train_time:.2f}s | F1-Score: {results[name]['F1-Score']:.4f}")

    print("\n" + "="*60)
    print("STEP 3: PART B - TRANSFORMER MODEL (DISTILBERT & PYTORCH)")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # Subsample for practical execution speed in a single script run (adjust or remove slice for full dataset)
    max_train = min(1500, len(X_train))
    max_test = min(300, len(X_test))
    X_tr_bert, y_tr_bert = X_train[:max_train], y_train[:max_train]
    X_te_bert, y_te_bert = X_test[:max_test], y_test[:max_test]

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.to(device)

    class SpamDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            encoding = self.tokenizer.encode_plus(
                str(self.texts[idx]),
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(int(self.labels[idx]), dtype=torch.long)
            }

    train_loader = DataLoader(SpamDataset(X_tr_bert, y_tr_bert, tokenizer), batch_size=16, shuffle=True)
    test_loader = DataLoader(SpamDataset(X_te_bert, y_te_bert, tokenizer), batch_size=16, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 2
    epoch_losses = []

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    distilbert_time = time.time() - start_time

    # Evaluation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            batch_preds = torch.argmax(outputs.logits, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            actuals.extend(batch['labels'].numpy())

    results["DistilBERT"] = {
        "Accuracy": accuracy_score(actuals, preds),
        "Precision": precision_score(actuals, preds, zero_division=0),
        "Recall": recall_score(actuals, preds, zero_division=0),
        "F1-Score": f1_score(actuals, preds, zero_division=0),
        "Training Time (s)": distilbert_time,
        "Confusion Matrix": confusion_matrix(actuals, preds)
    }
    print(f"[DistilBERT] Done in {distilbert_time:.2f}s | F1-Score: {results['DistilBERT']['F1-Score']:.4f}")

    print("\n" + "="*60)
    print("STEP 4: GENERATING & SAVING VISUALIZATIONS")
    print("="*60)

    sns.set_theme(style="whitegrid")
    model_names = list(results.keys())
    accuracies = [res["Accuracy"] for res in results.values()]
    f1_scores = [res["F1-Score"] for res in results.values()]
    training_times = [res["Training Time (s)"] for res in results.values()]

    # 1. Metric Comparison Chart
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#4C72B0')
    ax.bar(x + width/2, f1_scores, width, label='F1-Score', color='#55A868')
    ax.set_ylabel('Score')
    ax.set_title('Comparative Performance Metrics Across Models')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0.7, 1.02)
    plt.savefig('metric_comparison.png', bbox_inches='tight')
    plt.close()

    # 2. Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, res) in enumerate(results.items()):
        sns.heatmap(res["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', bbox_inches='tight')
    plt.close()

    # 3. Training Time / Computational Cost Comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=model_names, y=training_times, palette="viridis", ax=ax)
    ax.set_ylabel('Training Time (Seconds)')
    ax.set_title('Computational Cost: Training Time Comparison')
    plt.savefig('computational_cost.png', bbox_inches='tight')
    plt.close()

    print("Successfully generated and saved:")
    print(" - metric_comparison.png")
    print(" - confusion_matrices.png")
    print(" - computational_cost.png")
    print("="*60)

if __name__ == "__main__":
    run_comparative_study('spam.csv')