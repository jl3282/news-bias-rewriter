#!/usr/bin/env python3
"""
Evaluate the fine-tuned RoBERTa model on test set
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data():
    """Load and prepare test data (same split as training)"""
    print("📊 Loading test data...")
    
    # Load data - same process as training
    df = pd.read_csv('../data/allsides_balanced_news_headlines-texts.csv')
    df = df.dropna(subset=['text', 'bias_rating'])
    df = df[df['text'].str.len() > 10]
    
    # Encode labels - same as training
    label_encoder = LabelEncoder()
    label_encoder.fit(df['bias_rating'])
    labels = label_encoder.transform(df['bias_rating'])
    
    # Same splits as training (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['text'].values, labels, test_size=0.3, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"Test set size: {len(X_test)} articles")
    return X_test, y_test, label_encoder

def evaluate_model():
    """Evaluate the fine-tuned model"""
    print("🚀 Evaluating Fine-tuned RoBERTa Model")
    print("=" * 50)
    
    # Load model and tokenizer
    model_path = '../models/roberta-bias-classifier'
    print(f"🤖 Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Load test data
    X_test, y_test, label_encoder = load_test_data()
    
    # Load label mapping
    with open(f'{model_path}/label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    print(f"📝 Label mapping: {label_mapping['id2label']}")
    
    # Make predictions
    print("🔍 Making predictions on test set...")
    predictions = []
    
    with torch.no_grad():
        for i, text in enumerate(X_test):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(X_test)}")
            
            # Tokenize
            inputs = tokenizer(text, truncation=True, padding='max_length', 
                             max_length=512, return_tensors='pt')
            
            # Predict
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average=None)
    
    # Macro averages
    precision_macro = precision_recall_fscore_support(y_test, predictions, average='macro')[0]
    recall_macro = precision_recall_fscore_support(y_test, predictions, average='macro')[1]
    f1_macro = precision_recall_fscore_support(y_test, predictions, average='macro')[2]
    
    # Weighted averages
    precision_weighted = precision_recall_fscore_support(y_test, predictions, average='weighted')[0]
    recall_weighted = precision_recall_fscore_support(y_test, predictions, average='weighted')[1]
    f1_weighted = precision_recall_fscore_support(y_test, predictions, average='weighted')[2]
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Print results
    print(f"\n🎯 FINAL TEST RESULTS")
    print(f"=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 (Macro): {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro): {recall_macro:.4f}")
    
    print(f"\n📊 Per-Class Results:")
    bias_labels = list(label_encoder.classes_)
    for i, label in enumerate(bias_labels):
        print(f"  {label.upper()}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1-Score: {f1[i]:.4f}")
        print(f"    Support: {support[i]}")
    
    # Confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=bias_labels, yticklabels=bias_labels)
    plt.title('Fine-tuned RoBERTa - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"roberta_finetuned_final_results_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📈 Results plot saved as {plot_filename}")
    
    # Save results
    results = {
        'timestamp': timestamp,
        'model_path': model_path,
        'test_size': len(y_test),
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'per_class_precision': [float(p) for p in precision],
        'per_class_recall': [float(r) for r in recall],
        'per_class_f1': [float(f) for f in f1],
        'per_class_support': [int(s) for s in support],
        'confusion_matrix': cm.tolist(),
        'label_mapping': label_mapping
    }
    
    # Ensure results directory exists
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = f'{results_dir}/roberta_finetuned_final_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = evaluate_model()
    print(f"\n✅ Evaluation completed!")
    print(f"🏆 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"🏆 Final Test F1 (Macro): {results['f1_macro']:.4f}") 