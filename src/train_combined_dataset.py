#!/usr/bin/env python3
"""
Combined Dataset Training: AllSides + JSON Dataset
Trains RoBERTa from scratch on 58K+ articles with balanced classes
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class NewsDataset(Dataset):
    """Dataset for news bias classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Research-optimized tokenization for political bias detection
        encoding = self.tokenizer(
            text,
            padding='max_length',        # Ensures uniform batch processing
            max_length=self.max_length,  # 512 tokens optimal for news articles
            truncation=True,             # Handle articles > 512 tokens
            return_tensors='pt',         # PyTorch tensors
            return_attention_mask=True,  # Required for transformer models
            return_token_type_ids=False  # RoBERTa doesn't use token type IDs
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CombinedDatasetTrainer:
    """Trainer for combined AllSides + JSON dataset."""
    
    def __init__(self, model_name='FacebookAI/roberta-base', force_cpu=False):
        self.model_name = model_name
        
        # Device detection
        if force_cpu:
            self.device = torch.device('cpu')
            print("💻 Using CPU for training (forced)")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🚀 MPS (Apple Silicon GPU) detected and will be used!")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 CUDA GPU detected and will be used!")
        else:
            self.device = torch.device('cpu')
            print("💻 Using CPU for training")
        
        # Load fresh model
        print(f"🤖 Loading fresh {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3
        )
        self.model.to(self.device)
        
        print(f"✅ Fresh model loaded on device: {self.device}")
    
    def load_allsides_data(self):
        """Load AllSides CSV dataset."""
        print("📊 Loading AllSides dataset...")
        
        df = pd.read_csv('../data/allsides_balanced_news_headlines-texts.csv')
        df = df.dropna(subset=['text', 'bias_rating'])
        df = df[df['text'].str.len() > 10]
        
        # Standardize format
        allsides_data = []
        for _, row in df.iterrows():
            allsides_data.append({
                'text': row['text'],
                'bias_text': row['bias_rating'],
                'source': 'AllSides',
                'dataset': 'allsides'
            })
        
        print(f"AllSides: {len(allsides_data):,} articles")
        return allsides_data
    
    def load_json_data(self):
        """Load JSON dataset."""
        print("📊 Loading JSON dataset...")
        
        json_dir = Path('../data/jsons')
        json_files = list(json_dir.glob('*.json'))
        
        json_data = []
        bias_mapping = {0: 'center', 1: 'left', 2: 'right'}
        
        print(f"Processing {len(json_files):,} JSON files...")
        for i, json_file in enumerate(json_files):
            if i % 5000 == 0:
                print(f"  Progress: {i:,}/{len(json_files):,}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                # Use processed content if available, otherwise original
                text = article.get('content', article.get('content_original', ''))
                if len(text) > 10:  # Filter very short articles
                    json_data.append({
                        'text': text,
                        'bias_text': bias_mapping[article['bias']],
                        'source': article.get('source', 'Unknown'),
                        'dataset': 'json'
                    })
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        print(f"JSON dataset: {len(json_data):,} articles")
        return json_data
    
    def combine_datasets(self):
        """Combine both datasets and analyze."""
        print("\n🔄 Combining datasets...")
        
        # Load both datasets
        allsides_data = self.load_allsides_data()
        json_data = self.load_json_data()
        
        # Combine
        combined_data = allsides_data + json_data
        
        print(f"\n📊 Combined Dataset Analysis:")
        print(f"Total articles: {len(combined_data):,}")
        print(f"  AllSides: {len(allsides_data):,} ({len(allsides_data)/len(combined_data)*100:.1f}%)")
        print(f"  JSON: {len(json_data):,} ({len(json_data)/len(combined_data)*100:.1f}%)")
        
        # Analyze bias distribution
        bias_counts = {}
        for item in combined_data:
            bias = item['bias_text']
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
        
        print(f"\n🎯 Class Distribution:")
        total = len(combined_data)
        for bias, count in sorted(bias_counts.items()):
            percentage = count / total * 100
            print(f"  {bias}: {count:,} ({percentage:.1f}%)")
        
        # Create DataFrame for easy processing
        df = pd.DataFrame(combined_data)
        
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for training with stratified splits."""
        print("\n📋 Preparing training data...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(df['bias_text'])
        
        # Create label mappings
        self.id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        self.label2id = {label: i for i, label in enumerate(self.label_encoder.classes_)}
        
        print(f"📝 Label mapping: {self.id2label}")
        
        # Stratified splits: 70% train, 15% val, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'].values,
            labels,
            test_size=0.3,
            stratify=labels,
            random_state=RANDOM_SEED
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=RANDOM_SEED
        )
        
        print(f"\n📊 Data splits:")
        print(f"  Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
        
        # Create datasets
        train_dataset = NewsDataset(X_train, y_train, self.tokenizer)
        val_dataset = NewsDataset(X_val, y_val, self.tokenizer)
        test_dataset = NewsDataset(X_test, y_test, self.tokenizer)
        
        # Store for evaluation
        self.test_texts = X_test
        self.test_labels = y_test
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, train_dataset, val_dataset, output_dir='../models/roberta-combined-classifier'):
        """Train the model on combined dataset."""
        print(f"\n🚀 Starting training on combined dataset...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Model will be saved to: {output_dir}")
        
        # Conservative settings for stability
        batch_size = 8
        num_workers = 0
        epochs = 3
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=1000,  # More warmup for larger dataset
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=200,  # More frequent logging
            eval_strategy='steps',
            eval_steps=1000,    # Evaluate every 1000 steps
            save_strategy='steps',
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            seed=RANDOM_SEED,
            learning_rate=2e-5,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            lr_scheduler_type='linear',
            report_to=None,
            save_total_limit=3,
            dataloader_num_workers=num_workers,
        )
        
        print(f"🔧 Training configuration:")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: 2e-5")
        print(f"   Total training samples: {len(train_dataset):,}")
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        print("📈 Training started...")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f'{output_dir}/label_mapping.json', 'w') as f:
            json.dump({
                'id2label': self.id2label,
                'label2id': self.label2id
            }, f, indent=2)
        
        print(f"✅ Training completed!")
        print(f"📊 Training loss: {train_result.training_loss:.4f}")
        
        return trainer
    
    def evaluate_model(self, trainer, test_dataset):
        """Evaluate the trained model."""
        print(f"\n🧪 Evaluating model on test set...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = self.test_labels
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # Macro averages
        precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
        recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
        f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
        
        # Print results
        print(f"\n🎯 COMBINED DATASET RESULTS")
        print(f"=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        
        print(f"\n📊 Per-Class Results:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label.upper()}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'timestamp': timestamp,
            'model_name': 'roberta-combined-dataset',
            'test_size': len(y_true),
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'per_class_f1': [float(f) for f in f1],
            'per_class_support': [int(s) for s in support],
            'label_mapping': self.id2label
        }
        
        # Ensure results directory exists
        results_dir = '../results'
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = f'{results_dir}/combined_dataset_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
        
        return results

def main():
    """Main training pipeline for combined dataset."""
    print("🚀 Combined Dataset Training (AllSides + JSON)")
    print("=" * 60)
    
    # Configuration - CPU is faster based on performance test
    USE_CPU = True  # CPU is 1.1x faster than MPS for this workload
    
    # Initialize trainer
    trainer_obj = CombinedDatasetTrainer(force_cpu=USE_CPU)
    
    # Combine datasets
    combined_df = trainer_obj.combine_datasets()
    
    # Prepare training data
    train_dataset, val_dataset, test_dataset = trainer_obj.prepare_training_data(combined_df)
    
    # Train model
    trainer = trainer_obj.train_model(train_dataset, val_dataset)
    
    # Evaluate model
    results = trainer_obj.evaluate_model(trainer, test_dataset)
    
    print(f"\n✅ Combined dataset training completed!")
    print(f"📊 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"🏆 Final Test F1 (Macro): {results['f1_macro']:.4f}")
    print(f"💾 Model saved to: ../models/roberta-combined-classifier")
    
    return results

if __name__ == "__main__":
    results = main() 