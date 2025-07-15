#!/usr/bin/env python3
"""
RoBERTa-base Fine-tuning for Political Bias Detection

Fine-tunes FacebookAI/roberta-base on AllSides dataset with optimal hyperparameters
and stratified train/val/test splits (7:1.5:1.5).

Based on state-of-the-art practices for political bias detection.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class NewsDataset(Dataset):
    """Custom dataset for news bias classification."""
    
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
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BiasClassificationTrainer:
    """Fine-tuning trainer for political bias classification."""
    
    def __init__(self, model_name='FacebookAI/roberta-base', num_labels=3, force_cpu=False):
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Device detection with option to force CPU
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
        
        # Load tokenizer and model
        print(f"🤖 Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        
        print(f"✅ Model loaded on device: {self.device}")
        print(f"📊 Model configuration: {num_labels} classes")
    
    def load_and_prepare_data(self, csv_path, test_size=0.3, val_size=0.5):
        """
        Load and prepare data with stratified splitting.
        
        Args:
            csv_path: Path to the AllSides CSV file
            test_size: Proportion for test set (0.3 = 30% = 1.5/5 for test + 1.5/5 for val)
            val_size: Proportion of remaining data for validation (0.5 = half of 30%)
        
        Returns:
            Train, validation, and test datasets
        """
        print(f"📊 Loading data from {csv_path}...")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Bias distribution: {df['bias_rating'].value_counts()}")
        
        # Clean data
        df = df.dropna(subset=['text', 'bias_rating'])
        df = df[df['text'].str.len() > 10]  # Remove very short texts
        
        print(f"After cleaning: {len(df)} articles")
        
        # Encode labels
        self.label_encoder.fit(df['bias_rating'])
        labels = self.label_encoder.transform(df['bias_rating'])
        
        # Create label mapping for interpretation
        self.id2label = {i: label for i, label in enumerate(self.label_encoder.classes_)}
        self.label2id = {label: i for i, label in enumerate(self.label_encoder.classes_)}
        
        print(f"📝 Label mapping: {self.id2label}")
        
        # First split: 70% train, 30% temp (15% val + 15% test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'].values,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=RANDOM_SEED
        )
        
        # Second split: 15% val, 15% test from the 30% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=RANDOM_SEED
        )
        
        print(f"📊 Data splits:")
        print(f"  Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} ({len(X_val)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")
        
        # Create datasets
        train_dataset = NewsDataset(X_train, y_train, self.tokenizer)
        val_dataset = NewsDataset(X_val, y_val, self.tokenizer)
        test_dataset = NewsDataset(X_test, y_test, self.tokenizer)
        
        # Store for later use
        self.train_texts, self.train_labels = X_train, y_train
        self.val_texts, self.val_labels = X_val, y_val
        self.test_texts, self.test_labels = X_test, y_test
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
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
    
    def fine_tune(self, train_dataset, val_dataset, output_dir='../models/roberta-bias-classifier-v2'):
        """
        Fine-tune RoBERTa with optimal hyperparameters.
        
        Hyperparameters based on state-of-the-art bias detection research:
        - Learning rate: 2e-5
        - Batch size: 16 (GPU/MPS) or 8 (CPU)
        - Epochs: 3 (GPU/MPS) or 2 (CPU)
        - Adam optimizer with β=(0.9, 0.999), ε=1e-8
        - Linear decay with warmup
        - MPS acceleration on Apple Silicon
        """
        print(f"🚀 Starting fine-tuning...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Model will be saved to: {output_dir}")
        
        # Training arguments with conservative settings for stability
        batch_size = 8   # Conservative batch size for stability
        num_workers = 0  # No multiprocessing to avoid system overload
        epochs = 3 if self.device.type in ['mps', 'cuda'] else 2
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,              # Optimal epochs for GPU
            per_device_train_batch_size=batch_size,   # Larger batch for GPU
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,                # Warmup for linear decay
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            seed=RANDOM_SEED,
            learning_rate=2e-5,              # Optimal learning rate
            adam_beta1=0.9,                  # Optimal beta values
            adam_beta2=0.999,
            adam_epsilon=1e-8,               # Optimal epsilon
            lr_scheduler_type='linear',      # Linear decay with warmup
            report_to=None,                  # Disable wandb/tensorboard
            save_total_limit=3,
            dataloader_num_workers=num_workers,  # Enable workers for GPU
        )
        
        print(f"🔧 Training configuration:")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Workers: {num_workers}")
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        print("📈 Training started...")
        train_result = trainer.train()
        
        # Save the model
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
        """Comprehensive model evaluation."""
        print(f"🧪 Evaluating model on test set...")
        
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
        
        # Weighted averages
        precision_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
        recall_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
        f1_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Print results
        print(f"\n🎯 TEST SET RESULTS")
        print(f"=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        print(f"Precision (Macro): {precision_macro:.4f}")
        print(f"Recall (Macro): {recall_macro:.4f}")
        
        print(f"\n📊 Per-Class Results:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  {label.upper()}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Create visualizations
        self.create_evaluation_plots(y_true, y_pred, cm)
        
        # Save detailed results
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_name': self.model_name,
            'test_size': len(y_true),
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
            'label_mapping': self.id2label
        }
        
        # Ensure results directory exists
        results_dir = '../results'
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = f'{results_dir}/roberta_finetuned_results_{results["timestamp"]}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
        
        return results
    
    def create_evaluation_plots(self, y_true, y_pred, cm):
        """Create evaluation visualizations."""
        print(f"📊 Creating evaluation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RoBERTa Fine-tuned Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        bias_labels = list(self.label_encoder.classes_)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=bias_labels, yticklabels=bias_labels,
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')
        
        # 2. Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=bias_labels)
        
        metrics_df.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Per-Class Metrics')
        axes[0,1].set_ylabel('Score')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
        
        x = np.arange(len(bias_labels))
        width = 0.35
        
        true_counts_ordered = [counts[list(unique).index(i)] if i in unique else 0 for i in range(len(bias_labels))]
        pred_counts_ordered = [pred_counts[list(pred_unique).index(i)] if i in pred_unique else 0 for i in range(len(bias_labels))]
        
        axes[1,0].bar(x - width/2, true_counts_ordered, width, label='True', color='lightblue')
        axes[1,0].bar(x + width/2, pred_counts_ordered, width, label='Predicted', color='orange')
        
        axes[1,0].set_title('True vs Predicted Distribution')
        axes[1,0].set_xlabel('Bias Class')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(bias_labels)
        axes[1,0].legend()
        
        # 4. Overall metrics comparison
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
        recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
        f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
        f1_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]
        
        overall_metrics = {
            'Accuracy': accuracy,
            'Precision (Macro)': precision_macro,
            'Recall (Macro)': recall_macro,
            'F1 (Macro)': f1_macro,
            'F1 (Weighted)': f1_weighted
        }
        
        bars = axes[1,1].bar(overall_metrics.keys(), overall_metrics.values(), 
                            color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'orange'])
        axes[1,1].set_title('Overall Performance Metrics')
        axes[1,1].set_ylabel('Score')
        axes[1,1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom')
        
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f"roberta_finetuned_evaluation_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Evaluation plot saved as {plot_filename}")
        
        return plot_filename

def main():
    """Main fine-tuning pipeline."""
    print("🚀 RoBERTa Fine-tuning for Political Bias Detection")
    print("=" * 60)
    
    # Option to force CPU for stability (change force_cpu=True to use CPU)
    USE_CPU = True  # Set to False to try MPS again
    
    # Initialize trainer
    trainer_obj = BiasClassificationTrainer(
        model_name='FacebookAI/roberta-base',
        num_labels=3,
        force_cpu=USE_CPU
    )
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = trainer_obj.load_and_prepare_data(
        '../data/allsides_balanced_news_headlines-texts.csv'
    )
    
    # Fine-tune the model
    trainer = trainer_obj.fine_tune(train_dataset, val_dataset)
    
    # Evaluate the model
    results = trainer_obj.evaluate_model(trainer, test_dataset)
    
    print(f"\n✅ Fine-tuning completed successfully!")
    print(f"📊 Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"🏆 Final Test F1 (Macro): {results['f1_macro']:.4f}")
    print(f"💾 Model saved to: ../models/roberta-bias-classifier-v2")

if __name__ == "__main__":
    main() 