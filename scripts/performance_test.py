#!/usr/bin/env python3
"""
Quick Performance Test: CPU vs MPS
Tests training speed on small dataset subset to choose optimal device
"""

import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class QuickDataset(Dataset):
    """Small dataset for performance testing."""
    
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

def load_sample_data(sample_size=2000):
    """Load a small sample of combined data for testing."""
    print(f"📊 Loading sample data ({sample_size} articles)...")
    
    # Load AllSides sample
    df_allsides = pd.read_csv('allsides_balanced_news_headlines-texts.csv')
    df_allsides = df_allsides.dropna(subset=['text', 'bias_rating'])
    df_allsides = df_allsides[df_allsides['text'].str.len() > 10]
    
    # Sample from AllSides
    allsides_sample = df_allsides.sample(min(1000, len(df_allsides)), random_state=42)
    
    # Load JSON sample  
    json_dir = Path('data/jsons')
    json_files = list(json_dir.glob('*.json'))[:1000]  # First 1000 files
    
    json_data = []
    bias_mapping = {0: 'center', 1: 'left', 2: 'right'}
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                article = json.load(f)
            
            text = article.get('content', article.get('content_original', ''))
            if len(text) > 10:
                json_data.append({
                    'text': text,
                    'bias_text': bias_mapping[article['bias']]
                })
        except:
            continue
    
    # Combine samples
    combined_sample = []
    
    # Add AllSides sample
    for _, row in allsides_sample.iterrows():
        combined_sample.append({
            'text': row['text'],
            'bias_text': row['bias_rating']
        })
    
    # Add JSON sample
    combined_sample.extend(json_data[:1000])
    
    # Limit to requested sample size
    if len(combined_sample) > sample_size:
        combined_sample = combined_sample[:sample_size]
    
    df = pd.DataFrame(combined_sample)
    
    print(f"Sample size: {len(df)}")
    print(f"Class distribution: {df['bias_text'].value_counts().to_dict()}")
    
    return df

def test_device_performance(device_name, force_cpu=False, sample_size=2000):
    """Test training performance on specified device."""
    print(f"\n🧪 Testing {device_name.upper()} Performance")
    print("=" * 40)
    
    # Load sample data
    df = load_sample_data(sample_size)
    
    # Setup device
    if force_cpu:
        device = torch.device('cpu')
    elif torch.backends.mps.is_available() and device_name == 'mps':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained(
        'FacebookAI/roberta-base', 
        num_labels=3
    )
    model.to(device)
    
    # Prepare data
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['bias_text'])
    
    # Small train/val split for testing
    X_train, X_val, y_train, y_val = train_test_split(
        df['text'].values, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    train_dataset = QuickDataset(X_train, y_train, tokenizer)
    val_dataset = QuickDataset(X_val, y_val, tokenizer)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Training configuration - conservative for stability
    training_args = TrainingArguments(
        output_dir=f'./test-{device_name}',
        num_train_epochs=1,  # Just 1 epoch for testing
        per_device_train_batch_size=8,  # Conservative batch size
        per_device_eval_batch_size=8,
        max_steps=50,  # Only 50 steps for quick test
        warmup_steps=10,
        logging_steps=10,
        eval_strategy='steps',
        eval_steps=25,
        save_strategy='no',  # Don't save during test
        learning_rate=2e-5,
        dataloader_num_workers=0,  # Conservative
        report_to=None,
        disable_tqdm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Time the training
    print("⏱️ Starting timed training...")
    start_time = time.time()
    
    try:
        trainer.train()
        end_time = time.time()
        
        training_time = end_time - start_time
        time_per_step = training_time / 50  # 50 steps
        
        print(f"✅ Training completed successfully!")
        print(f"Total time: {training_time:.2f} seconds")
        print(f"Time per step: {time_per_step:.3f} seconds")
        
        return {
            'device': device_name,
            'success': True,
            'total_time': training_time,
            'time_per_step': time_per_step,
            'sample_size': len(train_dataset)
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {
            'device': device_name,
            'success': False,
            'error': str(e)
        }

def estimate_full_training_time(time_per_step, total_samples, batch_size=8, epochs=3):
    """Estimate full training time based on test results."""
    steps_per_epoch = total_samples // batch_size
    total_steps = steps_per_epoch * epochs
    estimated_hours = (total_steps * time_per_step) / 3600
    return estimated_hours, total_steps

def main():
    """Run performance comparison."""
    print("🚀 CPU vs MPS Performance Comparison")
    print("=" * 60)
    
    # Test both devices
    results = []
    
    # Test CPU
    cpu_result = test_device_performance('cpu', force_cpu=True)
    results.append(cpu_result)
    
    # Test MPS (if available)
    if torch.backends.mps.is_available():
        mps_result = test_device_performance('mps', force_cpu=False)
        results.append(mps_result)
    else:
        print("\n⚠️ MPS not available, skipping MPS test")
    
    # Compare results
    print(f"\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) >= 2:
        cpu_time = successful_results[0]['time_per_step']
        mps_time = successful_results[1]['time_per_step']
        
        print(f"CPU time per step: {cpu_time:.3f} seconds")
        print(f"MPS time per step: {mps_time:.3f} seconds")
        
        if mps_time < cpu_time:
            speedup = cpu_time / mps_time
            print(f"🏆 MPS is {speedup:.1f}x FASTER than CPU")
            recommended = 'MPS'
        else:
            slowdown = mps_time / cpu_time
            print(f"🐌 MPS is {slowdown:.1f}x SLOWER than CPU")
            recommended = 'CPU'
        
        print(f"\n💡 RECOMMENDATION: Use {recommended}")
        
        # Estimate full training time for both
        print(f"\n⏱️ FULL TRAINING TIME ESTIMATES (59K articles, 3 epochs):")
        for result in successful_results:
            if result['success']:
                hours, steps = estimate_full_training_time(result['time_per_step'], 59000)
                print(f"  {result['device'].upper()}: ~{hours:.1f} hours ({steps:,} total steps)")
    
    elif len(successful_results) == 1:
        result = successful_results[0]
        print(f"Only {result['device'].upper()} test successful")
        hours, steps = estimate_full_training_time(result['time_per_step'], 59000)
        print(f"Estimated full training time: ~{hours:.1f} hours")
    
    else:
        print("❌ Both tests failed")
    
    # Save results
    with open('performance_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Cleanup test directories
    import shutil
    for result in results:
        test_dir = f"./test-{result['device']}"
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
    
    return results

if __name__ == "__main__":
    results = main() 