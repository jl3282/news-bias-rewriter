#!/usr/bin/env python3
"""
Comprehensive Analysis: Model Comparison, Error Analysis, and Dataset Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class DatasetAnalyzer:
    """Analyze both AllSides and new JSON datasets"""
    
    def analyze_allsides_dataset(self):
        """Analyze the AllSides CSV dataset"""
        print("📊 Analyzing AllSides Dataset")
        print("=" * 50)
        
        df = pd.read_csv('allsides_balanced_news_headlines-texts.csv')
        
        print(f"Total articles: {len(df):,}")
        print(f"Date range: {df.get('date', 'N/A').describe() if 'date' in df.columns else 'No date column'}")
        
        # Bias distribution
        bias_dist = df['bias_rating'].value_counts()
        print(f"\nBias distribution:")
        for bias, count in bias_dist.items():
            percentage = count / len(df) * 100
            print(f"  {bias}: {count:,} ({percentage:.1f}%)")
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        print(f"\nText length statistics:")
        print(f"  Mean: {df['text_length'].mean():.0f} characters")
        print(f"  Median: {df['text_length'].median():.0f} characters")
        print(f"  Min: {df['text_length'].min():.0f} characters")
        print(f"  Max: {df['text_length'].max():.0f} characters")
        
        return {
            'name': 'AllSides',
            'total_articles': len(df),
            'bias_distribution': bias_dist.to_dict(),
            'text_length_stats': df['text_length'].describe().to_dict()
        }
    
    def analyze_json_dataset(self):
        """Analyze the new JSON dataset"""
        print("\n📊 Analyzing New JSON Dataset")
        print("=" * 50)
        
        # Count total files
        json_dir = Path('data/jsons')
        json_files = list(json_dir.glob('*.json'))
        total_files = len(json_files)
        print(f"Total JSON files: {total_files:,}")
        
        # Sample and analyze structure
        sample_size = min(1000, total_files)  # Analyze first 1000 files
        print(f"Analyzing sample of {sample_size} files...")
        
        articles = []
        bias_counts = Counter()
        source_counts = Counter()
        topic_counts = Counter()
        text_lengths = []
        dates = []
        
        for i, json_file in enumerate(json_files[:sample_size]):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    article = json.load(f)
                
                articles.append(article)
                bias_counts[article.get('bias_text', 'unknown')] += 1
                source_counts[article.get('source', 'unknown')] += 1
                topic_counts[article.get('topic', 'unknown')] += 1
                
                content_len = len(article.get('content', ''))
                text_lengths.append(content_len)
                
                if article.get('date'):
                    dates.append(article['date'])
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        # Bias distribution
        print(f"\nBias distribution (sample of {len(articles)}):")
        total_sample = sum(bias_counts.values())
        for bias, count in bias_counts.most_common():
            percentage = count / total_sample * 100
            print(f"  {bias}: {count:,} ({percentage:.1f}%)")
        
        # Top sources
        print(f"\nTop 10 sources:")
        for source, count in source_counts.most_common(10):
            percentage = count / total_sample * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
        
        # Top topics
        print(f"\nTop 10 topics:")
        for topic, count in topic_counts.most_common(10):
            percentage = count / total_sample * 100
            print(f"  {topic}: {count} ({percentage:.1f}%)")
        
        # Text length analysis
        if text_lengths:
            print(f"\nText length statistics:")
            print(f"  Mean: {np.mean(text_lengths):.0f} characters")
            print(f"  Median: {np.median(text_lengths):.0f} characters")
            print(f"  Min: {min(text_lengths):.0f} characters")
            print(f"  Max: {max(text_lengths):.0f} characters")
        
        # Date analysis
        if dates:
            print(f"\nDate range:")
            print(f"  Earliest: {min(dates)}")
            print(f"  Latest: {max(dates)}")
            print(f"  Unique dates: {len(set(dates))}")
        
        return {
            'name': 'JSON Dataset',
            'total_articles': total_files,
            'analyzed_sample': len(articles),
            'bias_distribution': dict(bias_counts),
            'top_sources': dict(source_counts.most_common(10)),
            'top_topics': dict(topic_counts.most_common(10)),
            'text_length_stats': {
                'mean': np.mean(text_lengths) if text_lengths else 0,
                'median': np.median(text_lengths) if text_lengths else 0,
                'min': min(text_lengths) if text_lengths else 0,
                'max': max(text_lengths) if text_lengths else 0
            },
            'date_range': {'earliest': min(dates), 'latest': max(dates)} if dates else None
        }
    
    def analyze_splits(self):
        """Analyze the train/val/test splits"""
        print("\n📊 Analyzing Dataset Splits")
        print("=" * 50)
        
        splits_analysis = {}
        
        for split_type in ['random', 'media']:
            print(f"\n{split_type.upper()} splits:")
            splits_analysis[split_type] = {}
            
            for split_name in ['train', 'valid', 'test']:
                split_path = f'data/splits/{split_type}/{split_name}.tsv'
                
                if os.path.exists(split_path):
                    df = pd.read_csv(split_path, sep='\t')
                    bias_dist = df['bias'].value_counts().sort_index()
                    
                    print(f"  {split_name}: {len(df):,} articles")
                    for bias_num, count in bias_dist.items():
                        bias_name = ['center', 'left', 'right'][bias_num]
                        percentage = count / len(df) * 100
                        print(f"    {bias_name}: {count:,} ({percentage:.1f}%)")
                    
                    splits_analysis[split_type][split_name] = {
                        'total': len(df),
                        'bias_distribution': bias_dist.to_dict()
                    }
        
        return splits_analysis

class ModelComparator:
    """Compare different models' performance"""
    
    def __init__(self):
        self.allsides_test_data = None
        self.label_encoder = None
    
    def load_allsides_test_data(self):
        """Load the same test data used for fine-tuned model evaluation"""
        print("📊 Loading AllSides test data...")
        
        # Same process as in training/evaluation
        df = pd.read_csv('allsides_balanced_news_headlines-texts.csv')
        df = df.dropna(subset=['text', 'bias_rating'])
        df = df[df['text'].str.len() > 10]
        
        # Same label encoding and splits as training
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(df['bias_rating'])
        labels = self.label_encoder.transform(df['bias_rating'])
        
        # Same splits (70/15/15) with same random seed
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'].values, labels, test_size=0.3, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        self.allsides_test_data = (X_test, y_test)
        print(f"Test set: {len(X_test)} articles")
        
        return X_test, y_test
    
    def evaluate_biascheck_model(self, X_test, y_test):
        """Evaluate BiasCheck-RoBERTa on the same test set"""
        print("\n🤖 Evaluating BiasCheck-RoBERTa...")
        
        from transformers import pipeline
        
        # Load BiasCheck model
        classifier = pipeline(
            "text-classification",
            model="peekayitachi/BiasCheck-RoBERTa",
            tokenizer="peekayitachi/BiasCheck-RoBERTa"
        )
        
        print("Making predictions...")
        predictions = []
        
        for i, text in enumerate(X_test):
            if i % 500 == 0:
                print(f"  Progress: {i}/{len(X_test)}")
            
            try:
                # Truncate text for model limits
                text_truncated = text[:500]  # BiasCheck has shorter context
                result = classifier(text_truncated)
                
                # Map BiasCheck labels to our format
                label = result[0]['label']
                if label == 'LABEL_0':  # left
                    pred = 1  # left in our encoding
                elif label == 'LABEL_1':  # center (but rarely used)
                    pred = 0  # center in our encoding
                else:  # LABEL_2 (right/non-left)
                    pred = 2  # right in our encoding
                
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error predicting sample {i}: {e}")
                predictions.append(0)  # Default to center
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average=None)
        precision_macro = precision_recall_fscore_support(y_test, predictions, average='macro')[0]
        recall_macro = precision_recall_fscore_support(y_test, predictions, average='macro')[1]
        f1_macro = precision_recall_fscore_support(y_test, predictions, average='macro')[2]
        
        return {
            'model_name': 'BiasCheck-RoBERTa',
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'predictions': predictions
        }
    
    def load_finetuned_results(self):
        """Load our fine-tuned model results"""
        print("\n📊 Loading Fine-tuned RoBERTa results...")
        
        # Find the most recent results file
        result_files = [f for f in os.listdir('.') if f.startswith('roberta_finetuned_final_results_') and f.endswith('.json')]
        if not result_files:
            print("❌ No fine-tuned results found!")
            return None
        
        latest_file = sorted(result_files)[-1]
        print(f"Loading results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            results = json.load(f)
        
        return results
    
    def compare_models(self):
        """Compare BiasCheck vs Fine-tuned RoBERTa"""
        print("\n🆚 Model Comparison")
        print("=" * 50)
        
        # Load test data
        X_test, y_test = self.load_allsides_test_data()
        
        # Load fine-tuned results
        finetuned_results = self.load_finetuned_results()
        if not finetuned_results:
            return None
        
        # Evaluate BiasCheck (this might take a while)
        print("⏳ This may take several minutes...")
        biascheck_results = self.evaluate_biascheck_model(X_test, y_test)
        
        # Compare results
        comparison = {
            'test_set_size': len(X_test),
            'models': {
                'Fine-tuned RoBERTa': {
                    'accuracy': finetuned_results['accuracy'],
                    'f1_macro': finetuned_results['f1_macro'],
                    'precision_macro': finetuned_results['precision_macro'],
                    'recall_macro': finetuned_results['recall_macro']
                },
                'BiasCheck-RoBERTa': {
                    'accuracy': biascheck_results['accuracy'],
                    'f1_macro': biascheck_results['f1_macro'],
                    'precision_macro': biascheck_results['precision_macro'],
                    'recall_macro': biascheck_results['recall_macro']
                }
            }
        }
        
        # Print comparison
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print(f"Test set: {len(X_test)} articles")
        print(f"")
        print(f"{'Metric':<20} {'Fine-tuned':<15} {'BiasCheck':<15} {'Improvement':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        
        ft_acc = finetuned_results['accuracy']
        bc_acc = biascheck_results['accuracy']
        acc_improve = ft_acc - bc_acc
        print(f"{'Accuracy':<20} {ft_acc:<15.4f} {bc_acc:<15.4f} {acc_improve:+.4f}")
        
        ft_f1 = finetuned_results['f1_macro']
        bc_f1 = biascheck_results['f1_macro']
        f1_improve = ft_f1 - bc_f1
        print(f"{'F1 (Macro)':<20} {ft_f1:<15.4f} {bc_f1:<15.4f} {f1_improve:+.4f}")
        
        ft_prec = finetuned_results['precision_macro']
        bc_prec = biascheck_results['precision_macro']
        prec_improve = ft_prec - bc_prec
        print(f"{'Precision (Macro)':<20} {ft_prec:<15.4f} {bc_prec:<15.4f} {prec_improve:+.4f}")
        
        ft_rec = finetuned_results['recall_macro']
        bc_rec = biascheck_results['recall_macro']
        rec_improve = ft_rec - bc_rec
        print(f"{'Recall (Macro)':<20} {ft_rec:<15.4f} {bc_rec:<15.4f} {rec_improve:+.4f}")
        
        return comparison, biascheck_results, finetuned_results

class ErrorAnalyzer:
    """Analyze prediction errors from fine-tuned model"""
    
    def analyze_errors(self):
        """Perform detailed error analysis"""
        print("\n🔍 Error Analysis - Fine-tuned Model")
        print("=" * 50)
        
        # Load the fine-tuned model and make predictions
        model_path = 'roberta-bias-classifier'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        # Load test data (same as evaluation)
        df = pd.read_csv('allsides_balanced_news_headlines-texts.csv')
        df = df.dropna(subset=['text', 'bias_rating'])
        df = df[df['text'].str.len() > 10]
        
        label_encoder = LabelEncoder()
        label_encoder.fit(df['bias_rating'])
        labels = label_encoder.transform(df['bias_rating'])
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['text'].values, labels, test_size=0.3, stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        # Get predictions and confidence scores
        predictions = []
        confidence_scores = []
        
        print("Making predictions for error analysis...")
        with torch.no_grad():
            for i, text in enumerate(X_test):
                if i % 500 == 0:
                    print(f"  Progress: {i}/{len(X_test)}")
                
                inputs = tokenizer(text, truncation=True, padding='max_length', 
                                 max_length=512, return_tensors='pt')
                outputs = model(**inputs)
                
                # Get prediction and confidence
                probs = torch.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs, dim=-1).item()
                confidence = torch.max(probs).item()
                
                predictions.append(pred)
                confidence_scores.append(confidence)
        
        predictions = np.array(predictions)
        confidence_scores = np.array(confidence_scores)
        
        # Analyze errors
        correct_mask = (predictions == y_test)
        error_mask = ~correct_mask
        
        print(f"\n📊 Error Statistics:")
        print(f"Total predictions: {len(predictions)}")
        print(f"Correct: {correct_mask.sum()} ({correct_mask.mean()*100:.1f}%)")
        print(f"Errors: {error_mask.sum()} ({error_mask.mean()*100:.1f}%)")
        
        # Confidence analysis
        print(f"\n🎯 Confidence Analysis:")
        print(f"Average confidence (correct): {confidence_scores[correct_mask].mean():.4f}")
        print(f"Average confidence (errors): {confidence_scores[error_mask].mean():.4f}")
        
        # Error breakdown by true class
        print(f"\n❌ Errors by True Class:")
        bias_labels = ['center', 'left', 'right']
        for true_class in range(3):
            true_mask = (y_test == true_class)
            class_errors = error_mask & true_mask
            if true_mask.sum() > 0:
                error_rate = class_errors.sum() / true_mask.sum()
                print(f"  {bias_labels[true_class]}: {class_errors.sum()}/{true_mask.sum()} ({error_rate*100:.1f}% error rate)")
        
        # Most confident errors (likely systematic issues)
        error_indices = np.where(error_mask)[0]
        error_confidences = confidence_scores[error_mask]
        high_confidence_errors = error_indices[error_confidences > 0.8]
        
        print(f"\n🔥 High-Confidence Errors (>80% confidence):")
        print(f"Count: {len(high_confidence_errors)}")
        
        if len(high_confidence_errors) > 0:
            # Analyze a few examples
            print(f"\nExample high-confidence errors:")
            for i, idx in enumerate(high_confidence_errors[:3]):
                true_label = bias_labels[y_test[idx]]
                pred_label = bias_labels[predictions[idx]]
                conf = confidence_scores[idx]
                text_preview = X_test[idx][:200] + "..."
                
                print(f"\nExample {i+1}:")
                print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {conf:.3f}")
                print(f"  Text: {text_preview}")
        
        return {
            'total_predictions': len(predictions),
            'correct_count': correct_mask.sum(),
            'error_count': error_mask.sum(),
            'accuracy': correct_mask.mean(),
            'avg_confidence_correct': confidence_scores[correct_mask].mean(),
            'avg_confidence_errors': confidence_scores[error_mask].mean(),
            'high_confidence_errors': len(high_confidence_errors),
            'error_breakdown': {
                bias_labels[i]: {
                    'total': (y_test == i).sum(),
                    'errors': (error_mask & (y_test == i)).sum(),
                    'error_rate': (error_mask & (y_test == i)).sum() / max((y_test == i).sum(), 1)
                } for i in range(3)
            }
        }

def main():
    """Run comprehensive analysis"""
    print("🚀 Comprehensive Model and Dataset Analysis")
    print("=" * 60)
    
    # 1. Dataset Analysis
    analyzer = DatasetAnalyzer()
    allsides_analysis = analyzer.analyze_allsides_dataset()
    json_analysis = analyzer.analyze_json_dataset()
    splits_analysis = analyzer.analyze_splits()
    
    # 2. Model Comparison
    comparator = ModelComparator()
    comparison_results = comparator.compare_models()
    
    # 3. Error Analysis
    error_analyzer = ErrorAnalyzer()
    error_results = error_analyzer.analyze_errors()
    
    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comprehensive_results = {
        'timestamp': timestamp,
        'allsides_analysis': allsides_analysis,
        'json_dataset_analysis': json_analysis,
        'splits_analysis': splits_analysis,
        'model_comparison': comparison_results[0] if comparison_results else None,
        'error_analysis': error_results
    }
    
    with open(f'comprehensive_analysis_{timestamp}.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Results saved to comprehensive_analysis_{timestamp}.json")
    
    return comprehensive_results

if __name__ == "__main__":
    results = main() 