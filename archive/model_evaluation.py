#!/usr/bin/env python3
"""
BiasCheck-RoBERTa Model Evaluation Script

Tests the peekayitachi/BiasCheck-RoBERTa model on AllSides balanced news dataset
using stratified sampling and comprehensive evaluation metrics.

Dataset: 21,754 news articles from AllSides (left: 47.2%, right: 33.2%, center: 19.6%)
Model: RoBERTa-base fine-tuned for political bias detection (Left/Center/Right)
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class BiasCheckEvaluator:
    """Comprehensive evaluation of BiasCheck-RoBERTa model."""
    
    def __init__(self, sample_size=1000):
        """Initialize evaluator with stratified sampling."""
        self.sample_size = sample_size
        self.model_name = 'peekayitachi/BiasCheck-RoBERTa'
        
        # Load model and tokenizer
        print(f"🔄 Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Label mappings (need to determine empirically)
        self.dataset_labels = ['left', 'center', 'right']
        self.model_labels = ['LABEL_0', 'LABEL_1', 'LABEL_2']
        
        print("✅ Model loaded successfully!")
    
    def load_and_sample_data(self, csv_path):
        """Load dataset and create stratified sample."""
        print(f"\n📊 Loading dataset from {csv_path}...")
        
        # Load full dataset
        df = pd.read_csv(csv_path)
        print(f"Full dataset: {len(df):,} articles")
        print(f"Bias distribution: {df['bias_rating'].value_counts().to_dict()}")
        
        # Create stratified sample
        print(f"\n🎯 Creating stratified sample of {self.sample_size} articles...")
        
        # Calculate sample sizes to maintain proportions
        bias_counts = df['bias_rating'].value_counts()
        sample_sizes = {}
        for bias in self.dataset_labels:
            if bias in bias_counts:
                proportion = bias_counts[bias] / len(df)
                sample_sizes[bias] = int(self.sample_size * proportion)
        
        # Adjust for rounding errors
        total_sampled = sum(sample_sizes.values())
        if total_sampled < self.sample_size:
            # Add extra samples to the largest class
            largest_class = max(sample_sizes, key=sample_sizes.get)
            sample_sizes[largest_class] += (self.sample_size - total_sampled)
        
        print(f"Sample sizes: {sample_sizes}")
        
        # Sample from each bias category
        sampled_dfs = []
        for bias, size in sample_sizes.items():
            bias_df = df[df['bias_rating'] == bias].sample(n=size, random_state=42)
            sampled_dfs.append(bias_df)
        
        sample_df = pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
        print(f"✅ Sample created: {len(sample_df)} articles")
        print(f"Sample bias distribution: {sample_df['bias_rating'].value_counts().to_dict()}")
        
        return sample_df
    
    def predict_bias(self, text):
        """Predict bias for a single text."""
        # Tokenize with truncation for long texts
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probabilities = softmax(logits)
        
        # Get predicted label and confidence
        predicted_idx = np.argmax(probabilities)
        predicted_label = self.model_labels[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        return {
            'predicted_label': predicted_label,
            'predicted_idx': predicted_idx,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }
    
    def determine_label_mapping(self, sample_df, num_test=50):
        """Empirically determine model label mapping by testing on known examples."""
        print(f"\n🧪 Determining label mapping using {num_test} examples per class...")
        
        mapping_votes = {'left': [], 'center': [], 'right': []}
        
        for true_bias in self.dataset_labels:
            if true_bias not in sample_df['bias_rating'].values:
                continue
                
            # Get examples for this bias
            examples = sample_df[sample_df['bias_rating'] == true_bias].head(num_test)
            
            for _, row in examples.iterrows():
                prediction = self.predict_bias(row['text'])
                mapping_votes[true_bias].append(prediction['predicted_idx'])
        
        # Determine most common prediction for each true label
        label_mapping = {}
        for true_bias, predictions in mapping_votes.items():
            if predictions:
                most_common_idx = max(set(predictions), key=predictions.count)
                label_mapping[true_bias] = most_common_idx
                print(f"  {true_bias} → LABEL_{most_common_idx} ({predictions.count(most_common_idx)}/{len(predictions)} votes)")
        
        return label_mapping
    
    def evaluate_model(self, sample_df):
        """Run comprehensive evaluation."""
        print(f"\n🚀 Running evaluation on {len(sample_df)} articles...")
        
        # Determine label mapping
        label_mapping = self.determine_label_mapping(sample_df)
        
        # Predict on all samples
        predictions = []
        true_labels = []
        confidences = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Predicting"):
            try:
                # Validate text input
                text = row['text']
                if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                    
                prediction = self.predict_bias(text)
                predictions.append(prediction['predicted_idx'])
                confidences.append(prediction['confidence'])
                
                # Map true label to model index
                true_bias = row['bias_rating']
                true_idx = label_mapping.get(true_bias, -1)
                true_labels.append(true_idx)
                
            except Exception as e:
                print(f"Error processing article {idx}: {e}")
                continue
        
        # Calculate metrics
        results = self.calculate_metrics(true_labels, predictions, confidences, label_mapping)
        
        # Create visualizations
        self.create_visualizations(true_labels, predictions, label_mapping, results)
        
        return results
    
    def calculate_metrics(self, true_labels, predictions, confidences, label_mapping):
        """Calculate comprehensive evaluation metrics."""
        print(f"\n📈 Calculating metrics...")
        
        # Remove any invalid predictions
        valid_indices = [i for i, true_label in enumerate(true_labels) if true_label != -1]
        true_labels = [true_labels[i] for i in valid_indices]
        predictions = [predictions[i] for i in valid_indices]
        confidences = [confidences[i] for i in valid_indices]
        
        # Calculate basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        target_names = [f"{bias} (LABEL_{idx})" for bias, idx in label_mapping.items()]
        class_report_raw = classification_report(
            true_labels, predictions, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Convert all numpy types in classification report to Python types
        def convert_to_python_types(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        class_report = convert_to_python_types(class_report_raw)
        
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'model_name': self.model_name,
            'sample_size': int(len(true_labels)),
            'label_mapping': label_mapping,
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'precision_per_class': [float(x) for x in precision.tolist()],
            'recall_per_class': [float(x) for x in recall.tolist()],
            'f1_per_class': [float(x) for x in f1.tolist()],
            'support_per_class': [int(x) for x in support.tolist()],
            'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
            'classification_report': class_report,
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences))
        }
        
        return results
    
    def create_visualizations(self, true_labels, predictions, label_mapping, results):
        """Create evaluation visualizations."""
        print(f"\n📊 Creating visualizations...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BiasCheck-RoBERTa Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = np.array(results['confusion_matrix'])
        bias_labels = list(label_mapping.keys())
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=bias_labels, yticklabels=bias_labels,
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')
        
        # 2. Metrics by Class
        metrics_df = pd.DataFrame({
            'Precision': results['precision_per_class'],
            'Recall': results['recall_per_class'],
            'F1-Score': results['f1_per_class']
        }, index=bias_labels)
        
        metrics_df.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Metrics by Class')
        axes[0,1].set_ylabel('Score')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Overall Metrics Comparison
        overall_metrics = {
            'Accuracy': results['accuracy'],
            'Precision (Macro)': results['precision_macro'],
            'Recall (Macro)': results['recall_macro'],
            'F1 (Macro)': results['f1_macro'],
            'F1 (Weighted)': results['f1_weighted']
        }
        
        bars = axes[1,0].bar(overall_metrics.keys(), overall_metrics.values(), 
                            color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'orange'])
        axes[1,0].set_title('Overall Performance Metrics')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom')
        
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Prediction Distribution
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        true_counts = pd.Series(true_labels).value_counts().sort_index()
        
        x = np.arange(len(bias_labels))
        width = 0.35
        
        axes[1,1].bar(x - width/2, [true_counts.get(i, 0) for i in range(len(bias_labels))], 
                     width, label='True', color='lightblue')
        axes[1,1].bar(x + width/2, [pred_counts.get(i, 0) for i in range(len(bias_labels))], 
                     width, label='Predicted', color='orange')
        
        axes[1,1].set_title('True vs Predicted Distribution')
        axes[1,1].set_xlabel('Bias Class')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(bias_labels)
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"bias_evaluation_{results['timestamp']}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved as {plot_filename}")
        
        return plot_filename
    
    def print_results(self, results):
        """Print formatted evaluation results."""
        print(f"\n" + "="*60)
        print(f"🎯 BIASCHECK-ROBERTA EVALUATION RESULTS")
        print(f"="*60)
        print(f"Model: {results['model_name']}")
        print(f"Sample Size: {results['sample_size']:,}")
        print(f"Timestamp: {results['timestamp']}")
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Accuracy: {results['accuracy']:.3f}")
        print(f"  Precision (Macro): {results['precision_macro']:.3f}")
        print(f"  Recall (Macro): {results['recall_macro']:.3f}")
        print(f"  F1-Score (Macro): {results['f1_macro']:.3f}")
        print(f"  F1-Score (Weighted): {results['f1_weighted']:.3f}")
        
        print(f"\n🔍 PER-CLASS PERFORMANCE:")
        bias_labels = list(results['label_mapping'].keys())
        for i, bias in enumerate(bias_labels):
            if i < len(results['precision_per_class']):
                print(f"  {bias.upper()}:")
                print(f"    Precision: {results['precision_per_class'][i]:.3f}")
                print(f"    Recall: {results['recall_per_class'][i]:.3f}")
                print(f"    F1-Score: {results['f1_per_class'][i]:.3f}")
                print(f"    Support: {results['support_per_class'][i]}")
        
        print(f"\n🎲 PREDICTION CONFIDENCE:")
        print(f"  Mean: {results['mean_confidence']:.3f}")
        print(f"  Std Dev: {results['std_confidence']:.3f}")
        
        print(f"\n🗂️  LABEL MAPPING:")
        for bias, idx in results['label_mapping'].items():
            print(f"  {bias} → LABEL_{idx}")
    
    def save_results(self, results):
        """Save detailed results to JSON file."""
        filename = f"bias_analysis_results_{results['timestamp']}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to {filename}")
        return filename


def main():
    """Main evaluation pipeline."""
    print("🚀 BiasCheck-RoBERTa Model Evaluation")
    print("="*50)
    
    # Initialize evaluator
    evaluator = BiasCheckEvaluator(sample_size=1000)
    
    # Load and sample data
    dataset_path = 'allsides_balanced_news_headlines-texts.csv'
    sample_df = evaluator.load_and_sample_data(dataset_path)
    
    # Run evaluation
    results = evaluator.evaluate_model(sample_df)
    
    # Display and save results
    evaluator.print_results(results)
    evaluator.save_results(results)
    
    print(f"\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main() 