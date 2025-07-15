#!/usr/bin/env python3
"""
BiasCheck-RoBERTa Political Bias Analyzer

This script uses the peekayitachi/BiasCheck-RoBERTa model to analyze 
political bias in news articles. The model classifies text into:
- Left bias (LABEL_0)
- Center/Neutral (LABEL_1)  
- Right bias (LABEL_2)

Trained on AllSides dataset for reliable political bias detection.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import argparse


class BiasCheckAnalyzer:
    """Political bias analyzer using BiasCheck-RoBERTa model."""
    
    def __init__(self):
        """Initialize the BiasCheck analyzer."""
        self.model_name = "peekayitachi/BiasCheck-RoBERTa"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔄 Loading BiasCheck-RoBERTa model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # Label mapping based on model documentation
        self.label_names = {
            0: "Left",
            1: "Center", 
            2: "Right"
        }
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📊 Classes: {', '.join(self.label_names.values())}")
    
    def chunk_text(self, text: str, max_tokens: int = 400) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ''
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence + '. '
            tokens = self.tokenizer(test_chunk, return_tensors='pt')['input_ids'].shape[1]
            
            if tokens < max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + '. '
                else:
                    # Handle very long sentences
                    chunks.append(sentence + '. ')
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def analyze_chunk(self, text: str) -> Dict:
        """
        Analyze political bias of a single text chunk.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with bias analysis results
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probabilities = softmax(logits)
        
        # Format results
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        scores = {
            self.label_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "prediction": self.label_names[predicted_class],
            "confidence": float(confidence),
            "scores": scores,
            "token_count": inputs['input_ids'].shape[1]
        }
    
    def analyze_article(self, text: str, show_chunks: bool = True) -> Dict:
        """
        Analyze political bias of a full article.
        
        Args:
            text: Article text to analyze
            show_chunks: Whether to show individual chunk results
            
        Returns:
            Complete bias analysis results
        """
        print(f"\n🔍 Analyzing article ({len(text)} characters)")
        
        # Split into chunks
        chunks = self.chunk_text(text)
        print(f"📄 Split into {len(chunks)} chunks")
        
        chunk_results = []
        all_predictions = []
        
        for i, chunk in enumerate(chunks):
            result = self.analyze_chunk(chunk)
            chunk_results.append(result)
            all_predictions.append(result["prediction"])
            
            if show_chunks:
                print(f"\n📝 Chunk {i+1} ({result['token_count']} tokens):")
                print(f"   Preview: \"{chunk[:80]}...\"")
                print(f"   Prediction: {result['prediction']} ({result['confidence']:.1%})")
                print(f"   Scores: " + 
                      f"Left={result['scores']['Left']:.2f}, " +
                      f"Center={result['scores']['Center']:.2f}, " +
                      f"Right={result['scores']['Right']:.2f}")
        
        # Calculate overall statistics
        prediction_counts = Counter(all_predictions)
        total_chunks = len(chunks)
        
        # Weighted average of probabilities
        avg_scores = {"Left": 0, "Center": 0, "Right": 0}
        for result in chunk_results:
            for label, score in result["scores"].items():
                avg_scores[label] += score / total_chunks
        
        overall_prediction = max(avg_scores.keys(), key=lambda k: avg_scores[k])
        overall_confidence = avg_scores[overall_prediction]
        
        return {
            "text_length": len(text),
            "num_chunks": total_chunks,
            "chunk_results": chunk_results,
            "overall_prediction": overall_prediction,
            "overall_confidence": overall_confidence,
            "average_scores": avg_scores,
            "chunk_distribution": {
                label: {
                    "count": prediction_counts[label],
                    "percentage": prediction_counts[label] / total_chunks * 100
                }
                for label in self.label_names.values()
            }
        }
    
    def print_summary(self, results: Dict):
        """Print a formatted summary of bias analysis results."""
        print(f"\n📊 BIAS ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Overall Prediction: {results['overall_prediction']}")
        print(f"Confidence: {results['overall_confidence']:.1%}")
        
        print(f"\n📈 Average Scores:")
        for label, score in results['average_scores'].items():
            print(f"  {label:>6}: {score:.3f} ({score*100:.1f}%)")
        
        print(f"\n📄 Chunk Distribution:")
        for label, stats in results['chunk_distribution'].items():
            count = stats['count']
            pct = stats['percentage']
            print(f"  {label:>6}: {count}/{results['num_chunks']} chunks ({pct:.1f}%)")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Analyze political bias in news articles")
    parser.add_argument("file_path", help="Path to text file to analyze")
    parser.add_argument("--no-chunks", action="store_true", help="Hide individual chunk results")
    args = parser.parse_args()
    
    try:
        # Load text
        from input_handler import load_text_from_file, get_text_stats
        
        print("🔍 BiasCheck-RoBERTa Political Bias Analyzer")
        print("=" * 50)
        
        text = load_text_from_file(args.file_path)
        stats = get_text_stats(text)
        
        print(f"📄 File: {args.file_path}")
        print(f"📊 Stats: {stats['word_count']} words, {stats['sentence_count']} sentences")
        
        # Initialize analyzer
        analyzer = BiasCheckAnalyzer()
        
        # Analyze
        results = analyzer.analyze_article(text, show_chunks=not args.no_chunks)
        
        # Print summary
        analyzer.print_summary(results)
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 