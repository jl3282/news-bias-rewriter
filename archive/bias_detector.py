#!/usr/bin/env python3
"""
News Bias Detection and Evaluation Tool

This script provides comprehensive bias detection capabilities for news articles using
various pre-trained models. It can analyze political bias, emotional tone, toxicity,
and other forms of bias commonly found in news content.

Based on research from:
- Hugging Face bias detection models 
- Cardiff NLP Twitter RoBERTa models
- Various academic papers on news bias detection
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Core ML libraries  
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    pipeline, logging as transformers_logging
)
from scipy.special import softmax

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress some transformers warnings for cleaner output
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class BiasDetectionSuite:
    """
    A comprehensive suite for detecting various types of bias in news articles.
    
    Supports multiple models and bias types including:
    - Political bias (left/center/right)
    - Emotional sentiment (positive/negative/neutral)
    - Toxicity and hate speech
    - Gender and demographic bias
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the bias detection suite."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Store loaded models to avoid reloading
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Model configurations for different bias types
        self.model_configs = {
            "hate_speech": {
                "model_name": "cardiffnlp/twitter-roberta-base-hate",
                "task": "text-classification",
                "description": "Detects hate speech and toxic content"
            },
            "sentiment": {
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                "task": "text-classification",
                "description": "Analyzes emotional sentiment"
            },
            "offensive": {
                "model_name": "cardiffnlp/twitter-roberta-base-offensive",
                "task": "text-classification", 
                "description": "Detects offensive language"
            },
            "toxicity": {
                "model_name": "martin-ha/toxic-comment-model",
                "task": "text-classification",
                "description": "Comprehensive toxicity detection"
            }
        }
        
    def load_model(self, bias_type: str) -> bool:
        """
        Load a specific bias detection model.
        
        Args:
            bias_type: Type of bias to detect (see model_configs)
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        if bias_type in self.pipelines:
            return True
            
        if bias_type not in self.model_configs:
            logger.error(f"Unknown bias type: {bias_type}")
            return False
            
        config = self.model_configs[bias_type]
        model_name = config["model_name"]
        
        try:
            logger.info(f"Loading {bias_type} model: {model_name}")
            
            # Try to load model with pipeline first (simpler)
            try:
                self.pipelines[bias_type] = pipeline(
                    config["task"],
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info(f"✅ Successfully loaded {bias_type} model")
                return True
                
            except Exception as e:
                logger.warning(f"Pipeline loading failed, trying manual load: {e}")
                
                # Fallback to manual model/tokenizer loading
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.to(self.device)
                
                self.tokenizers[bias_type] = tokenizer
                self.models[bias_type] = model
                
                logger.info(f"✅ Successfully loaded {bias_type} model (manual)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to load {bias_type} model: {e}")
            return False
    
    def analyze_text(self, text: str, bias_types: List[str] = None) -> Dict:
        """
        Analyze text for various types of bias.
        
        Args:
            text: Input text to analyze
            bias_types: List of bias types to check (default: all available)
            
        Returns:
            Dictionary containing bias analysis results
        """
        if bias_types is None:
            bias_types = list(self.model_configs.keys())
            
        results = {
            "text": text[:200] + "..." if len(text) > 200 else text,
            "timestamp": datetime.now().isoformat(),
            "bias_scores": {}
        }
        
        for bias_type in bias_types:
            if not self.load_model(bias_type):
                continue
                
            try:
                if bias_type in self.pipelines:
                    # Use pipeline
                    predictions = self.pipelines[bias_type](text)
                    
                    # Format results
                    scores = {}
                    for pred in predictions[0] if isinstance(predictions[0], list) else predictions:
                        scores[pred['label']] = round(pred['score'], 4)
                        
                else:
                    # Use manual model
                    tokenizer = self.tokenizers[bias_type]
                    model = self.models[bias_type]
                    
                    inputs = tokenizer(text, return_tensors="pt", 
                                     truncation=True, max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probabilities = softmax(outputs.logits.cpu().numpy()[0])
                        
                    # Get label names (model dependent)
                    labels = getattr(model.config, 'id2label', 
                                   {i: f"class_{i}" for i in range(len(probabilities))})
                    
                    scores = {labels[i]: round(float(prob), 4) 
                             for i, prob in enumerate(probabilities)}
                
                results["bias_scores"][bias_type] = {
                    "scores": scores,
                    "prediction": max(scores.keys(), key=lambda k: scores[k]),
                    "confidence": max(scores.values()),
                    "model": self.model_configs[bias_type]["model_name"]
                }
                
            except Exception as e:
                logger.error(f"Error analyzing {bias_type}: {e}")
                results["bias_scores"][bias_type] = {
                    "error": str(e),
                    "model": self.model_configs[bias_type]["model_name"]
                }
        
        return results
    
    def evaluate_accuracy(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate model accuracy on labeled test data.
        
        Args:
            test_data: List of dicts with 'text' and 'labels' keys
            
        Returns:
            Accuracy metrics for each bias type
        """
        results = {"evaluation_metrics": {}}
        
        for bias_type in self.model_configs.keys():
            if not self.load_model(bias_type):
                continue
                
            # Filter test data that has labels for this bias type
            relevant_data = [item for item in test_data 
                           if bias_type in item.get('labels', {})]
            
            if not relevant_data:
                logger.warning(f"No test data available for {bias_type}")
                continue
                
            correct = 0
            total = len(relevant_data)
            predictions = []
            ground_truth = []
            
            logger.info(f"Evaluating {bias_type} on {total} examples...")
            
            for item in tqdm(relevant_data, desc=f"Testing {bias_type}"):
                try:
                    result = self.analyze_text(item['text'], [bias_type])
                    
                    if bias_type in result["bias_scores"] and "prediction" in result["bias_scores"][bias_type]:
                        pred = result["bias_scores"][bias_type]["prediction"]
                        true_label = item['labels'][bias_type]
                        
                        predictions.append(pred)
                        ground_truth.append(true_label)
                        
                        if pred.lower() == true_label.lower():
                            correct += 1
                            
                except Exception as e:
                    logger.error(f"Error evaluating example: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0
            results["evaluation_metrics"][bias_type] = {
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": total,
                "model": self.model_configs[bias_type]["model_name"]
            }
            
            logger.info(f"{bias_type} accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return results
    
    def batch_analyze(self, texts: List[str], bias_types: List[str] = None) -> List[Dict]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            bias_types: List of bias types to check
            
        Returns:
            List of analysis results for each text
        """
        results = []
        
        logger.info(f"Analyzing {len(texts)} texts...")
        for text in tqdm(texts, desc="Analyzing texts"):
            result = self.analyze_text(text, bias_types)
            results.append(result)
            
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about available models."""
        return {
            "available_models": self.model_configs,
            "loaded_models": list(self.pipelines.keys()) + list(self.models.keys()),
            "device": self.device,
            "torch_version": torch.__version__,
            "transformers_version": getattr(__import__('transformers'), '__version__', 'unknown')
        }


def create_sample_test_data() -> List[Dict]:
    """
    Create sample test data for evaluating bias detection.
    
    Returns:
        List of test examples with ground truth labels
    """
    return [
        {
            "text": "This is a wonderful article about economic policy.",
            "labels": {
                "sentiment": "POSITIVE",
                "hate_speech": "NOT_HATE",
                "offensive": "NOT_OFFENSIVE"
            }
        },
        {
            "text": "I hate this stupid policy and everyone who supports it.",
            "labels": {
                "sentiment": "NEGATIVE", 
                "hate_speech": "HATE",
                "offensive": "OFFENSIVE"
            }
        },
        {
            "text": "The report provides a neutral analysis of the economic situation.",
            "labels": {
                "sentiment": "NEUTRAL",
                "hate_speech": "NOT_HATE", 
                "offensive": "NOT_OFFENSIVE"
            }
        }
    ]


def main():
    """Main function demonstrating the bias detection suite."""
    print("🔍 News Bias Detection Suite")
    print("=" * 50)
    
    # Initialize the bias detection suite
    detector = BiasDetectionSuite()
    
    # Show model information
    info = detector.get_model_info()
    print(f"Device: {info['device']}")
    print(f"PyTorch: {info['torch_version']}")
    print(f"Transformers: {info['transformers_version']}")
    print()
    
    # Example analysis using the example.txt file if available
    example_file = "example.txt"
    if os.path.exists(example_file):
        print("📰 Analyzing example article...")
        try:
            from input_handler import load_text_from_file, get_text_stats
            
            # Load the article
            article_text = load_text_from_file(example_file)
            stats = get_text_stats(article_text)
            
            print(f"Article stats: {stats['word_count']} words, {stats['sentence_count']} sentences")
            
            # Analyze the article
            results = detector.analyze_text(article_text)
            
            print("\n📊 Bias Analysis Results:")
            print("-" * 30)
            
            for bias_type, analysis in results["bias_scores"].items():
                if "error" in analysis:
                    print(f"❌ {bias_type}: Error - {analysis['error']}")
                else:
                    pred = analysis["prediction"]
                    conf = analysis["confidence"]
                    print(f"✅ {bias_type}: {pred} (confidence: {conf:.3f})")
                    
                    # Show detailed scores
                    print(f"   Detailed scores: {analysis['scores']}")
                    print()
            
        except ImportError:
            print("❌ Could not import input_handler. Please ensure it's in the same directory.")
        except Exception as e:
            print(f"❌ Error analyzing example file: {e}")
    else:
        print("ℹ️  No example.txt found. Skipping file analysis.")
    
    # Test with sample data
    print("\n🧪 Testing with sample data...")
    test_data = create_sample_test_data()
    
    # Evaluate accuracy
    eval_results = detector.evaluate_accuracy(test_data)
    
    print("\n📈 Accuracy Results:")
    print("-" * 30)
    for bias_type, metrics in eval_results["evaluation_metrics"].items():
        accuracy = metrics["accuracy"]
        correct = metrics["correct"]
        total = metrics["total"]
        print(f"{bias_type}: {accuracy:.2%} ({correct}/{total})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"bias_analysis_results_{timestamp}.json"
    
    combined_results = {
        "model_info": info,
        "evaluation_results": eval_results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main() 